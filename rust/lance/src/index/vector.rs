// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::sync::Arc;
use std::{any::Any, collections::HashMap};

pub mod dedup;
pub(crate) mod details;
pub mod hamming;
pub mod ivf;
pub mod pq;
pub mod utils;

use self::{ivf::*, pq::PQIndex};
use arrow_schema::Schema;
use datafusion_physical_plan::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_index::frag_reuse::CompactFragReuseIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::vector::bq::{RQBuildParams, RQRotationType};
use lance_index::vector::ivf::storage::IvfModel;
use object_store::path::Path;

use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::{
    VectorIndex,
    hnsw::{
        builder::HnswBuildParams,
        index::{HNSWIndex, HNSWIndexOptions},
    },
    ivf::IvfBuildParams,
    pq::PQBuildParams,
    sq::{ScalarQuantizer, builder::SQBuildParams},
};
use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, IndexType, pb};
use lance_io::object_reader::LazyReader;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Reader;
use lance_linalg::distance::*;
use lance_table::format::IndexMetadata;
use serde::Serialize;
use tracing::instrument;
use uuid::Uuid;

use super::IndexParams;
use crate::{Error, Result, dataset::Dataset};

use lance_index::pb::vector_index_stage::Stage;

pub const LANCE_VECTOR_INDEX: &str = "__lance_vector_index";

/// A materialized snapshot of one logical vector index and all of its segments.
#[derive(Debug)]
pub struct LogicalVectorIndex {
    name: String,
    column: String,
    segments: Vec<(IndexMetadata, Arc<dyn VectorIndex>)>,
}

/// An IVF-specific inspection and maintenance view over a [`LogicalVectorIndex`].
///
/// Callers must explicitly opt into this view before accessing partition-level
/// APIs so that IVF semantics stay separated from the generic logical vector
/// index abstraction.
#[derive(Clone, Copy, Debug)]
pub struct LogicalIvfView<'a> {
    logical_index: &'a LogicalVectorIndex,
}

impl LogicalVectorIndex {
    pub(crate) fn try_new(
        name: String,
        column: String,
        segments: Vec<(IndexMetadata, Arc<dyn VectorIndex>)>,
    ) -> Result<Self> {
        if segments.is_empty() {
            return Err(Error::invalid_input(format!(
                "LogicalVectorIndex '{}' on column '{}' must contain at least one segment",
                name, column
            )));
        }

        Ok(Self {
            name,
            column,
            segments,
        })
    }

    /// Returns the logical index name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the vector column that this logical index is built on.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// Returns the number of physical segments in this logical index.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Returns the committed metadata for all physical segments.
    pub fn metadatas(&self) -> impl ExactSizeIterator<Item = &IndexMetadata> + '_ {
        self.segments.iter().map(|(metadata, _)| metadata)
    }

    /// Returns the indexed row count for each segment.
    pub fn num_rows_per_segment(&self) -> Vec<(Uuid, u64)> {
        self.segments
            .iter()
            .map(|(metadata, index)| (metadata.uuid, index.num_rows()))
            .collect()
    }

    /// Returns an IVF view over this logical index.
    ///
    /// This is currently fallible only to leave room for future non-IVF vector
    /// index families. All vector indices currently opened through this path are
    /// IVF-backed.
    pub fn as_ivf(&self) -> Result<LogicalIvfView<'_>> {
        Ok(LogicalIvfView {
            logical_index: self,
        })
    }

    pub(crate) fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = (&IndexMetadata, &Arc<dyn VectorIndex>)> + '_ {
        self.segments
            .iter()
            .map(|(metadata, index)| (metadata, index))
    }
}

impl<'a> LogicalIvfView<'a> {
    pub(crate) fn indices(&self) -> impl ExactSizeIterator<Item = &Arc<dyn VectorIndex>> + '_ {
        self.logical_index.iter().map(|(_, index)| index)
    }

    /// Returns the partition count for each segment in this IVF index.
    pub fn num_partitions_per_segment(&self) -> Vec<(Uuid, usize)> {
        self.logical_index
            .iter()
            .map(|(metadata, index)| (metadata.uuid, index.ivf_model().num_partitions()))
            .collect()
    }

    /// Returns the partition sizes for each segment in this IVF index.
    pub fn partition_sizes(&self) -> Vec<(Uuid, Vec<usize>)> {
        let mut partition_sizes = Vec::with_capacity(self.logical_index.num_segments());
        for (metadata, index) in self.logical_index.iter() {
            let num_partitions = index.ivf_model().num_partitions();
            let mut sizes = Vec::with_capacity(num_partitions);
            for partition_id in 0..num_partitions {
                sizes.push(index.partition_size(partition_id));
            }
            partition_sizes.push((metadata.uuid, sizes));
        }
        partition_sizes
    }

    /// Reads one IVF partition across all segments in this logical index.
    ///
    /// The returned stream preserves segment boundaries only implicitly through
    /// concatenation order; callers should treat it as a merged partition view.
    pub async fn read_partition(
        &self,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let mut schema: Option<Arc<Schema>> = None;
        let mut partition_streams = Vec::with_capacity(self.logical_index.num_segments());
        for index in self.indices() {
            let stream = index
                .partition_reader(partition_id, with_vector, &NoOpMetricsCollector)
                .await?;
            if schema.is_none() {
                schema = Some(stream.schema());
            }
            partition_streams.push(stream);
        }

        match schema {
            Some(schema) => {
                let merged = stream::select_all(partition_streams);
                let stream = RecordBatchStreamAdapter::new(schema, merged);
                Ok(Box::pin(stream))
            }
            None => Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                stream::empty(),
            ))),
        }
    }
}

/// Parameters of each index stage.
#[derive(Debug, Clone)]
pub enum StageParams {
    Ivf(IvfBuildParams),
    Hnsw(HnswBuildParams),
    PQ(PQBuildParams),
    SQ(SQBuildParams),
    RQ(RQBuildParams),
}

// The version of the index file.
// `Legacy` is used for only IVF_PQ index, and is the default value.
// The other index types are using `V3`.
#[derive(Debug, Clone, Serialize)]
pub enum IndexFileVersion {
    Legacy,
    V3,
}

impl IndexFileVersion {
    pub fn try_from(version: &str) -> Result<Self> {
        match version.to_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "v3" => Ok(Self::V3),
            _ => Err(Error::index(format!(
                "Invalid index file version: {}",
                version
            ))),
        }
    }
}

/// The parameters to build vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexParams {
    pub stages: Vec<StageParams>,

    /// Vector distance metrics type.
    pub metric_type: MetricType,

    /// The version of the index file.
    pub version: IndexFileVersion,

    /// Skip transpose / packing for PQ and RQ storage.
    pub skip_transpose: bool,

    /// Runtime hints: optional build preferences stored in the index manifest.
    /// Keys use reverse-DNS namespacing (e.g., "lance.ivf.max_iters").
    /// Populated by the build path and merged into VectorIndexDetails at creation time.
    pub runtime_hints: HashMap<String, String>,
}

impl VectorIndexParams {
    pub fn version(&mut self, version: IndexFileVersion) -> &mut Self {
        self.version = version;
        self
    }

    pub fn skip_transpose(&mut self, skip_transpose: bool) -> &mut Self {
        self.skip_transpose = skip_transpose;
        self
    }

    pub fn ivf_flat(num_partitions: usize, metric_type: MetricType) -> Self {
        let ivf_params = IvfBuildParams::new(num_partitions);
        let stages = vec![StageParams::Ivf(ivf_params)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn with_ivf_flat_params(metric_type: MetricType, ivf: IvfBuildParams) -> Self {
        let stages = vec![StageParams::Ivf(ivf)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    /// Create index parameters for `IVF_PQ` index.
    ///
    /// Parameters
    ///
    ///  - `num_partitions`: the number of IVF partitions.
    ///  - `num_bits`: the number of bits to present the centroids used in PQ. Can only be `8` for now.
    ///  - `num_sub_vectors`: the number of sub vectors used in PQ.
    ///  - `metric_type`: how to compute distance, i.e., `L2` or `Cosine`.
    pub fn ivf_pq(
        num_partitions: usize,
        num_bits: u8,
        num_sub_vectors: usize,
        metric_type: MetricType,
        max_iterations: usize,
    ) -> Self {
        let mut stages: Vec<StageParams> = vec![];
        stages.push(StageParams::Ivf(IvfBuildParams::new(num_partitions)));

        let pq_params = PQBuildParams {
            num_bits: num_bits as usize,
            num_sub_vectors,
            max_iters: max_iterations,
            ..Default::default()
        };
        stages.push(StageParams::PQ(pq_params));

        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn ivf_rq(num_partitions: usize, num_bits: u8, distance_type: DistanceType) -> Self {
        Self::ivf_rq_with_rotation(
            num_partitions,
            num_bits,
            distance_type,
            RQRotationType::default(),
        )
    }

    pub fn ivf_rq_with_rotation(
        num_partitions: usize,
        num_bits: u8,
        distance_type: DistanceType,
        rotation_type: RQRotationType,
    ) -> Self {
        let ivf = IvfBuildParams::new(num_partitions);
        let rq = RQBuildParams::with_rotation_type(num_bits, rotation_type);
        let stages = vec![StageParams::Ivf(ivf), StageParams::RQ(rq)];
        Self {
            stages,
            metric_type: distance_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    /// Create index parameters with `IVF` and `PQ` parameters, respectively.
    pub fn with_ivf_pq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        pq: PQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::PQ(pq)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn with_ivf_sq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        sq: SQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::SQ(sq)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn with_ivf_rq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        rq: RQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::RQ(rq)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn ivf_hnsw(
        distance_type: DistanceType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::Hnsw(hnsw)];
        Self {
            stages,
            metric_type: distance_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    /// Create index parameters with `IVF`, `PQ` and `HNSW` parameters, respectively.
    /// This is used for `IVF_HNSW_PQ` index.
    pub fn with_ivf_hnsw_pq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
        pq: PQBuildParams,
    ) -> Self {
        let stages = vec![
            StageParams::Ivf(ivf),
            StageParams::Hnsw(hnsw),
            StageParams::PQ(pq),
        ];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    /// Create index parameters with `IVF`, `HNSW` and `SQ` parameters, respectively.
    /// This is used for `IVF_HNSW_SQ` index.
    pub fn with_ivf_hnsw_sq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
        sq: SQBuildParams,
    ) -> Self {
        let stages = vec![
            StageParams::Ivf(ivf),
            StageParams::Hnsw(hnsw),
            StageParams::SQ(sq),
        ];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    /// Create index parameters with `IVF`, `HNSW` and `RQ` parameters.
    pub fn with_ivf_hnsw_rq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
        rq: RQBuildParams,
    ) -> Self {
        let stages = vec![
            StageParams::Ivf(ivf),
            StageParams::Hnsw(hnsw),
            StageParams::RQ(rq),
        ];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }

    pub fn index_type(&self) -> IndexType {
        let len = self.stages.len();
        match (len, self.stages.get(1), self.stages.last()) {
            (0, _, _) => IndexType::Vector,
            (1, _, Some(StageParams::Ivf(_))) => IndexType::IvfFlat,
            (2, _, Some(StageParams::PQ(_))) => IndexType::IvfPq,
            (2, _, Some(StageParams::SQ(_))) => IndexType::IvfSq,
            (2, _, Some(StageParams::RQ(_))) => IndexType::IvfRq,
            (2, _, Some(StageParams::Hnsw(_))) => IndexType::IvfHnswFlat,
            (3, Some(StageParams::Hnsw(_)), Some(StageParams::PQ(_))) => IndexType::IvfHnswPq,
            (3, Some(StageParams::Hnsw(_)), Some(StageParams::SQ(_))) => IndexType::IvfHnswSq,
            (3, Some(StageParams::Hnsw(_)), Some(StageParams::RQ(_))) => IndexType::IvfHnswRq,
            _ => IndexType::Vector,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_name(&self) -> &str {
        LANCE_VECTOR_INDEX
    }
}

/// Rows a PQ codebook needs before it can be trained.
///
/// One row per code, which is `2^num_bits` — 256 at the default 8 bits, and not
/// 256 at any other setting. `None` when the width does not fit a `usize`.
pub fn pq_quantizer_minimum_rows(num_bits: usize) -> Option<usize> {
    1_usize.checked_shl(num_bits as u32)
}

/// Open the Vector index on dataset, specified by the `uuid`.
#[instrument(level = "debug", skip(dataset, vec_idx, reader))]
pub(crate) async fn open_vector_index(
    dataset: Arc<Dataset>,
    uuid: &Uuid,
    vec_idx: &lance_index::pb::VectorIndex,
    reader: Arc<dyn Reader>,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
) -> Result<Arc<dyn VectorIndex>> {
    let metric_type = pb::VectorMetricType::try_from(vec_idx.metric_type)?.into();

    let mut last_stage: Option<Arc<dyn VectorIndex>> = None;

    let frag_reuse_uuid = crate::index::loading::frag_reuse_cache_scope(&dataset).await;

    for stg in vec_idx.stages.iter().rev() {
        match stg.stage.as_ref() {
            #[allow(unused_variables)]
            Some(Stage::Transform(tf)) => {
                if last_stage.is_none() {
                    return Err(Error::index(format!(
                        "Invalid vector index stages: {:?}",
                        vec_idx.stages
                    )));
                }
            }
            Some(Stage::Ivf(ivf_pb)) => {
                if last_stage.is_none() {
                    return Err(Error::index(format!(
                        "Invalid vector index stages: {:?}",
                        vec_idx.stages
                    )));
                }
                let ivf = IvfModel::try_from(ivf_pb.to_owned())?;
                last_stage = Some(Arc::new(IVFIndex::try_new(
                    *uuid,
                    ivf,
                    reader.clone(),
                    last_stage.unwrap(),
                    metric_type,
                    dataset
                        .index_cache
                        .for_index(uuid, frag_reuse_uuid.as_ref()),
                )?));
            }
            Some(Stage::Pq(pq_proto)) => {
                if last_stage.is_some() {
                    return Err(Error::index(format!(
                        "Invalid vector index stages: {:?}",
                        vec_idx.stages
                    )));
                };
                let pq = ProductQuantizer::from_proto(pq_proto, metric_type)?;
                last_stage = Some(Arc::new(PQIndex::new(
                    pq,
                    metric_type,
                    frag_reuse_index.clone(),
                )));
            }
            Some(Stage::Diskann(_)) => {
                return Err(Error::index(
                    "DiskANN support is removed from Lance.".to_string(),
                ));
            }
            _ => {}
        }
    }

    if last_stage.is_none() {
        return Err(Error::index(format!(
            "Invalid index stages: {:?}",
            vec_idx.stages
        )));
    }
    let idx = last_stage.unwrap();
    Ok(idx)
}

/// Open an index file without a HEAD request when the size is already known.
///
/// `file_sizes` maps a file name to its size in bytes (see
/// `IndexMetadata::file_size_map`). If `file_name` is missing, which is the case
/// for older indices that did not record sizes, this falls back to `open`, which
/// issues a HEAD to learn the size.
pub(crate) async fn open_index_file(
    object_store: &ObjectStore,
    path: &Path,
    file_name: &str,
    file_sizes: &HashMap<String, u64>,
) -> Result<Box<dyn Reader>> {
    match file_sizes.get(file_name) {
        Some(&size) => object_store.open_with_size(path, size as usize).await,
        None => object_store.open(path).await,
    }
}

/// Open an index file for storage inside a long-lived cache entry.
///
/// On local stores a reader keeps its file open, which pins the inode even
/// after cleanup unlinks it. Legacy vector indexes are cached whole, so such a
/// cache entry would hold a descriptor for every index it has seen. Those
/// stores get a [`LazyReader`] instead, which reopens the file per read.
///
/// Other stores hold no descriptor, and their readers cache small files in
/// memory, so they keep the regular reader.
pub(crate) async fn open_cache_reader(
    object_store: Arc<ObjectStore>,
    path: &Path,
    file_name: &str,
    file_sizes: &HashMap<String, u64>,
) -> Result<Arc<dyn Reader>> {
    if object_store.scheme().starts_with("file") {
        let known_size = file_sizes.get(file_name).map(|&size| size as usize);
        Ok(Arc::new(LazyReader::new(
            object_store,
            path.clone(),
            known_size,
        )))
    } else {
        Ok(
            open_index_file(object_store.as_ref(), path, file_name, file_sizes)
                .await?
                .into(),
        )
    }
}

#[instrument(
    level = "debug",
    skip(dataset, index_meta, reader),
    fields(uuid = %index_meta.uuid)
)]
pub(crate) async fn open_vector_index_v2(
    dataset: Arc<Dataset>,
    column: &str,
    index_meta: &IndexMetadata,
    reader: V1FileReader,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
) -> Result<Arc<dyn VectorIndex>> {
    let uuid = &index_meta.uuid;
    let index_metadata = reader
        .schema()
        .metadata
        .get(INDEX_METADATA_SCHEMA_KEY)
        .ok_or(Error::index("Index Metadata not found".to_owned()))?;
    let index_metadata: lance_index::IndexMetadata = serde_json::from_str(index_metadata)?;
    let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

    let frag_reuse_uuid = crate::index::loading::frag_reuse_cache_scope(&dataset).await;
    // From the caller's metadata, not the listing: a segment a commit is
    // about to publish is not listed yet.
    let index_dir = dataset.indice_files_dir(index_meta)?;
    let object_store = dataset.object_store_for_index(index_meta).await?;
    let file_sizes = index_meta.file_size_map();

    let index: Arc<dyn VectorIndex> = match index_metadata.index_type.as_str() {
        "IVF_HNSW_PQ" => {
            let aux_path = index_dir
                .clone()
                .join(uuid.to_string())
                .join(INDEX_AUXILIARY_FILE_NAME);
            // The cached index keeps this reader alive, so it must not pin an
            // open descriptor (see [`open_cache_reader`]).
            let aux_reader = open_cache_reader(
                object_store.clone(),
                &aux_path,
                INDEX_AUXILIARY_FILE_NAME,
                &file_sizes,
            )
            .await?;

            let ivf_data = IvfModel::load(&reader).await?;
            let options = HNSWIndexOptions { use_residual: true };
            let hnsw = HNSWIndex::<ProductQuantizer>::try_new(
                reader.object_reader.clone(),
                aux_reader,
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = IvfModel::try_from(pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                *uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
                dataset
                    .index_cache
                    .for_index(uuid, frag_reuse_uuid.as_ref()),
            )?)
        }

        "IVF_HNSW_SQ" => {
            let aux_path = index_dir
                .clone()
                .join(uuid.to_string())
                .join(INDEX_AUXILIARY_FILE_NAME);
            let aux_reader = open_cache_reader(
                object_store.clone(),
                &aux_path,
                INDEX_AUXILIARY_FILE_NAME,
                &file_sizes,
            )
            .await?;

            let ivf_data = IvfModel::load(&reader).await?;
            let options = HNSWIndexOptions {
                use_residual: false,
            };

            let hnsw = HNSWIndex::<ScalarQuantizer>::try_new(
                reader.object_reader.clone(),
                aux_reader,
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = IvfModel::try_from(pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                *uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
                dataset
                    .index_cache
                    .for_index(uuid, frag_reuse_uuid.as_ref()),
            )?)
        }

        index_type => {
            if let Some(ext) = dataset
                .session
                .index_extensions
                .get(&(IndexType::Vector, index_type.to_string()))
            {
                ext.clone()
                    .to_vector()
                    .ok_or(Error::internal(
                        "unable to cast index extension to vector".to_string(),
                    ))?
                    .load_index(dataset.clone(), column, uuid, reader)
                    .await?
            } else {
                return Err(Error::index(format!(
                    "Unsupported index type: {}",
                    index_metadata.index_type
                )));
            }
        }
    };

    Ok(index)
}

#[cfg(any())]
#[rustfmt::skip]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, WriteParams};
    use crate::index::DatasetIndexExt;
    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_array::{
        Array, FixedSizeListArray, Float16Array, Float32Array, Int32Array, RecordBatch,
        RecordBatchIterator,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
    use lance_core::utils::row_addr_remap::GroupInput;
    use lance_core::utils::tempfile::{TempStdDir, TempStrDir};
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_file::writer::FileWriterOptions;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::vector::ivf::builder::IvfBuildParams;
    use lance_linalg::distance::MetricType;
    use roaring::RoaringTreemap;

    #[tokio::test]
    async fn test_nullable_vector_count_stops_at_threshold() {
        const ROWS: usize = 16 * 1024;
        const DIM: usize = 16;
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values((0..ROWS * DIM).map(|i| i as f32)),
            DIM as i32,
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            vectors.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 1024,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 16);

        dataset.table_state.object_store.io_stats_incremental();
        let counted = count_trainable_vectors(&dataset, "vec", ROWS, 256)
            .await
            .unwrap();
        let bounded_reads = dataset.table_state.object_store.io_stats_incremental();
        assert!((256..ROWS).contains(&counted), "counted {counted}");

        assert_eq!(
            count_trainable_vectors(&dataset, "vec", ROWS, usize::MAX)
                .await
                .unwrap(),
            ROWS
        );
        let full_reads = dataset.table_state.object_store.io_stats_incremental();
        assert!(
            bounded_reads.read_bytes < full_reads.read_bytes / 2,
            "threshold check read {} bytes; full count read {} bytes",
            bounded_reads.read_bytes,
            full_reads.read_bytes
        );
    }

    #[rstest::rstest]
    #[case::all_valid(64, 1)]
    #[case::sparse(64, 4)]
    #[case::all_null(64, 0)]
    #[case::empty(0, 1)]
    #[tokio::test]
    async fn test_nullable_vector_count_nulls_and_deletions(
        #[case] rows: usize,
        #[case] valid_every: usize,
    ) {
        let valid: Vec<bool> = (0..rows)
            .map(|i| valid_every != 0 && i % valid_every == 0)
            .collect();
        let vectors = FixedSizeListArray::new(
            Arc::new(Field::new("item", ArrowDataType::Float32, false)),
            2,
            Arc::new(Float32Array::from_iter_values(
                (0..rows * 2).map(|i| i as f32),
            )),
            Some(NullBuffer::from(valid.clone())),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", ArrowDataType::Int32, false),
            Field::new("vec", vectors.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..rows as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 16,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        if rows > 0 {
            dataset.delete("id % 5 = 0").await.unwrap();
        }
        let expected = valid
            .iter()
            .enumerate()
            .filter(|(i, valid)| **valid && i % 5 != 0)
            .count();
        let total = dataset.count_rows(None).await.unwrap();
        for enough in [0, 1, 8, usize::MAX] {
            let count = count_trainable_vectors(&dataset, "vec", total, enough)
                .await
                .unwrap();
            if expected < enough {
                assert_eq!(count, expected);
            } else {
                assert!(count >= enough && count <= expected);
            }
            assert_eq!(
                has_vectors_to_train(&dataset, "vec", enough).await.unwrap(),
                expected >= enough
            );
        }
    }

    /// IVF_RQ encodes residuals under L2 or dot; the RQ transform rejects other
    /// metrics, but only once the IVF model has been trained, so the build spends
    /// the training pass before failing. Hamming is also contradictory for RQ,
    /// which requires float vectors while Hamming is a binary-vector metric.
    #[rstest::rstest]
    #[tokio::test]
    async fn test_build_rejects_hamming_for_rq(#[values(false, true)] with_hnsw: bool) {
        use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
        use lance_index::vector::bq::RQBuildParams;

        let dim = 16;
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(dim)),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(256))
            .await
            .unwrap();

        let (ivf, rq) = (IvfBuildParams::new(2), RQBuildParams::default());
        let params = if with_hnsw {
            let hnsw = HnswBuildParams::default();
            VectorIndexParams::with_ivf_hnsw_rq_params(MetricType::Hamming, ivf, hnsw, rq)
        } else {
            VectorIndexParams::with_ivf_rq_params(MetricType::Hamming, ivf, rq)
        };
        let index_type = params.index_type();
        let err = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains(&format!("{index_type} does not support the hamming metric")),
            "expected the up-front metric rejection, got: {err}"
        );
    }

    /// A build that was handed its codebook has nothing to fit, so the rows
    /// that fitting would have needed are not required of it.
    ///
    /// Without this a caller who supplies both the centroids and the codebook
    /// -- a fully specified index, needing no training data at all -- has the
    /// build deferred and gets an empty index back.
    #[test]
    fn a_supplied_codebook_needs_no_rows_to_train_on() {
        use arrow_array::{FixedSizeListArray, Float32Array};
        use lance_index::vector::pq::PQBuildParams;

        let eight_bit = PQBuildParams {
            num_bits: 8,
            ..Default::default()
        };
        assert_eq!(
            vector_quantizer_minimum_rows(&[StageParams::PQ(eight_bit.clone())]),
            Some(256),
            "fitting an 8-bit codebook needs a row per code"
        );

        let values = Float32Array::from(vec![0.0_f32; 2 * 256 * 4]);
        let codebook = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let supplied = PQBuildParams {
            codebook: Some(Arc::new(codebook)),
            ..eight_bit
        };
        assert_eq!(
            vector_quantizer_minimum_rows(&[StageParams::PQ(supplied)]),
            None,
            "a supplied codebook is the fit, so no rows are needed for it"
        );
    }

    /// A null vector inside a multivector row is not one to train on.
    ///
    /// The row's list length counts it, so counting lengths reports more
    /// trainable vectors than exist -- and this count decides whether there is
    /// enough data to train at all, so erring high is the direction that
    /// trains an index it should have deferred.
    #[tokio::test]
    async fn a_null_vector_inside_a_multivector_row_is_not_trainable() {
        use arrow_array::RecordBatchIterator;
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder};

        const DIM: i32 = 2;
        let item = Arc::new(Field::new("item", ArrowDataType::Float32, true));
        let vector = Arc::new(Field::new(
            "item",
            ArrowDataType::FixedSizeList(item, DIM),
            true,
        ));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            ArrowDataType::List(vector),
            true,
        )]));

        // Three rows: two vectors, then one vector and one null, then a null
        // list. Four elements are present; three of them are trainable.
        let mut builder = ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), DIM));
        for row in [
            vec![Some([1.0, 2.0]), Some([3.0, 4.0])],
            vec![Some([5.0, 6.0]), None],
        ] {
            for cell in row {
                match cell {
                    Some(v) => {
                        builder.values().values().append_slice(&v);
                        builder.values().append(true);
                    }
                    None => {
                        builder.values().values().append_slice(&[0.0; DIM as usize]);
                        builder.values().append(false);
                    }
                }
            }
            builder.append(true);
        }
        builder.append(false);
        let array = builder.finish();

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, dir.as_str(), None).await.unwrap();

        let counted = count_trainable_vectors(&dataset, "vec", 3, usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            counted, 3,
            "the null vector in the second row must not be counted"
        );
    }

    #[test]
    fn test_remap_for_owned_fragments_drops_stale_rows() {
        const NUM_OWNED_ROWS: u32 = 1_048_576;
        let owned_addr = |offset| u64::from(RowAddress::new_from_parts(1, offset));
        let moved_addr = |offset| u64::from(RowAddress::new_from_parts(4, offset));
        let stale_addr = u64::from(RowAddress::new_from_parts(2, 0));
        let mapping = Arc::new(
            RowAddrRemap::compact([GroupInput {
                rewritten_old_row_addrs: RoaringTreemap::from_iter(
                    (0..NUM_OWNED_ROWS).map(owned_addr),
                ),
                old_frag_ids: vec![1],
                new_frags: vec![(4, NUM_OWNED_ROWS)],
            }])
            .unwrap(),
        );

        let filtered = remap_for_owned_fragments(
            mapping,
            (0..NUM_OWNED_ROWS)
                .map(owned_addr)
                .chain(std::iter::once(stale_addr)),
            &RoaringBitmap::from_iter([1]),
        );

        assert!(matches!(filtered, RowAddrRemap::Compact(_)));
        assert_eq!(filtered.get(owned_addr(0)), Some(Some(moved_addr(0))));
        assert_eq!(
            filtered.get(owned_addr(NUM_OWNED_ROWS - 1)),
            Some(Some(moved_addr(NUM_OWNED_ROWS - 1)))
        );
        assert_eq!(filtered.get(stale_addr), Some(None));
    }

    /// `open_index_file` skips the HEAD when the size is known and still falls
    /// back to a HEAD for older indices that did not record sizes. A HEAD is
    /// issued as a `get_opts` call with `head = true`, so a proxy store counts
    /// those against the index file.
    ///
    /// Regression test for <https://github.com/lance-format/lance/issues/6944>.
    #[tokio::test]
    async fn test_open_index_file_skips_head_when_size_known() {
        use lance_index::INDEX_FILE_NAME;
        use lance_io::assert_io_eq;
        use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};

        let (store, base) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();

        let path = base.join(INDEX_FILE_NAME);
        // Larger than the block size so size discovery needs a separate HEAD.
        let data = vec![7u8; 2 * store.block_size()];
        store.put(&path, &data).await.unwrap();

        let file_sizes = HashMap::from([(INDEX_FILE_NAME.to_string(), data.len() as u64)]);

        // Size recorded in the manifest, so reading the size issues no HEAD.
        let _ = store.io_stats_incremental(); // reset
        let reader = open_index_file(store.as_ref(), &path, INDEX_FILE_NAME, &file_sizes)
            .await
            .unwrap();
        assert_eq!(reader.size().await.unwrap(), data.len());
        let stats = store.io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "a known file size must not trigger a HEAD request"
        );

        // Size unknown, as in an older index, so it falls back to a HEAD.
        let _ = store.io_stats_incremental(); // reset
        let reader = open_index_file(store.as_ref(), &path, INDEX_FILE_NAME, &HashMap::new())
            .await
            .unwrap();
        assert_eq!(reader.size().await.unwrap(), data.len());
        let stats = store.io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            1,
            "an unknown file size must fall back to exactly one HEAD request"
        );
    }

    /// `open_index_file` looks up sizes in `IndexMetadata::file_size_map()` by
    /// bare file name. This pins that a freshly created HNSW index records both
    /// the main and auxiliary files under those exact names with nonzero sizes,
    /// which is what lets the open path skip the HEAD.
    #[tokio::test]
    async fn test_hnsw_index_records_file_sizes() {
        use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME};

        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());

        let reader = lance_datagen::gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(64), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            IvfBuildParams {
                num_partitions: Some(2),
                max_iters: 2,
                sample_rate: 2,
                ..Default::default()
            },
            HnswBuildParams::default()
                .max_level(2)
                .num_edges(4)
                .ef_construction(16),
            PQBuildParams {
                num_sub_vectors: 2,
                num_bits: 4,
                max_iters: 2,
                sample_rate: 2,
                ..Default::default()
            },
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("hnsw".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let index = indices.iter().find(|idx| idx.name == "hnsw").unwrap();
        let file_sizes = index.file_size_map();

        assert!(
            file_sizes.get(INDEX_FILE_NAME).copied().unwrap_or(0) > 0,
            "manifest should record a nonzero {INDEX_FILE_NAME} size, got {file_sizes:?}"
        );
        assert!(
            file_sizes
                .get(INDEX_AUXILIARY_FILE_NAME)
                .copied()
                .unwrap_or(0)
                > 0,
            "manifest should record a nonzero {INDEX_AUXILIARY_FILE_NAME} size, got {file_sizes:?}"
        );
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_pq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Ten partitions, each needing a codebook's worth of vectors.
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(2560), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_PQ index on source
        let params = VectorIndexParams::ivf_pq(10, 8, 16, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_pq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_PQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_pq",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_PQ"),
            "Index type should be IVF_PQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("l2"),
            "Metric type should be L2"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(10),
            "Should have 10 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(10),
            "Should have 10 partitions as configured"
        );

        // Verify PQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_pq: ProductQuantizer = source_quantizer.try_into().unwrap();
        let target_pq: ProductQuantizer = target_quantizer.try_into().unwrap();

        let source_pq_params = derive_pq_params(&source_pq);
        let target_pq_params = derive_pq_params(&target_pq);

        assert_eq!(
            source_pq_params.num_sub_vectors, target_pq_params.num_sub_vectors,
            "PQ num_sub_vectors should match"
        );
        assert_eq!(
            source_pq_params.num_bits, target_pq_params.num_bits,
            "PQ num_bits should match"
        );
        assert_eq!(
            target_pq_params.num_sub_vectors, 16,
            "PQ should have 16 sub vectors"
        );
        assert_eq!(target_pq_params.num_bits, 8, "PQ should use 8 bits");

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 10, "Should return 10 nearest neighbors");
    }

    /// The scratch directory the guard owns has to be gone once the build is
    /// over, or the OS temp dir grows by one shuffled copy of the vector column
    /// per index build.
    ///
    /// The OS temp dir is process-global, so an in-process check cannot
    /// attribute a leftover directory to our own build. This test re-executes
    /// itself in a child process with its temp dir pointed at an isolated dir we
    /// own: the child builds, the parent asserts nothing survives. Same shape as
    /// `index::vector::ivf::io::tests::test_hnsw_pq_scratch_dir_is_not_leaked`,
    /// which covers the legacy partition-staging dir.
    #[test]
    fn test_shuffle_scratch_dir_is_not_leaked() {
        const ROOT_VAR: &str = "LANCE_SHUFFLE_LEAK_TEST_ROOT";

        // Child half: build under the root the parent handed us and let it do the
        // leak detection. The dataset goes outside the temp dir's `.tmp*` namespace
        // so the parent never mistakes it for a leaked scratch directory. Read the
        // value as an `OsString`: with `env::var`, a root that is not valid UTF-8
        // would send the child down the parent branch and have it spawn a child of
        // its own, without end.
        if let Some(root) = std::env::var_os(ROOT_VAR) {
            let root = root
                .into_string()
                .expect("the isolated root must be valid UTF-8");

            // The parent can only scan the directory it owns, so if its temp-dir env
            // vars did not take effect the scan is vacuous and passes no matter what
            // the build does. Fail here instead of reporting a clean run.
            let temp_dir = std::env::temp_dir();
            let canonical = |path: &std::path::Path| {
                path.canonicalize()
                    .unwrap_or_else(|e| panic!("cannot canonicalize {path:?}: {e}"))
            };
            assert_eq!(
                canonical(&temp_dir),
                canonical(std::path::Path::new(&root)),
                "the temp dir was not redirected to the isolated root; temp_dir() is \
                 {temp_dir:?}, root is {root:?}. On Windows 11 and Server 2022 and newer, \
                 `std::env::temp_dir` goes through `GetTempPath2`, which ignores TMP and \
                 TEMP for a process running as SYSTEM; older Windows falls back to \
                 `GetTempPathW`, which honors them.",
            );

            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async move {
                    let uri = format!("{root}/dataset");
                    let reader = lance_datagen::gen_batch()
                        .col("id", array::step::<Int32Type>())
                        .col("vector", array::rand_vec::<Float32Type>(8.into()))
                        .into_reader_rows(RowCount::from(256), BatchCount::from(1));
                    let mut dataset = Dataset::write(reader, &uri, None).await.unwrap();

                    let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
                    for i in 0..2 {
                        dataset
                            .create_index(
                                &["vector"],
                                IndexType::Vector,
                                Some(format!("vector_idx_{i}")),
                                &params,
                                false,
                            )
                            .await
                            .unwrap();
                    }
                });
            return;
        }

        let isolated_root = TempStdDir::default();
        // libtest names the thread after the running test, so the child's filter
        // cannot drift out of sync with this function's name.
        let this_test = std::thread::current().name().unwrap().to_string();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([&this_test, "--exact", "--nocapture"])
            // `std::env::temp_dir()` reads `TMPDIR` on POSIX and `TMP`/`TEMP` on
            // Windows. Set all three, and let the child confirm the redirect took
            // effect: a child staging into the real temp dir would leave the scan
            // below looking at an empty directory and passing for the wrong reason.
            .env("TMPDIR", isolated_root.as_ref())
            .env("TMP", isolated_root.as_ref())
            .env("TEMP", isolated_root.as_ref())
            .env(ROOT_VAR, isolated_root.as_ref())
            .output()
            .expect("failed to spawn child test process");
        assert!(
            output.status.success(),
            "child build process failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );

        // A filter that matches nothing also exits 0, so confirm the child did the
        // work instead of reporting a clean scan of an untouched directory.
        assert!(
            isolated_root.join("dataset").is_dir(),
            "the child process did not run the build; filter was {this_test:?}"
        );

        // Every scratch dir a build creates sits directly under the child's temp
        // dir and is owned by a guard, so none should survive the child process.
        let leaked: Vec<std::path::PathBuf> = std::fs::read_dir(&isolated_root)
            .expect("read isolated temp root")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(".tmp"))
            })
            .collect();

        assert!(
            leaked.is_empty(),
            "vector index build leaked scratch directories under the temp dir: {leaked:?}. \
             `.tmp` is tempfile's default prefix, so one of these may belong to another \
             tempfile user rather than to the shuffler's guard."
        );
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_flat() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(64.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_FLAT index on source
        let params = VectorIndexParams::ivf_flat(8, MetricType::Cosine);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_flat".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_flat")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(64.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_FLAT index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_flat",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_FLAT"),
            "Index type should be IVF_FLAT"
        );

        // Check metric type (Cosine might be stored as "cosine")
        let metric = stats
            .get("metric_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            metric == "cosine" || metric == "Cosine",
            "Metric type should be Cosine, got: {}",
            metric
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(8),
            "Should have 8 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(8),
            "Should have 8 partitions as configured"
        );

        // Verify the index is functional
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(64.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }

    #[tokio::test]
    async fn test_build_distributed_invalid_fragment_ids() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(128), BatchCount::from(1));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let fragments = dataset.fragments();
        assert!(
            !fragments.is_empty(),
            "Dataset should have at least one fragment"
        );
        let max_id = fragments.iter().map(|f| f.id as u32).max().unwrap();
        let invalid_id = max_id + 1000;

        // let params = VectorIndexParams::ivf_flat(4, MetricType::L2);
        let uuid = Uuid::new_v4();

        let mut ivf_params = IvfBuildParams {
            num_partitions: Some(4),
            ..Default::default()
        };
        let dim = utils::get_vector_dim(dataset.schema(), "vector").unwrap();
        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            dim,
            MetricType::L2,
            &ivf_params,
            None,
            noop_progress(),
        )
        .await
        .unwrap();

        // Attach precomputed global centroids to ivf_params for distributed build.
        ivf_params.centroids = ivf_model.centroids.clone().map(Arc::new);

        let params = VectorIndexParams::with_ivf_flat_params(MetricType::L2, ivf_params);

        let result = build_distributed_vector_index(
            &dataset,
            "vector",
            "vector_ivf_flat_dist",
            uuid,
            &params,
            None,
            &[invalid_id],
            noop_progress(),
        )
        .await;

        assert!(
            result.is_ok(),
            "Expected Ok for invalid fragment ids, got {:?}",
            result
        );
    }

    /// Distributed builds take the codebook through their own path, so they
    /// need the same value-type check as `create_index`.
    #[tokio::test]
    async fn test_build_distributed_rejects_codebook_of_another_value_type() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());

        let vectors = FixedSizeListArray::try_new_from_values(
            Float16Array::from_iter_values((0..128 * 32).map(|i| half::f16::from_f32(i as f32))),
            32,
        )
        .unwrap();
        let batch =
            RecordBatch::try_from_iter([("vector", Arc::new(vectors) as arrow_array::ArrayRef)])
                .unwrap();
        let schema = batch.schema();
        let dataset = Dataset::write(RecordBatchIterator::new([Ok(batch)], schema), &uri, None)
            .await
            .unwrap();

        let centroids = FixedSizeListArray::try_new_from_values(
            Float16Array::from(vec![half::f16::ZERO; 2 * 32]),
            32,
        )
        .unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(centroids)).unwrap();
        let codebook = Arc::new(Float32Array::from(vec![0.0f32; 256 * 32]));
        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            ivf_params,
            PQBuildParams::with_codebook(4, 8, codebook),
        );
        let fragments: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        let error = build_distributed_vector_index(
            &dataset,
            "vector",
            "vector_ivf_pq_dist",
            Uuid::new_v4(),
            &params,
            None,
            &fragments,
            noop_progress(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(
                "PQ codebook value type Float32 does not match the vector value type Float16"
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_build_distributed_empty_fragment_ids() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(128), BatchCount::from(1));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let uuid = Uuid::new_v4();
        let mut ivf_params = IvfBuildParams {
            num_partitions: Some(4),
            ..Default::default()
        };
        let dim = utils::get_vector_dim(dataset.schema(), "vector").unwrap();
        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            dim,
            MetricType::L2,
            &ivf_params,
            None,
            noop_progress(),
        )
        .await
        .unwrap();

        // Attach precomputed global centroids to ivf_params for distributed build.
        ivf_params.centroids = ivf_model.centroids.clone().map(Arc::new);

        let params = VectorIndexParams::with_ivf_flat_params(MetricType::L2, ivf_params);

        let result = build_distributed_vector_index(
            &dataset,
            "vector",
            "vector_ivf_flat_dist",
            uuid,
            &params,
            None,
            &[],
            noop_progress(),
        )
        .await;

        assert!(
            result.is_ok(),
            "Expected Ok for empty fragment ids, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_train_ivf_progress_is_emitted_before_completion() {
        use std::sync::atomic::{AtomicBool, Ordering};

        #[derive(Debug)]
        struct RecordingProgress {
            train_ivf_complete: AtomicBool,
            saw_train_ivf_progress_before_complete: AtomicBool,
            saw_train_ivf_progress_after_complete: AtomicBool,
        }

        #[async_trait::async_trait]
        impl IndexBuildProgress for RecordingProgress {
            async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
                Ok(())
            }

            async fn stage_progress(&self, stage: &str, _: u64) -> Result<()> {
                if stage == "train_ivf" {
                    if self.train_ivf_complete.load(Ordering::Relaxed) {
                        self.saw_train_ivf_progress_after_complete
                            .store(true, Ordering::Relaxed);
                    } else {
                        self.saw_train_ivf_progress_before_complete
                            .store(true, Ordering::Relaxed);
                    }
                }
                Ok(())
            }

            async fn stage_complete(&self, stage: &str) -> Result<()> {
                if stage == "train_ivf" {
                    self.train_ivf_complete.store(true, Ordering::Relaxed);
                }
                Ok(())
            }
        }

        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(128), BatchCount::from(1));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_flat(4, MetricType::L2);
        let uuid = Uuid::new_v4();
        let progress = Arc::new(RecordingProgress {
            train_ivf_complete: AtomicBool::new(false),
            saw_train_ivf_progress_before_complete: AtomicBool::new(false),
            saw_train_ivf_progress_after_complete: AtomicBool::new(false),
        });

        build_vector_index(
            &dataset,
            "vector",
            "vector_ivf_flat_progress",
            uuid,
            &params,
            None,
            progress.clone(),
        )
        .await
        .unwrap();

        assert!(
            progress
                .saw_train_ivf_progress_before_complete
                .load(Ordering::Relaxed),
            "expected at least one train_ivf progress event before completion"
        );
        assert!(
            !progress
                .saw_train_ivf_progress_after_complete
                .load(Ordering::Relaxed),
            "found train_ivf progress after completion"
        );
    }

    #[tokio::test]
    async fn test_build_distributed_training_metadata_missing() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(128), BatchCount::from(1));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_flat(4, MetricType::L2);
        let uuid = Uuid::new_v4();

        // Pre-create a malformed global training file that is missing the
        // `lance:global_ivf_centroids` metadata key.
        let out_base = dataset.indices_dir().join(uuid.to_string());
        let training_path = out_base.clone().join("global_training.idx");

        let writer = dataset
            .table_state
            .object_store
            .as_ref()
            .create(&training_path)
            .await
            .unwrap();
        let arrow_schema = ArrowSchema::new(vec![Field::new("dummy", ArrowDataType::Int32, true)]);
        let mut v2w = lance_file::versions::create_writer(
            dataset_format_version(&dataset),
            writer,
            lance_core::datatypes::Schema::try_from(&arrow_schema).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        let empty_batch = RecordBatch::new_empty(Arc::new(arrow_schema));
        v2w.write_batch(&empty_batch).await.unwrap();
        v2w.finish().await.unwrap();

        let fragments = dataset.fragments();
        assert!(
            !fragments.is_empty(),
            "Dataset should have at least one fragment"
        );

        let valid_id = fragments[0].id as u32;
        let result = build_distributed_vector_index(
            &dataset,
            "vector",
            "vector_ivf_flat_dist",
            uuid,
            &params,
            None,
            &[valid_id],
            noop_progress(),
        )
        .await;

        match result {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("missing precomputed IVF centroids"),
                    "Unexpected error message: {}",
                    message
                );
            }
            Ok(_) => panic!("Expected Error::Index when IVF training metadata is missing, got Ok"),
            Err(e) => panic!(
                "Expected Error::Index when IVF training metadata is missing, got {:?}",
                e
            ),
        }
    }

    #[tokio::test]
    async fn test_initialize_vector_index_empty_dataset() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_PQ index on source
        let params = VectorIndexParams::ivf_pq(10, 8, 16, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_pq")
            .unwrap();

        // Create EMPTY target dataset with same schema
        let empty_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(0), BatchCount::from(1)); // Empty dataset
        let mut target_dataset = Dataset::write(empty_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_PQ index on empty target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created even though dataset is empty
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Empty target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_pq",
            "Index name should match"
        );

        // Open both indices to compare centroids
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Empty dataset should still have same number of partitions as source"
        );

        // Verify the centroids are exactly the same even for empty dataset
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length even for empty dataset"
            );

            // Compare actual centroid values
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Empty dataset should have identical centroids from source"
                );
            }
        } else {
            panic!("Both source and empty target should have centroids");
        }

        // Now add data to the target dataset
        let new_data_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        target_dataset.append(new_data_reader, None).await.unwrap();

        // Run optimize_indices to index the newly added data and merge indices
        // We set num_indices_to_merge to a high value to force merging all indices into one
        use lance_index::optimize::OptimizeOptions;
        target_dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();

        // Reload dataset to get updated index metadata
        let target_dataset = Dataset::open(&target_uri).await.unwrap();

        // Verify we have only one merged index after optimization
        let index_stats = target_dataset
            .index_statistics("vector_ivf_pq")
            .await
            .unwrap();
        let stats_json: serde_json::Value = serde_json::from_str(&index_stats).unwrap();
        assert_eq!(
            stats_json["num_indices"], 1,
            "Should have only 1 merged index after optimize with high num_indices_to_merge"
        );
        assert_eq!(
            stats_json["num_indexed_fragments"], 1,
            "Should have indexed the appended fragment (empty dataset has no fragments)"
        );
        assert_eq!(
            stats_json["num_unindexed_fragments"], 0,
            "All fragments should be indexed after optimization"
        );

        // The index should now work with the new data
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();

        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            results.num_rows(),
            5,
            "Should return 5 nearest neighbors after optimizing index"
        );

        // Verify that the optimized index still shares centroids with the source
        let target_indices = target_dataset.load_indices().await.unwrap();
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        let target_ivf_model = target_vector_index.ivf_model();

        // Verify centroids are still the same after optimization
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroids should remain identical after optimize_indices"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_sq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_SQ index on source
        use lance_index::vector::ivf::IvfBuildParams;
        use lance_index::vector::sq::builder::SQBuildParams;
        let ivf_params = IvfBuildParams::new(6);
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_sq_params(MetricType::Dot, ivf_params, sq_params);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_sq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_sq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_SQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_sq",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_SQ"),
            "Index type should be IVF_SQ"
        );

        // Check metric type (Dot might be stored as "dot")
        let metric = stats
            .get("metric_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            metric == "dot" || metric == "Dot",
            "Metric type should be Dot, got: {}",
            metric
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(6),
            "Should have 6 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(6),
            "Should have 6 partitions as configured"
        );

        // Verify SQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_sq: ScalarQuantizer = source_quantizer.try_into().unwrap();
        let target_sq: ScalarQuantizer = target_quantizer.try_into().unwrap();

        let source_sq_params = derive_sq_params(&source_sq);
        let target_sq_params = derive_sq_params(&target_sq);

        assert_eq!(
            source_sq_params.num_bits, target_sq_params.num_bits,
            "SQ num_bits should match"
        );

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 15)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 15, "Should return 15 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_hnsw_pq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // A 4-bit PQ codebook needs at least 16 training rows.
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(64), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_HNSW_PQ index on source with custom HNSW parameters
        let ivf_params = IvfBuildParams {
            num_partitions: Some(2),
            max_iters: 2,
            sample_rate: 2,
            ..Default::default()
        };
        let hnsw_params = HnswBuildParams {
            max_level: 2,
            m: 4,
            ef_construction: 16,
            prefetch_distance: None,
        };
        let pq_params = PQBuildParams {
            num_sub_vectors: 2,
            num_bits: 4,
            max_iters: 2,
            sample_rate: 2,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            pq_params,
        );

        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_hnsw_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_hnsw_pq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(32), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_HNSW_PQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_hnsw_pq",
            "Index name should match"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_HNSW_PQ"),
            "Index type should be IVF_HNSW_PQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("l2"),
            "Metric type should be L2"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(2),
            "Should have 2 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare first centroid to verify they're identical
            let source_centroid = source_centroids.value(0);
            let target_centroid = target_centroids.value(0);

            let source_data = source_centroid
                .as_any()
                .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                .expect("Centroid should be Float32Array");
            let target_data = target_centroid
                .as_any()
                .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                .expect("Centroid should be Float32Array");

            assert_eq!(
                source_data.values(),
                target_data.values(),
                "Centroid values should be identical between source and target"
            );
        } else {
            panic!("Both source and target should have centroids");
        }

        // Check sub_index contains HNSW and PQ information
        let sub_index = stats
            .get("sub_index")
            .and_then(|v| v.as_object())
            .expect("IVF_HNSW_PQ index should have sub_index");

        // Verify PQ parameters
        assert_eq!(
            sub_index.get("nbits").and_then(|v| v.as_u64()),
            Some(4),
            "PQ should use 4 bits"
        );
        assert_eq!(
            sub_index.get("num_sub_vectors").and_then(|v| v.as_u64()),
            Some(2),
            "PQ should have 2 sub vectors"
        );

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(2),
            "Should have 2 partitions as configured"
        );

        // Verify PQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_pq: ProductQuantizer = source_quantizer.try_into().unwrap();
        let target_pq: ProductQuantizer = target_quantizer.try_into().unwrap();

        let source_pq_params = derive_pq_params(&source_pq);
        let target_pq_params = derive_pq_params(&target_pq);

        assert_eq!(
            source_pq_params.num_sub_vectors, target_pq_params.num_sub_vectors,
            "PQ num_sub_vectors should match"
        );
        assert_eq!(
            source_pq_params.num_bits, target_pq_params.num_bits,
            "PQ num_bits should match"
        );
        assert_eq!(
            target_pq_params.num_sub_vectors, 2,
            "PQ should have 2 sub vectors"
        );
        assert_eq!(target_pq_params.num_bits, 4, "PQ should use 4 bits");

        // Verify HNSW parameters are extracted and used correctly
        let derived_hnsw_params = derive_hnsw_params(target_vector_index.as_ref());
        assert_eq!(
            derived_hnsw_params.max_level, 2,
            "HNSW max_level should be extracted as 2 from source index"
        );
        assert_eq!(
            derived_hnsw_params.m, 4,
            "HNSW m should be extracted as 4 from source index"
        );
        assert_eq!(
            derived_hnsw_params.ef_construction, 16,
            "HNSW ef_construction should be extracted as 16 from source index"
        );

        // Verify the index is functional
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(8.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_hnsw_sq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_HNSW_SQ index on source with custom HNSW parameters
        let ivf_params = IvfBuildParams {
            num_partitions: Some(6),
            ..Default::default()
        };
        let hnsw_params = HnswBuildParams {
            max_level: 5,
            m: 16,
            ef_construction: 80,
            prefetch_distance: None,
        };
        let sq_params = SQBuildParams {
            num_bits: 8,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::Cosine,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_hnsw_sq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_hnsw_sq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_HNSW_SQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_hnsw_sq",
            "Index name should match"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index("vector", &target_indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_HNSW_SQ"),
            "Index type should be IVF_HNSW_SQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("cosine"),
            "Metric type should be cosine"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(6),
            "Should have 6 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index("vector", &source_index.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Check sub_index contains SQ information
        let sub_index = stats
            .get("sub_index")
            .and_then(|v| v.as_object())
            .expect("IVF_HNSW_SQ index should have sub_index");
        // Verify SQ parameters
        assert_eq!(
            sub_index.get("num_bits").and_then(|v| v.as_u64()),
            Some(8),
            "SQ should use 8 bits"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(6),
            "Should have 6 partitions as configured"
        );

        // Verify SQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_sq: ScalarQuantizer = source_quantizer.try_into().unwrap();
        let target_sq: ScalarQuantizer = target_quantizer.try_into().unwrap();

        let source_sq_params = derive_sq_params(&source_sq);
        let target_sq_params = derive_sq_params(&target_sq);

        assert_eq!(
            source_sq_params.num_bits, target_sq_params.num_bits,
            "SQ num_bits should match"
        );
        assert_eq!(target_sq_params.num_bits, 8, "SQ should use 8 bits");

        // Verify HNSW parameters are extracted and used correctly
        let derived_hnsw_params = derive_hnsw_params(target_vector_index.as_ref());
        assert_eq!(
            derived_hnsw_params.max_level, 5,
            "HNSW max_level should be extracted as 5 from source index"
        );
        assert_eq!(
            derived_hnsw_params.m, 16,
            "HNSW m should be extracted as 16 from source index"
        );
        assert_eq!(
            derived_hnsw_params.ef_construction, 80,
            "HNSW ef_construction should be extracted as 80 from source index"
        );

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }
}
