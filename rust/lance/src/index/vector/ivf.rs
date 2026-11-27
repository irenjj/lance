// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use super::pq::PQIndex;
use super::utils::PartitionLoadLock;
use lance_index::prefilter::PreFilter;

use crate::index::vector::IndexFileVersion;
use arrow::datatypes::UInt8Type;
use arrow_arith::numeric::sub;
use arrow_array::Float32Array;
use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, UInt32Array,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion_execution::SendableRecordBatchStream;
use futures::TryFutureExt;
use futures::{TryStreamExt, stream::StreamExt};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{
    Error, Result,
    cache::{CacheKeySchema, KeyBuilder, LanceCache, UnsizedCacheKey, WeakLanceCache},
    utils::parse::parse_env_as_bool,
    utils::tracing::{IO_TYPE_LOAD_VECTOR_PART, TRACE_IO_EVENTS},
};
use lance_index::metrics::MetricsCollector;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::scalar::RowAddrTranslator;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    Index, IndexType,
    vector::{
        Query, VectorIndex, hnsw::HNSWIndex, pq::ProductQuantizer, quantizer::Quantizer,
        sq::ScalarQuantizer,
    },
};
use lance_io::{local::to_local_path, traits::Reader};
use lance_linalg::distance::MetricType;
use lance_select::RowAddrTreeMap;
use log::warn;
use roaring::RoaringBitmap;
use serde::Serialize;
use std::{
    any::Any,
    sync::{Arc, OnceLock},
};
use tracing::instrument;
use uuid::Uuid;

mod compatibility;
mod partition_serde;
mod shape;
pub mod v2;

pub(crate) use compatibility::VectorQueryContract;

// Cache wrapper for vector index trait objects
// Cache key for IVF partitions in the legacy IVF index
#[derive(Debug, Clone)]
pub struct LegacyIVFPartitionKey {
    pub partition_id: usize,
}

impl LegacyIVFPartitionKey {
    pub fn new(partition_id: usize) -> Self {
        Self { partition_id }
    }
}

impl UnsizedCacheKey for LegacyIVFPartitionKey {
    type ValueType = dyn VectorIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("ivf-{}", self.partition_id).into()
    }

    fn type_name() -> &'static str {
        "LegacyIVFPartition"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.legacy-ivf-partition-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.partition_id as u64);
    }
}

/// IVF Index.
/// WARNING: Internal API with no stability guarantees.
pub struct IVFIndex {
    uuid: Uuid,

    /// Ivf model
    pub ivf: IvfModel,

    reader: Arc<dyn Reader>,

    /// Index in each partition.
    sub_index: Arc<dyn VectorIndex>,

    partition_locks: PartitionLoadLock,

    pub metric_type: MetricType,

    index_cache: WeakLanceCache,
    partition_rows: Vec<OnceLock<Arc<RowAddrTreeMap>>>,
}

impl DeepSizeOf for IVFIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // `Uuid` is a fixed 16-byte struct with no heap children, so contributes 0.
        self.reader.deep_size_of_children(context)
            + self.sub_index.deep_size_of_children(context)
            + self
                .partition_rows
                .iter()
                .filter_map(OnceLock::get)
                .map(|rows| rows.deep_size_of_children(context))
                .sum::<usize>()
    }
}

impl IVFIndex {
    /// Create a new IVF index.
    pub(crate) fn try_new(
        uuid: Uuid,
        ivf: IvfModel,
        reader: Arc<dyn Reader>,
        sub_index: Arc<dyn VectorIndex>,
        metric_type: MetricType,
        index_cache: LanceCache,
    ) -> Result<Self> {
        if !sub_index.is_loadable() {
            return Err(Error::index(format!(
                "IVF sub index must be loadable, got: {:?}",
                sub_index
            )));
        }

        let num_partitions = ivf.num_partitions();
        Ok(Self {
            uuid,
            ivf,
            reader,
            sub_index,
            metric_type,
            partition_locks: PartitionLoadLock::new(num_partitions),
            index_cache: WeakLanceCache::from(&index_cache),
            partition_rows: (0..num_partitions).map(|_| OnceLock::new()).collect(),
        })
    }

    fn cache_partition_rows(
        &self,
        partition_id: usize,
        partition: &dyn VectorIndex,
    ) -> Result<Arc<RowAddrTreeMap>> {
        let rows = self.partition_rows.get(partition_id).ok_or_else(|| {
            Error::index(format!(
                "partition id {partition_id} is out of range of {} partitions",
                self.ivf.num_partitions()
            ))
        })?;
        Ok(rows
            .get_or_init(|| Arc::new(partition.row_ids().collect()))
            .clone())
    }

    fn prefilter_for_partition(
        &self,
        partition_id: usize,
        partition: &dyn VectorIndex,
        pre_filter: Arc<dyn PreFilter>,
    ) -> Result<Arc<dyn PreFilter>> {
        if pre_filter.is_empty() {
            return Ok(Arc::new(NoFilter));
        }
        if !pre_filter.needs_partition_row_ids() {
            return Ok(pre_filter);
        }
        let rows = self.cache_partition_rows(partition_id, partition)?;
        if pre_filter.is_empty_for(rows.as_ref()) {
            Ok(Arc::new(NoFilter))
        } else {
            Ok(pre_filter)
        }
    }

    /// Load one partition of the IVF sub-index.
    ///
    /// Internal API with no stability guarantees.
    ///
    /// Parameters
    /// ----------
    ///  - partition_id: partition ID.
    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        let cache_key = LegacyIVFPartitionKey::new(partition_id);
        let part_index = if let Some(part_idx) =
            self.index_cache.get_unsized_with_key(&cache_key).await
        {
            part_idx
        } else {
            metrics.record_part_load();
            tracing::info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=cache_key.key().as_ref());

            let mtx = self.partition_locks.get_partition_mutex(partition_id);
            let _guard = mtx.lock().await;
            // check the cache again, as the partition may have been loaded by another
            // thread that held the lock on loading the partition
            if let Some(part_idx) = self.index_cache.get_unsized_with_key(&cache_key).await {
                part_idx
            } else {
                if partition_id >= self.ivf.num_partitions() {
                    return Err(Error::index(format!(
                        "partition id {} is out of range of {} partitions",
                        partition_id,
                        self.ivf.num_partitions()
                    )));
                }

                let range = self.ivf.row_range(partition_id);
                let idx = self
                    .sub_index
                    .load_partition(
                        self.reader.clone(),
                        range.start,
                        range.end - range.start,
                        partition_id,
                    )
                    .await?;
                let idx: Arc<dyn VectorIndex> = idx.into();
                if write_cache {
                    self.index_cache
                        .insert_unsized_with_key(&cache_key, idx.clone())
                        .await;
                }
                idx
            }
        };
        Ok(part_index)
    }

    /// preprocess the query vector given the partition id.
    ///
    /// Internal API with no stability guarantees.
    pub fn preprocess_query(&self, partition_id: usize, query: &Query) -> Result<Query> {
        if self.sub_index.use_residual() {
            let partition_centroids = self.ivf.centroids.as_ref().unwrap().value(partition_id);
            let residual_key = sub(&query.key, &partition_centroids)?;
            let mut part_query = query.clone();
            part_query.key = residual_key;
            Ok(part_query)
        } else {
            Ok(query.clone())
        }
    }
}

impl std::fmt::Debug for IVFIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ivf({}) -> {:?}", self.metric_type, self.sub_index)
    }
}

#[derive(Serialize)]
pub struct IvfIndexPartitionStatistics {
    size: u32,
}

#[derive(Serialize)]
pub struct IvfIndexStatistics {
    index_type: String,
    uuid: String,
    uri: String,
    metric_type: String,
    num_partitions: usize,
    sub_index: serde_json::Value,
    partitions: Vec<IvfIndexPartitionStatistics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    centroids: Option<Vec<Vec<f32>>>,
    loss: Option<f64>,
    index_file_version: IndexFileVersion,
}

/// Environment variable controlling whether vector index statistics include
/// the centroid vectors. When unset, centroids are still included for
/// backward compatibility, but a one-time warning is logged. Set to a truthy
/// value (e.g. `1`, `true`) to keep the current behavior without the warning,
/// or any other value (e.g. `0`, `false`) to omit centroids from stats.
pub const LANCE_INCLUDE_VECTOR_CENTROIDS_ENV: &str = "LANCE_INCLUDE_VECTOR_CENTROIDS";

/// Read the centroids for inclusion in index stats, honoring
/// `LANCE_INCLUDE_VECTOR_CENTROIDS`.
///
/// - If the env var is set to a truthy value (per `parse_env_as_bool`),
///   returns the converted centroids.
/// - If the env var is set to any other value, returns `Ok(None)` without
///   reading the centroids.
/// - If unset, returns the converted centroids and logs a one-time
///   deprecation warning that the default will change in a future release.
pub(crate) fn maybe_centroids_for_stats(
    centroids: &FixedSizeListArray,
) -> Result<Option<Vec<Vec<f32>>>> {
    use std::sync::Once;
    static WARN_ONCE: Once = Once::new();

    if std::env::var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV).is_err() {
        WARN_ONCE.call_once(|| {
            warn!(
                "Vector index statistics currently include centroids, which can use \
                 significant memory for large indexes. In a future release, centroids \
                 will be excluded from statistics by default. Set {}=true to preserve \
                 the current behavior (and silence this warning), or {}=false to opt \
                 in to the new behavior now.",
                LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, LANCE_INCLUDE_VECTOR_CENTROIDS_ENV
            );
        });
    }
    if !parse_env_as_bool(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, true) {
        return Ok(None);
    }
    Ok(Some(centroids_to_vectors(centroids)?))
}

fn centroids_to_vectors(centroids: &FixedSizeListArray) -> Result<Vec<Vec<f32>>> {
    centroids
        .iter()
        .map(|v| {
            if let Some(row) = v {
                match row.data_type() {
                    DataType::Float16 => Ok(row
                        .as_primitive::<Float16Type>()
                        .values()
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>()),
                    DataType::Float32 => Ok(row.as_primitive::<Float32Type>().values().to_vec()),
                    DataType::Float64 => Ok(row
                        .as_primitive::<Float64Type>()
                        .values()
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>()),
                    DataType::UInt8 => Ok(row
                        .as_primitive::<UInt8Type>()
                        .values()
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>()),
                    _ => Err(Error::index(format!(
                        "IVF centroids must be FixedSizeList of floating number, got: {}",
                        row.data_type()
                    ))),
                }
            } else {
                Err(Error::index("Invalid centroid".to_string()))
            }
        })
        .collect()
}

#[async_trait]
impl Index for IVFIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        if self.sub_index.as_any().downcast_ref::<PQIndex>().is_some() {
            IndexType::IvfPq
        } else if self
            .sub_index
            .as_any()
            .downcast_ref::<HNSWIndex<ScalarQuantizer>>()
            .is_some()
        {
            IndexType::IvfHnswSq
        } else if self
            .sub_index
            .as_any()
            .downcast_ref::<HNSWIndex<ProductQuantizer>>()
            .is_some()
        {
            IndexType::IvfHnswPq
        } else {
            IndexType::Vector
        }
    }

    async fn prewarm(&self) -> Result<()> {
        futures::stream::iter(0..self.ivf.num_partitions())
            .map(Ok)
            .try_for_each_concurrent(Some(self.reader.io_parallelism()), |part_id| {
                self.load_partition(part_id, true, &NoOpMetricsCollector)
                    .map_ok(|_| ())
            })
            .await
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = (0..self.ivf.num_partitions())
            .map(|part_id| IvfIndexPartitionStatistics {
                size: self.ivf.partition_size(part_id) as u32,
            })
            .collect::<Vec<_>>();

        let centroid_vecs = maybe_centroids_for_stats(self.ivf.centroids.as_ref().unwrap())?;

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type: self.index_type().to_string(),
            uuid: self.uuid.to_string(),
            uri: to_local_path(self.reader.path()),
            metric_type: self.metric_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: self.sub_index.statistics()?,
            partitions: partitions_statistics,
            centroids: centroid_vecs,
            loss: self.ivf.loss(),
            index_file_version: IndexFileVersion::Legacy,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();
        let part_ids = 0..self.ivf.num_partitions();
        for part_id in part_ids {
            let part = self
                .load_partition(part_id, false, &NoOpMetricsCollector)
                .await?;
            frag_ids |= part.calculate_included_frags().await?;
        }
        Ok(frag_ids)
    }
}

#[async_trait]
impl VectorIndex for IVFIndex {
    #[instrument(level = "debug", skip_all, name = "IVFIndex::search")]
    async fn search(
        &self,
        _query: &Query,
        _pre_filter: Arc<dyn PreFilter>,
        _metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        unimplemented!(
            "IVFIndex not currently used as sub-index and top-level indices do partition-aware search"
        )
    }

    /// find the IVF partitions ids given the query vector.
    ///
    /// Internal API with no stability guarantees.
    ///
    /// Assumes the query vector is normalized if the metric type is cosine.
    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
        let mt = if self.metric_type == MetricType::Cosine {
            MetricType::L2
        } else {
            self.metric_type
        };

        let max_nprobes = query.maximum_nprobes.unwrap_or(self.ivf.num_partitions());

        self.ivf.find_partitions(&query.key, max_nprobes, mt)
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let part_index = self.load_partition(partition_id, true, metrics).await?;
        pre_filter.wait_for_ready().await?;
        let pre_filter =
            self.prefilter_for_partition(partition_id, part_index.as_ref(), pre_filter)?;

        let query = self.preprocess_query(partition_id, query)?;
        let batch = part_index.search(&query, pre_filter, metrics).await?;
        Ok(batch)
    }

    fn total_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: Arc<dyn Reader>,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::index("Flat index does not support load".to_string()))
    }

    async fn partition_reader(
        &self,
        partition_id: usize,
        with_vector: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        let partition = self.load_partition(partition_id, false, metrics).await?;
        partition.to_batch_stream(with_vector).await
    }

    async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
        unimplemented!("this method is for only sub index")
    }

    fn num_rows(&self) -> u64 {
        self.ivf.num_rows()
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &u64>> {
        todo!("this method is for only IVF_HNSW_* index");
    }

    async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
        // This will be needed if we want to clean up IVF to allow more than just
        // one layer (e.g. IVF -> IVF -> PQ).  We need to pass on the call to
        // remap to the lower layers.

        // Currently, remapping for IVF is implemented in remap_index_file which
        // mirrors some of the other IVF routines like build_ivf_pq_index
        Err(Error::index(
            "Remapping IVF in this way not supported".to_string(),
        ))
    }

    async fn remap_streaming(&mut self, _translator: &RowAddrTranslator) -> Result<()> {
        // No mapping to materialize: see `remap`.
        Err(Error::index(
            "Remapping IVF in this way not supported".to_string(),
        ))
    }

    fn ivf_model(&self) -> &IvfModel {
        &self.ivf
    }

    fn quantizer(&self) -> Quantizer {
        // A legacy index keeps its quantizer on the sub-index. The shared-model
        // check in `optimize_vector_indices` reads it before the v1/v2 dispatch.
        self.sub_index.quantizer()
    }

    fn partition_size(&self, part_id: usize) -> usize {
        self.ivf.partition_size(part_id)
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        self.sub_index.sub_index_type()
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }
}

#[cfg(any())]
#[rustfmt::skip]
mod tests {
    use super::*;
    use lance_core::utils::row_addr_remap::RowAddrRemap;

    use std::collections::HashSet;
    use std::f32::consts::TAU;
    use std::iter::repeat_n;
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use arrow_array::types::UInt64Type;
    use arrow_array::{
        FixedSizeListArray, Float16Array, Float32Array, Float64Array, Int8Array, RecordBatch,
        RecordBatchIterator, RecordBatchReader, UInt16Array, UInt64Array, make_array,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    use arrow_schema::{DataType, Field, Schema};
    use half::f16;
    use itertools::Itertools;
    use lance_core::ROW_ID;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount, array, gen_batch};
    use lance_index::VECTOR_INDEX_VERSION;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::progress::noop_progress;
    use lance_index::scalar::OldIndexDataFilter;
    use lance_index::vector::bq::RQBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_linalg::distance::l2_distance_batch_into;
    use lance_testing::datagen::{
        generate_random_array, generate_random_array_with_range, generate_random_array_with_seed,
        generate_scaled_random_array, sample_without_replacement,
    };
    use rand::{rng, seq::SliceRandom};
    use rstest::rstest;
    use tokio::sync::Notify;

    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::prefilter::DatasetPreFilter;
    use crate::index::vector::IndexFileVersion;
    use crate::index::vector::details::vector_index_details_default;
    use crate::index::{
        DatasetIndexExt, loading::DatasetIndexInternalExt, vector::VectorIndexParams,
    };
    use crate::utils::test::copy_test_data_to_tmp;

    mod optimize;

    const DIM: usize = 32;

    #[derive(Debug)]
    struct BlockingFailingProgress {
        is_first_call: AtomicBool,
        is_callback_active: Arc<AtomicBool>,
        callback_started: Arc<Notify>,
        release_callback: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl lance_index::progress::IndexBuildProgress for BlockingFailingProgress {
        async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
            Ok(())
        }

        async fn stage_progress(&self, _: &str, _: u64) -> Result<()> {
            if self.is_first_call.swap(false, Ordering::SeqCst) {
                self.is_callback_active.store(true, Ordering::SeqCst);
                self.callback_started.notify_one();
                self.release_callback.notified().await;
                self.is_callback_active.store(false, Ordering::SeqCst);
                return Err(Error::io("injected progress failure"));
            }
            Ok(())
        }

        async fn stage_complete(&self, _: &str) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_shared_quantizer_model_compares_skipped_payloads() {
        let codebook = |offset| {
            let values = Float32Array::from_iter_values((0..32).map(|v| v as f32 + offset));
            FixedSizeListArray::try_new_from_values(values, 2).unwrap()
        };
        let pq1 = Quantizer::Product(ProductQuantizer::new(
            1,
            4,
            2,
            codebook(0.0),
            DistanceType::L2,
        ));
        let pq2 = Quantizer::Product(ProductQuantizer::new(
            1,
            4,
            2,
            codebook(100.0),
            DistanceType::L2,
        ));
        assert!(!shared_quantizer_model(&pq1, &pq2));

        let rq1 = Quantizer::Rabit(RabitQuantizer::new_with_rotation::<Float32Type>(
            1,
            8,
            lance_index::vector::bq::RQRotationType::Matrix,
        ));
        let rq2 = Quantizer::Rabit(RabitQuantizer::new_with_rotation::<Float32Type>(
            1,
            8,
            lance_index::vector::bq::RQRotationType::Matrix,
        ));
        assert!(!shared_quantizer_model(&rq1, &rq2));
    }

    async fn compute_test_ivf_loss(dataset: &Dataset, column: &str, ivf: &IvfModel) -> f64 {
        let centroids = ivf
            .centroids_array()
            .expect("test IVF model should include centroids");
        let mut scanner = dataset.scan();
        scanner.project(&[column]).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let data = batch
            .column_by_name(column)
            .expect("test vector column should exist")
            .as_fixed_size_list()
            .clone();
        let kmeans = KMeans::with_centroids(
            centroids.values().clone(),
            centroids.value_length() as usize,
            DistanceType::L2,
            f64::MAX,
        );
        kmeans.compute_loss(&data).unwrap()
    }

    // Verifies LANCE_INCLUDE_VECTOR_CENTROIDS env var is honored by
    // maybe_centroids_for_stats. The env var is process-global, so this test
    // is serialized against any other test that touches the same key.
    #[test]
    #[serial_test::serial(LANCE_INCLUDE_VECTOR_CENTROIDS)]
    fn test_maybe_centroids_for_stats_env_var() {
        let centroids = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let centroids = FixedSizeListArray::try_new_from_values(centroids, 2).unwrap();
        let expected = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Save the original value so we can restore it afterwards.
        let original = std::env::var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV).ok();

        // Unset → centroids included (with one-time warning).
        unsafe {
            std::env::remove_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV);
        }
        assert_eq!(maybe_centroids_for_stats(&centroids).unwrap(), expected);

        // Truthy values → centroids included.
        for truthy in ["1", "true", "TRUE", "on", "yes", "y"] {
            unsafe {
                std::env::set_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, truthy);
            }
            assert_eq!(
                maybe_centroids_for_stats(&centroids).unwrap(),
                expected,
                "expected centroids to be included for {truthy:?}",
            );
        }

        // Non-truthy values → centroids omitted.
        for falsy in ["0", "false", "FALSE", "no", "off"] {
            unsafe {
                std::env::set_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, falsy);
            }
            assert_eq!(
                maybe_centroids_for_stats(&centroids).unwrap(),
                None,
                "expected centroids to be omitted for {falsy:?}",
            );
        }

        // Restore original value.
        unsafe {
            match original {
                Some(v) => std::env::set_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, v),
                None => std::env::remove_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV),
            }
        }
    }

    // Verifies that when centroids are omitted via the env var, the
    // serialized stats JSON does not contain the `centroids` field at all
    // (instead of an explicit null), since downstream code distinguishes
    // missing from null.
    #[test]
    #[serial_test::serial(LANCE_INCLUDE_VECTOR_CENTROIDS)]
    fn test_stats_centroids_omitted_when_disabled() {
        let original = std::env::var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV).ok();

        unsafe {
            std::env::set_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, "false");
        }
        let stats = IvfIndexStatistics {
            index_type: "IVF_PQ".to_string(),
            uuid: "uuid".to_string(),
            uri: "uri".to_string(),
            metric_type: "l2".to_string(),
            num_partitions: 0,
            sub_index: serde_json::Value::Null,
            partitions: vec![],
            centroids: None,
            loss: None,
            index_file_version: IndexFileVersion::V3,
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert!(json.get("centroids").is_none());

        unsafe {
            match original {
                Some(v) => std::env::set_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV, v),
                None => std::env::remove_var(LANCE_INCLUDE_VECTOR_CENTROIDS_ENV),
            }
        }
    }

    /// This goal of this function is to generate data that behaves in a very deterministic way so that
    /// we can evaluate the correctness of an IVF_PQ implementation.  Currently it is restricted to the
    /// L2 distance metric.
    ///
    /// First, we generate a set of centroids.  These are generated randomly but we ensure that is
    /// sufficient distance between each of the centroids.
    ///
    /// Then, we generate 256 vectors per centroid.  Each vector is generated by making a line by
    /// adding / subtracting [1,1,1...,1] (with the centroid in the middle)
    ///
    /// The trained result should have our generated centroids (without these centroids actually being
    /// a part of the input data) and the PQ codes for every data point should be identical and, given
    /// any three data points a, b, and c that are in the same centroid then the distance between a and
    /// b should be different than the distance between a and c.
    struct WellKnownIvfPqData {
        dim: u32,
        num_centroids: u32,
        centroids: Option<Float32Array>,
        vectors: Option<Float32Array>,
    }

    impl WellKnownIvfPqData {
        // Right now we are assuming 8-bit codes
        const VALS_PER_CODE: u32 = 256;
        const COLUMN: &'static str = "vector";

        fn new(dim: u32, num_centroids: u32) -> Self {
            Self {
                dim,
                num_centroids,
                centroids: None,
                vectors: None,
            }
        }

        fn generate_centroids(dim: u32, num_centroids: u32) -> Float32Array {
            const MAX_ATTEMPTS: u32 = 10;
            let distance_needed = (dim as f32).sqrt() * Self::VALS_PER_CODE as f32 * 2_f32;
            let mut attempts_remaining = MAX_ATTEMPTS;
            let num_values = dim * num_centroids;
            while attempts_remaining > 0 {
                // Use some biggish numbers to ensure we get the distance we want but make them positive
                // and not too big for easier debugging.
                let centroids: Float32Array =
                    generate_scaled_random_array(num_values as usize, 0_f32, 1000_f32);
                let mut broken = false;
                for (index, centroid) in centroids.values().chunks_exact(dim as usize).enumerate() {
                    let offset = (index + 1) * dim as usize;
                    let length = centroids.len() - offset;
                    if length == 0 {
                        // This will be true for the last item since we ignore comparison with self
                        continue;
                    }
                    let mut distances = vec![0.0; length / dim as usize];
                    l2_distance_batch_into(
                        centroid,
                        &centroids.values()[offset..offset + length],
                        dim as usize,
                        &mut distances,
                    );
                    let min_distance = distances.into_iter().min_by(|a, b| a.total_cmp(b)).unwrap();
                    // In theory we could just replace this one vector but, out of laziness, we just retry all of them
                    if min_distance < distance_needed {
                        broken = true;
                        break;
                    }
                }
                if !broken {
                    return centroids;
                }
                attempts_remaining -= 1;
            }
            panic!(
                "Unable to generate centroids with sufficient distance after {} attempts",
                MAX_ATTEMPTS
            );
        }

        fn get_centroids(&mut self) -> &Float32Array {
            let dim = self.dim;
            let num_centroids = self.num_centroids;
            self.centroids
                .get_or_insert_with(|| Self::generate_centroids(dim, num_centroids))
        }

        fn get_centroids_as_list_arr(&mut self) -> Arc<FixedSizeListArray> {
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    self.get_centroids().clone(),
                    self.dim as i32,
                )
                .unwrap(),
            )
        }

        fn generate_vectors(
            dim: u32,
            num_centroids: u32,
            centroids: &Float32Array,
        ) -> Float32Array {
            let dim = dim as usize;
            let mut vectors: Vec<f32> =
                vec![0_f32; Self::VALS_PER_CODE as usize * dim * num_centroids as usize];
            for (centroid, dst_batch) in centroids
                .values()
                .chunks_exact(dim)
                .zip(vectors.chunks_exact_mut(dim * Self::VALS_PER_CODE as usize))
            {
                for (offset, dst) in (-128..0).chain(1..129).zip(dst_batch.chunks_exact_mut(dim)) {
                    for (cent_val, dst_val) in centroid.iter().zip(dst) {
                        *dst_val = *cent_val + offset as f32;
                    }
                }
            }
            Float32Array::from(vectors)
        }

        fn get_vectors(&mut self) -> &Float32Array {
            let dim = self.dim;
            let num_centroids = self.num_centroids;
            let centroids = self.get_centroids().clone();
            self.vectors
                .get_or_insert_with(|| Self::generate_vectors(dim, num_centroids, &centroids))
        }

        fn get_vector(&mut self, idx: u32) -> Float32Array {
            let dim = self.dim as usize;
            let vectors = self.get_vectors();
            let start = idx as usize * dim;
            vectors.slice(start, dim)
        }

        fn generate_batches(&mut self) -> impl RecordBatchReader + Send + 'static {
            let dim = self.dim as usize;
            let vectors_array = self.get_vectors();

            let schema = Arc::new(Schema::new(vec![Field::new(
                Self::COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            )]));
            let array = Arc::new(
                FixedSizeListArray::try_new_from_values(vectors_array.clone(), dim as i32).unwrap(),
            );
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema)
        }

        async fn generate_dataset(&mut self, test_uri: &str) -> Result<Dataset> {
            let batches = self.generate_batches();
            Dataset::write(batches, test_uri, None).await
        }

        async fn check_index<F: Fn(u64) -> Option<u64>>(
            &mut self,
            index: &IVFIndex,
            prefilter: Arc<dyn PreFilter>,
            ids_to_test: &[u64],
            row_id_map: F,
        ) {
            const ROWS_TO_TEST: u32 = 500;
            let num_vectors = ids_to_test.len() as u32 / self.dim;
            let num_tests = ROWS_TO_TEST.min(num_vectors);
            let row_ids_to_test = sample_without_replacement(ids_to_test, num_tests);
            for row_id in row_ids_to_test {
                let row = self.get_vector(row_id as u32);
                let query = Query {
                    column: Self::COLUMN.to_string(),
                    key: Arc::new(row),
                    k: 5,
                    lower_bound: None,
                    upper_bound: None,
                    minimum_nprobes: 1,
                    maximum_nprobes: None,
                    search_effort: 0.5,
                    ef: None,
                    refine_factor: None,
                    metric_type: Some(MetricType::L2),
                    use_index: true,
                    query_parallelism: lance_index::vector::DEFAULT_QUERY_PARALLELISM,
                    dist_q_c: 0.0,
                    approx_mode: Default::default(),
                    rq_graph_scoring: Default::default(),
                };
                let (partitions, _) = index.find_partitions(&query).unwrap();
                let nearest_partition_id = partitions.value(0) as usize;
                let search_result = index
                    .search_in_partition(
                        nearest_partition_id,
                        &query,
                        prefilter.clone(),
                        &NoOpMetricsCollector,
                    )
                    .await
                    .unwrap();

                let found_ids = search_result.column(1);
                let found_ids = found_ids.as_any().downcast_ref::<UInt64Array>().unwrap();
                let expected_id = row_id_map(row_id);

                match expected_id {
                    // Original id was deleted, results can be anything, just make sure they don't
                    // include the original id
                    None => assert!(!found_ids.iter().any(|f_id| f_id.unwrap() == row_id)),
                    // Original id remains or was remapped, make sure expected id in results
                    Some(expected_id) => {
                        assert!(found_ids.iter().any(|f_id| f_id.unwrap() == expected_id))
                    }
                };
                // The invalid row id should never show up in results
                assert!(
                    !found_ids
                        .iter()
                        .any(|f_id| f_id.unwrap() == RowAddress::TOMBSTONE_ROW)
                );
            }
        }
    }

    async fn generate_test_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        generate_test_dataset_with_rows(test_uri, range, 1000).await
    }

    async fn generate_test_dataset_with_rows(
        test_uri: &str,
        range: Range<f32>,
        num_rows: usize,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range::<Float32Type>(num_rows * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(
            Schema::new(vec![Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            )])
            .with_metadata(metadata),
        );
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, array)
    }

    #[tokio::test]
    async fn test_create_ivf_rejects_zero_target_partition_size() {
        assert_zero_target_rejected(IvfBuildParams::with_target_partition_size(0)).await;
    }

    #[tokio::test]
    async fn test_create_ivf_rejects_zero_target_with_num_partitions() {
        assert_zero_target_rejected(IvfBuildParams {
            target_partition_size: Some(0),
            ..IvfBuildParams::new(20)
        })
        .await;
    }

    #[tokio::test]
    async fn test_create_ivf_rejects_zero_target_with_centroids() {
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(vec![0.0; 2 * DIM]),
                DIM as i32,
            )
            .unwrap(),
        );
        assert_zero_target_rejected(IvfBuildParams {
            target_partition_size: Some(0),
            centroids: Some(centroids),
            ..Default::default()
        })
        .await;
    }

    async fn assert_zero_target_rejected(ivf_params: IvfBuildParams) {
        let test_dir = TempStrDir::default();
        let (mut dataset, _) = generate_test_dataset(test_dir.as_str(), 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_flat_params(MetricType::L2, ivf_params);
        let err = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("target_partition_size must be greater than zero"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_create_ivf_pq_with_centroids() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let sample_query = vector_array.value(10);
        let query = sample_query.as_primitive::<Float32Type>();
        let results = dataset
            .scan()
            .nearest("vector", query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(5, results[0].num_rows());
    }

    /// A precomputed partitions file is user data. Its ids index the centroids
    /// in the transform, so IVF_PQ's residual step and IVF_RQ's distance to the
    /// chosen centroid reach a bad id before the shuffler's own range check
    /// (`v3/shuffler.rs`) can run.
    #[rstest]
    #[case::ivf_pq(IndexType::IvfPq)]
    #[case::ivf_rq(IndexType::IvfRq)]
    #[tokio::test]
    async fn test_create_index_rejects_out_of_range_precomputed_partition(
        #[case] index_type: IndexType,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = format!("{}/ds", test_dir.as_str());
        let partitions_uri = format!("{}/parts", test_dir.as_str());

        let (mut dataset, _) = generate_test_dataset(&test_uri, 0.0..1.0).await;
        let num_rows = dataset.count_rows(None).await.unwrap() as u64;

        // Partition 7 does not exist in a 2-partition index.
        let parts_schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("partition", DataType::UInt32, false),
        ]));
        let parts_batch = RecordBatch::try_new(
            parts_schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..num_rows)),
                Arc::new(UInt32Array::from_iter_values(
                    (0..num_rows).map(|i| if i == 3 { 7 } else { 0 }),
                )),
            ],
        )
        .unwrap();
        let parts_reader =
            RecordBatchIterator::new(vec![parts_batch].into_iter().map(Ok), parts_schema);
        Dataset::write(parts_reader, &partitions_uri, None)
            .await
            .unwrap();

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();
        ivf_params.precomputed_partitions_file = Some(partitions_uri);
        let params = match index_type {
            IndexType::IvfPq => {
                let pq_params =
                    PQBuildParams::with_codebook(4, 8, Arc::new(generate_random_array(256 * DIM)));
                VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params)
            }
            IndexType::IvfRq => VectorIndexParams::with_ivf_rq_params(
                MetricType::L2,
                ivf_params,
                RQBuildParams::default(),
            ),
            _ => unreachable!("not a case of this test"),
        };

        let error = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .expect_err("an out-of-range precomputed partition must be rejected");
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("partition id 7") && message.contains("out of range for 2"),
            "the error must name the id and the partition count: {message}"
        );
    }

    fn partition_ids(mut ids: Vec<u64>, num_parts: u32) -> Vec<Vec<u64>> {
        if num_parts > ids.len() as u32 {
            panic!("Not enough ids to break into {num_parts} parts");
        }
        let mut rng = rng();
        ids.shuffle(&mut rng);

        let values_per_part = ids.len() / num_parts as usize;
        let parts_with_one_extra = ids.len() % num_parts as usize;

        let mut parts = Vec::with_capacity(num_parts as usize);
        let mut offset = 0;
        for part_size in (0..num_parts).map(|part_idx| {
            if part_idx < parts_with_one_extra as u32 {
                values_per_part + 1
            } else {
                values_per_part
            }
        }) {
            parts.push(Vec::from_iter(
                ids[offset..(offset + part_size)].iter().copied(),
            ));
            offset += part_size;
        }

        parts
    }

    const BIG_OFFSET: u64 = 10000;

    fn build_mapping(
        row_ids_to_modify: &[u64],
        row_ids_to_remove: &[u64],
        max_id: u64,
    ) -> HashMap<u64, Option<u64>> {
        // Some big number we can add to row ids so they are remapped but don't intersect with anything
        if max_id > BIG_OFFSET {
            panic!("This logic will only work if the max row id is less than BIG_OFFSET");
        }
        row_ids_to_modify
            .iter()
            .copied()
            .map(|val| (val, Some(val + BIG_OFFSET)))
            .chain(row_ids_to_remove.iter().copied().map(|val| (val, None)))
            .collect()
    }

    #[tokio::test]
    async fn remap_ivf_pq_index() {
        // Use small numbers to keep runtime down
        const DIM: u32 = 8;
        const CENTROIDS: u32 = 2;
        const NUM_SUBVECTORS: u32 = 4;
        const NUM_BITS: u32 = 8;
        const INDEX_NAME: &str = "my_index";

        // In this test we create a sample dataset with reliable data, train an IVF PQ index
        // remap the rows, and then verify that we can still search the index and will get
        // back the remapped row ids.

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let mut test_data = WellKnownIvfPqData::new(DIM, CENTROIDS);

        let dataset = Arc::new(test_data.generate_dataset(test_uri).await.unwrap());
        let ivf_params = IvfBuildParams::try_with_centroids(
            CENTROIDS as usize,
            test_data.get_centroids_as_list_arr(),
        )
        .unwrap();
        let pq_params = PQBuildParams::new(NUM_SUBVECTORS as usize, NUM_BITS as usize);

        let uuid = Uuid::new_v4();

        build_ivf_pq_index(
            &dataset,
            WellKnownIvfPqData::COLUMN,
            INDEX_NAME,
            uuid,
            MetricType::L2,
            &ivf_params,
            &pq_params,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();

        // After building the index file, we need to register the index metadata
        // so that the dataset can find it when we try to open it
        let field = dataset.schema().field(WellKnownIvfPqData::COLUMN).unwrap();
        let index_meta = lance_table::format::IndexMetadata {
            uuid,
            dataset_version: dataset.version().version,
            fields: vec![field.id],
            covering_fields: vec![],
            name: INDEX_NAME.to_string(),
            fragment_bitmap: Some(dataset.table_state.fragment_bitmap.as_ref().clone()),
            index_details: Some(Arc::new(vector_index_details_default())),
            index_version: VECTOR_INDEX_VERSION as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };

        // We need to commit this index to the dataset so that it can be found
        use crate::dataset::transaction::{Operation, Transaction};
        let transaction = Transaction::new(
            dataset.version().version,
            Operation::CreateIndex {
                new_indices: vec![index_meta.clone()],
                removed_indices: vec![],
            },
            None,
        );

        // Apply the transaction to register the index
        // Since dataset is Arc<Dataset>, we need to create a mutable reference
        let mut dataset_mut = (*dataset).clone();
        dataset_mut
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let index = dataset_mut
            .open_vector_index(WellKnownIvfPqData::COLUMN, &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        let ivf_index = index.as_any().downcast_ref::<IVFIndex>().unwrap();

        let index_meta = lance_table::format::IndexMetadata {
            uuid,
            dataset_version: 0,
            fields: Vec::new(),
            covering_fields: vec![],
            name: INDEX_NAME.to_string(),
            fragment_bitmap: None,
            index_details: Some(Arc::new(vector_index_details_default())),
            index_version: VECTOR_INDEX_VERSION as i32,
            created_at: None, // Test index, not setting timestamp
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };

        let prefilter = Arc::new(DatasetPreFilter::new(dataset.clone(), &[index_meta], None));

        let is_not_remapped = Some;
        let is_remapped = |row_id| Some(row_id + BIG_OFFSET);
        let is_removed = |_| None;
        let max_id = test_data.get_vectors().len() as u64 / test_data.dim as u64;
        let row_ids = Vec::from_iter(0..max_id);

        // Sanity check to make sure the index we built is behaving correctly.  Any
        // input row, when used as a query, should be found in the results list with
        // the same id
        test_data
            .check_index(ivf_index, prefilter.clone(), &row_ids, is_not_remapped)
            .await;

        // When remapping we change the id of 1/3 of the rows, we remove another 1/3,
        // and we keep 1/3 as they are
        let partitioned_row_ids = partition_ids(row_ids, 3);
        let row_ids_to_modify = &partitioned_row_ids[0];
        let row_ids_to_remove = &partitioned_row_ids[1];
        let row_ids_to_remain = &partitioned_row_ids[2];

        let mapping = build_mapping(row_ids_to_modify, row_ids_to_remove, max_id);

        let new_uuid = Uuid::new_v4();

        remap_index_file(
            &dataset_mut,
            &uuid,
            &new_uuid,
            dataset_mut.version().version,
            ivf_index,
            &RowAddrTranslator::sync(RowAddrRemap::direct(mapping)),
            INDEX_NAME.to_string(),
            WellKnownIvfPqData::COLUMN.to_string(),
            vec![],
        )
        .await
        .unwrap();

        // After remapping the index file, we need to register the new index metadata
        // so that the dataset can find it when we try to open it
        let field = dataset_mut
            .schema()
            .field(WellKnownIvfPqData::COLUMN)
            .unwrap();
        let new_index_meta = lance_table::format::IndexMetadata {
            uuid: new_uuid,
            dataset_version: dataset_mut.version().version,
            fields: vec![field.id],
            covering_fields: vec![],
            name: format!("{}_remapped", INDEX_NAME),
            fragment_bitmap: Some(dataset_mut.table_state.fragment_bitmap.as_ref().clone()),
            index_details: Some(Arc::new(vector_index_details_default())),
            index_version: VECTOR_INDEX_VERSION as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };

        // We need to commit this new index to the dataset so it can be found
        let transaction = Transaction::new(
            dataset_mut.version().version,
            Operation::CreateIndex {
                new_indices: vec![new_index_meta],
                removed_indices: vec![],
            },
            None,
        );

        // Apply the transaction to register the new index
        dataset_mut
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let remapped = dataset_mut
            .open_vector_index(WellKnownIvfPqData::COLUMN, &new_uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let ivf_remapped = remapped.as_any().downcast_ref::<IVFIndex>().unwrap();

        // If the ids were remapped then make sure the new row id is in the results
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_modify,
                is_remapped,
            )
            .await;
        // If the ids were removed then make sure the old row id isn't in the results
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_remove,
                is_removed,
            )
            .await;
        // If the ids were not remapped then make sure they still return the old id
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_remain,
                is_not_remapped,
            )
            .await;
    }

    struct TestPqParams {
        num_sub_vectors: usize,
        num_bits: usize,
    }

    impl TestPqParams {
        fn small() -> Self {
            Self {
                num_sub_vectors: 2,
                num_bits: 8,
            }
        }
    }

    fn fast_ivf_params(num_partitions: usize) -> IvfBuildParams {
        IvfBuildParams {
            num_partitions: Some(num_partitions),
            max_iters: 2,
            sample_rate: 2,
            ..Default::default()
        }
    }

    fn fast_pq_params(num_sub_vectors: usize, num_bits: usize) -> PQBuildParams {
        PQBuildParams {
            num_sub_vectors,
            num_bits,
            max_iters: 2,
            sample_rate: 2,
            ..Default::default()
        }
    }

    fn fast_hnsw_params() -> HnswBuildParams {
        HnswBuildParams::default()
            .max_level(3)
            .num_edges(8)
            .ef_construction(32)
    }

    // Clippy doesn't like that all start with Ivf but we might have some in the future
    // that _don't_ start with Ivf so I feel it is meaningful to keep the prefix
    #[allow(clippy::enum_variant_names)]
    enum TestIndexType {
        IvfPq { pq: TestPqParams },
        IvfHnswPq { pq: TestPqParams, num_edges: usize },
        IvfHnswSq { num_edges: usize },
        IvfFlat,
    }

    struct CreateIndexCase {
        metric_type: MetricType,
        num_partitions: usize,
        dimension: usize,
        index_type: TestIndexType,
    }

    // We test L2 and Dot, because L2 PQ uses residuals while Dot doesn't,
    // so they have slightly different code paths.
    #[tokio::test]
    #[rstest]
    #[case::ivf_pq_l2(CreateIndexCase {
        metric_type: MetricType::L2,
        num_partitions: 2,
        dimension: 16,
        index_type: TestIndexType::IvfPq { pq: TestPqParams::small() },
    })]
    #[case::ivf_pq_dot(CreateIndexCase {
        metric_type: MetricType::Dot,
        num_partitions: 2,
        dimension: 16,
        index_type: TestIndexType::IvfPq { pq: TestPqParams::small() },
    })]
    #[case::ivf_flat(CreateIndexCase { num_partitions: 1, metric_type: MetricType::Dot, dimension: 16, index_type: TestIndexType::IvfFlat })]
    #[case::ivf_hnsw_pq(CreateIndexCase {
        num_partitions: 2,
        metric_type: MetricType::Dot,
        dimension: 16,
        index_type: TestIndexType::IvfHnswPq { pq: TestPqParams::small(), num_edges: 4 },
    })]
    #[case::ivf_hnsw_sq(CreateIndexCase {
        metric_type: MetricType::Dot,
        num_partitions: 2,
        dimension: 16,
        index_type: TestIndexType::IvfHnswSq { num_edges: 4 },
    })]
    async fn test_create_index_nulls(
        #[case] test_case: CreateIndexCase,
        #[values(IndexFileVersion::Legacy, IndexFileVersion::V3)] index_version: IndexFileVersion,
    ) {
        // Most vector search algorithms are approximate, so they may not return all results.
        // IvfFlat is exact under this test's parameters.
        let is_approximate = !matches!(&test_case.index_type, TestIndexType::IvfFlat);
        let mut index_params = match test_case.index_type {
            TestIndexType::IvfPq { pq } => VectorIndexParams::with_ivf_pq_params(
                test_case.metric_type,
                fast_ivf_params(test_case.num_partitions),
                fast_pq_params(pq.num_sub_vectors, pq.num_bits),
            ),
            TestIndexType::IvfHnswPq { pq, num_edges } => {
                VectorIndexParams::with_ivf_hnsw_pq_params(
                    test_case.metric_type,
                    fast_ivf_params(test_case.num_partitions),
                    fast_hnsw_params().num_edges(num_edges),
                    fast_pq_params(pq.num_sub_vectors, pq.num_bits),
                )
            }
            TestIndexType::IvfFlat => VectorIndexParams::with_ivf_flat_params(
                test_case.metric_type,
                fast_ivf_params(test_case.num_partitions),
            ),
            TestIndexType::IvfHnswSq { num_edges } => VectorIndexParams::with_ivf_hnsw_sq_params(
                test_case.metric_type,
                fast_ivf_params(test_case.num_partitions),
                fast_hnsw_params().num_edges(num_edges),
                SQBuildParams::default(),
            ),
        };
        index_params.version(index_version);

        let nrows = 512_usize;
        let data = gen_batch()
            .col(
                "vec",
                array::rand_vec::<Float32Type>(Dimension::from(test_case.dimension as u32)),
            )
            .into_batch_rows(RowCount::from(nrows as u64))
            .unwrap();

        // Make every other row null
        let null_buffer = (0..nrows).map(|i| i % 2 == 0).collect::<BooleanBuffer>();
        let null_buffer = NullBuffer::new(null_buffer);
        let vectors = data["vec"]
            .clone()
            .to_data()
            .into_builder()
            .nulls(Some(null_buffer))
            .build()
            .unwrap();
        let vectors = make_array(vectors);
        let num_non_null = vectors.len() - vectors.logical_null_count();
        let data = RecordBatch::try_new(data.schema(), vec![vectors]).unwrap();

        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![data])
            .await
            .unwrap();

        // Create index
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &index_params, false)
            .await
            .unwrap();

        let query = vec![0.0; test_case.dimension]
            .into_iter()
            .collect::<Float32Array>();
        let results = dataset
            .scan()
            .nearest("vec", &query, nrows)
            .unwrap()
            .ef(nrows)
            .minimum_nprobes(2)
            .try_into_batch()
            .await
            .unwrap();
        // Use a relaxed assertion for approximate indexes.
        if is_approximate {
            let recall = results.num_rows() as f32 / num_non_null as f32;
            assert!(
                recall >= 0.5,
                "Recall {} below threshold {} ({}/{})",
                recall,
                0.5,
                results.num_rows(),
                num_non_null,
            );
        } else {
            assert_eq!(results.num_rows(), num_non_null);
        }
        assert_eq!(results["vec"].logical_null_count(), 0);
    }

    #[tokio::test]
    async fn test_index_lifecycle_nulls() {
        // Generate random data with nulls
        let nrows = 2_000;
        let dims = 32;
        let data = gen_batch()
            .col(
                "vec",
                array::rand_vec::<Float32Type>(Dimension::from(dims as u32)).with_random_nulls(0.5),
            )
            .into_batch_rows(RowCount::from(nrows))
            .unwrap();
        let num_non_null = data["vec"].len() - data["vec"].logical_null_count();

        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![data])
            .await
            .unwrap();

        // Create index
        let index_params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams::new(2, 8),
        );
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &index_params, false)
            .await
            .unwrap();

        // Check that the index is working
        async fn check_index(dataset: &Dataset, num_non_null: usize, dims: usize) {
            let query = vec![0.0; dims].into_iter().collect::<Float32Array>();
            let results = dataset
                .scan()
                .nearest("vec", &query, 2_000)
                .unwrap()
                .minimum_nprobes(2)
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(results.num_rows(), num_non_null);
        }
        check_index(&dataset, num_non_null, dims).await;

        // Append more data
        let data = gen_batch()
            .col(
                "vec",
                array::rand_vec::<Float32Type>(Dimension::from(dims as u32)).with_random_nulls(0.5),
            )
            .into_batch_rows(RowCount::from(500))
            .unwrap();
        let num_non_null = data["vec"].len() - data["vec"].logical_null_count() + num_non_null;
        let mut dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![data])
            .await
            .unwrap();
        check_index(&dataset, num_non_null, dims).await;

        // Optimize the index
        dataset.optimize_indices(&Default::default()).await.unwrap();
        check_index(&dataset, num_non_null, dims).await;
    }

    #[tokio::test]
    async fn test_create_ivf_pq_cosine() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let pq_params = PQBuildParams::new(4, 8);

        let params =
            VectorIndexParams::with_ivf_pq_params(MetricType::Cosine, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let sample_query = vector_array.value(10);
        let query = sample_query.as_primitive::<Float32Type>();
        let results = dataset
            .scan()
            .nearest("vector", query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(5, results[0].num_rows());
        for batch in results.iter() {
            let dist = &batch["_distance"];
            dist.as_primitive::<Float32Type>()
                .values()
                .iter()
                .for_each(|v| {
                    assert!(
                        (0.0..2.0).contains(v),
                        "Expect cosine value in range [0.0, 2.0], got: {}",
                        v
                    )
                });
        }
    }

    #[rstest]
    #[case::l2(MetricType::L2, [4.5f32, 6.0])]
    #[case::cosine(MetricType::Cosine, [0.6f32, 0.8])]
    #[tokio::test]
    async fn train_ivf_from_sample_preprocesses_invalid_vectors(
        #[case] metric: MetricType,
        #[case] expected: [f32; 2],
    ) {
        let values = Float32Array::from(vec![
            Some(3.0),
            Some(4.0),
            Some(6.0),
            Some(8.0),
            Some(f32::NAN),
            Some(1.0),
            Some(f32::INFINITY),
            Some(1.0),
            None,
            Some(1.0),
            Some(100.0),
            Some(100.0),
        ]);
        let sample = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, true, true, true, true, false])),
        );
        let model =
            train_ivf_model_from_sample(sample, metric, &IvfBuildParams::new(1), noop_progress())
                .await
                .unwrap();
        let centroids = model.centroids.as_ref().unwrap();
        assert_eq!(centroids.len(), 1);
        for (&actual, expected) in centroids
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .zip(expected)
        {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[rstest]
    #[case::empty(Vec::<Option<f32>>::new())]
    #[case::all_null(vec![None, None])]
    #[case::all_invalid(vec![Some(f32::NAN), Some(1.0)])]
    #[tokio::test]
    async fn train_ivf_from_sample_rejects_insufficient_valid_vectors(
        #[case] values: Vec<Option<f32>>,
    ) {
        let sample =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), 2).unwrap();
        let error = train_ivf_model_from_sample(
            sample,
            MetricType::L2,
            &IvfBuildParams::new(1),
            noop_progress(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Unprocessable { .. }));
        assert!(
            error
                .to_string()
                .contains("cannot train 1 centroids with 0 vectors")
        );
    }

    #[rstest]
    #[case::missing_partitions(IvfBuildParams::default(), "num_partitions")]
    #[case::zero_partitions(IvfBuildParams::new(0), "num_partitions")]
    #[case::zero_sample_rate(IvfBuildParams { sample_rate: 0, ..IvfBuildParams::new(1) }, "sample_rate")]
    #[case::zero_iterations(IvfBuildParams { max_iters: 0, ..IvfBuildParams::new(1) }, "max_iters")]
    #[case::streaming(IvfBuildParams { streaming_sample_rate: Some(1), ..IvfBuildParams::new(1) }, "streaming_sample_rate")]
    #[case::overflow(IvfBuildParams { sample_rate: usize::MAX, ..IvfBuildParams::new(2) }, "overflows")]
    #[tokio::test]
    async fn train_ivf_from_sample_rejects_invalid_parameters(
        #[case] params: IvfBuildParams,
        #[case] message: &str,
    ) {
        let sample = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]),
            2,
        )
        .unwrap();
        let error = train_ivf_model_from_sample(sample, MetricType::L2, &params, noop_progress())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[tokio::test]
    async fn train_ivf_from_sample_retrains_initial_centroids() {
        let sample = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![3.0, 4.0, 5.0, 6.0]),
            2,
        )
        .unwrap();
        let mut params =
            IvfBuildParams::try_with_centroids(1, Arc::new(sample.slice(0, 1))).unwrap();
        let error =
            train_ivf_model_from_sample(sample.clone(), MetricType::L2, &params, noop_progress())
                .await
                .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("retrain=true"));
        params.retrain = true;
        let model = train_ivf_model_from_sample(sample, MetricType::L2, &params, noop_progress())
            .await
            .unwrap();
        assert_eq!(
            model
                .centroids
                .as_ref()
                .unwrap()
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .as_ref(),
            &[4.0, 5.0]
        );
    }

    #[tokio::test]
    async fn train_ivf_from_sample_builds_searchable_index() {
        let sample = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..64)
                    .flat_map(|i| [i as f32, (i as f32) * 0.5])
                    .collect::<Vec<_>>(),
            ),
            2,
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(UInt32Array::from((0..64).collect::<Vec<u32>>())) as ArrayRef,
            ),
            ("vector", Arc::new(sample.clone()) as ArrayRef),
        ])
        .unwrap();
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 32,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let query = Float32Array::from(vec![17.25, 8.5]);
        let expected = dataset
            .scan()
            .nearest("vector", &query, 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let model = train_ivf_model_from_sample(
            sample,
            MetricType::L2,
            &IvfBuildParams::new(2),
            noop_progress(),
        )
        .await
        .unwrap();
        let ivf =
            IvfBuildParams::try_with_centroids(2, Arc::new(model.centroids.unwrap())).unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::with_ivf_flat_params(MetricType::L2, ivf),
                false,
            )
            .await
            .unwrap();
        let found = dataset
            .scan()
            .nearest("vector", &query, 10)
            .unwrap()
            .minimum_nprobes(2)
            .try_into_batch()
            .await
            .unwrap();
        let expected_ids = expected["id"]
            .as_primitive::<arrow_array::types::UInt32Type>()
            .values();
        let found_ids = found["id"]
            .as_primitive::<arrow_array::types::UInt32Type>()
            .values();
        let recall = expected_ids
            .iter()
            .filter(|id| found_ids.contains(id))
            .count() as f64
            / 10.0;
        assert_eq!(recall, 1.0);
    }

    #[tokio::test]
    async fn test_build_ivf_model_l2() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (dataset, _) = generate_test_dataset(test_uri, 1000.0..1100.0).await;

        let ivf_params = IvfBuildParams::new(2);
        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            DIM,
            MetricType::L2,
            &ivf_params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(2, ivf_model.centroids.as_ref().unwrap().len());
        assert_eq!(32, ivf_model.centroids.as_ref().unwrap().value_length());
        assert_eq!(2, ivf_model.num_partitions());
        assert!(ivf_model.loss().is_some_and(|loss| loss > 0.0));

        // All centroids values should be in the range [1000, 1100]
        ivf_model
            .centroids
            .unwrap()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((1000.0..1100.0).contains(v));
            });

        // An index build skips the sample loss its shuffle replaces.
        let without_loss = build_ivf_model_with_sample_loss(
            &dataset,
            "vector",
            DIM,
            MetricType::L2,
            &ivf_params,
            None,
            lance_index::progress::noop_progress(),
            SampleLoss::Skip,
        )
        .await
        .unwrap();
        assert_eq!(2, without_loss.num_partitions());
        assert!(without_loss.loss().is_none());
    }

    #[tokio::test]
    async fn test_build_ivf_model_cosine() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (dataset, _) = generate_test_dataset(test_uri, 1000.0..1100.0).await;

        let ivf_params = IvfBuildParams::new(2);
        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            DIM,
            MetricType::Cosine,
            &ivf_params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(2, ivf_model.centroids.as_ref().unwrap().len());
        assert_eq!(32, ivf_model.centroids.as_ref().unwrap().value_length());
        assert_eq!(2, ivf_model.num_partitions());

        // All centroids values should be in the range [1000, 1100]
        ivf_model
            .centroids
            .unwrap()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!(
                    (-1.0..1.0).contains(v),
                    "Expect cosine value in range [-1.0, 1.0], got: {}",
                    v
                );
            });
    }

    #[tokio::test]
    async fn test_create_ivf_pq_dot() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::Dot, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let sample_query = vector_array.value(10);
        let query = sample_query.as_primitive::<Float32Type>();
        let results = dataset
            .scan()
            .nearest("vector", query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(5, results[0].num_rows());

        for batch in results.iter() {
            let dist = &batch["_distance"];
            dist.as_primitive::<Float32Type>()
                .values()
                .iter()
                .for_each(|v| {
                    assert!(
                        (-2.0 * DIM as f32..0.0).contains(v),
                        "Expect dot product value in range [-2.0 * DIM, 0.0], got: {}",
                        v
                    )
                });
        }
    }

    #[tokio::test]
    async fn test_build_ivf_model_streaming_training() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(512), BatchCount::from(2));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let mut params = IvfBuildParams::new(8);
        params.sample_rate = 16;
        params.streaming_sample_rate = Some(4);
        params.streaming_refine_passes = 1;
        params.max_iters = 2;

        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            32,
            MetricType::L2,
            &params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();

        assert_eq!(ivf_model.num_partitions(), 8);
        assert_eq!(ivf_model.dimension(), 32);
        assert!(ivf_model.loss().is_none());
        assert!(
            compute_test_ivf_loss(&dataset, "vector", &ivf_model)
                .await
                .is_finite()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_ivf_error_joins_progress_worker() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>((DIM as u32).into()),
            )
            .into_reader_rows(RowCount::from(128), BatchCount::from(2));
        let dataset = Arc::new(Dataset::write(reader, &uri, None).await.unwrap());

        let mut params = IvfBuildParams::new(2);
        params.sample_rate = 8;
        params.streaming_sample_rate = Some(4);
        params.max_iters = 2;

        let is_callback_active = Arc::new(AtomicBool::new(false));
        let callback_started = Arc::new(Notify::new());
        let release_callback = Arc::new(Notify::new());
        let progress = Arc::new(BlockingFailingProgress {
            is_first_call: AtomicBool::new(true),
            is_callback_active: is_callback_active.clone(),
            callback_started: callback_started.clone(),
            release_callback: release_callback.clone(),
        });
        let mut trainer = tokio::spawn({
            let dataset = dataset.clone();
            async move {
                train_streaming_ivf_model(
                    dataset.as_ref(),
                    "vector",
                    DIM - 1,
                    MetricType::L2,
                    &params,
                    None,
                    progress,
                )
                .await
            }
        });

        tokio::time::timeout(Duration::from_secs(5), callback_started.notified())
            .await
            .expect("streaming IVF did not report progress");
        assert!(
            is_callback_active.load(Ordering::SeqCst),
            "progress callback should still be active"
        );

        let returned_while_callback_active =
            tokio::time::timeout(Duration::from_millis(100), &mut trainer).await;
        release_callback.notify_one();
        assert!(
            returned_while_callback_active.is_err(),
            "streaming IVF returned while its progress callback was still active"
        );

        let error = tokio::time::timeout(Duration::from_secs(5), trainer)
            .await
            .expect("streaming IVF did not return after the callback was released")
            .expect("streaming IVF trainer task panicked")
            .expect_err("the mismatched dimension should fail training");
        assert!(
            matches!(error, Error::Arrow { .. }),
            "expected the primary centroid conversion error, got: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("Incorrect length of values buffer for FixedSizeListArray"),
            "unexpected training error: {error}"
        );
        assert!(
            !is_callback_active.load(Ordering::SeqCst),
            "progress callback remained active after streaming IVF returned"
        );
    }

    /// Padding the partition count with clones of the heaviest cluster left
    /// partitions that started empty; reject like the flat hierarchical
    /// trainer. A coreset smaller than the partition count and a coreset whose
    /// rows are identical both reach the shortfall, and each gets its own
    /// advice.
    #[rstest::rstest]
    #[case::fewer_rows_than_partitions(6, 16, "Increase sample_rate")]
    #[case::enough_rows_all_identical(32, 16, "provide more diverse data")]
    #[test]
    fn test_weighted_hierarchical_rejects_padding_with_duplicates(
        #[case] num_points: usize,
        #[case] target_k: usize,
        #[case] advice: &str,
    ) {
        let dimension = 8;
        // Every row identical in the second case, a ramp in the first: either
        // way the trainer cannot reach `target_k` non-empty clusters.
        let values: Vec<f32> = if num_points < target_k {
            (0..num_points * dimension)
                .map(|i| i as f32 * 0.01)
                .collect()
        } else {
            vec![0.5; num_points * dimension]
        };
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(values, dimension).unwrap();
        let params = WeightedHierarchicalKMeansParams {
            dimension,
            target_k,
            metric_policy: &L2_METRIC_POLICY,
            max_iters: 2,
            on_progress: Arc::new(|_, _| {}),
        };
        let err = train_weighted_hierarchical_kmeans::<Float32Type>(
            &data,
            &vec![1.0; num_points],
            &vec![0.1; num_points],
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("Cannot create {target_k} IVF partitions")),
            "got: {msg}"
        );
        // The row count belongs in the message: it is the coreset size, not
        // the dataset's, so a reader can tell the two shortfalls apart.
        assert!(
            msg.contains(&format!("{num_points} weighted coreset rows")),
            "got: {msg}"
        );
        assert!(msg.contains(advice), "got: {msg}");
    }

    /// Under Dot, positive 1-D rows all go to the largest centroid, so every
    /// k-means split leaves its rows on one side. The rows are distinct, so the
    /// trainer must still reach `target_k` distinct centroids rather than
    /// reject the request, for f32 and f64 centroids alike.
    fn bisects_clusters_kmeans_cannot_split<T: StreamingKMeansArrowType>() {
        let num_points = 8;
        let target_k = 4;
        let values = (1..=num_points)
            .map(|i| T::Native::from_f64(i as f64).unwrap())
            .collect::<Vec<_>>();
        let data = streaming_kmeans_fsl_from_values::<T>(values, 1).unwrap();
        let params = WeightedHierarchicalKMeansParams {
            dimension: 1,
            target_k,
            metric_policy: &DOT_METRIC_POLICY,
            max_iters: 5,
            on_progress: Arc::new(|_, _| {}),
        };

        let centroids = train_weighted_hierarchical_kmeans::<T>(
            &data,
            &vec![1.0; num_points],
            &vec![0.0; num_points],
            &params,
        )
        .unwrap();
        let mut values = centroids
            .values()
            .as_primitive::<T>()
            .values()
            .iter()
            .map(|value| value.as_())
            .collect::<Vec<f64>>();
        values.sort_by(f64::total_cmp);
        values.dedup();
        assert_eq!(values.len(), target_k, "centroids: {values:?}");
    }

    #[rstest::rstest]
    #[case::f32(bisects_clusters_kmeans_cannot_split::<Float32Type> as fn())]
    #[case::f64(bisects_clusters_kmeans_cannot_split::<Float64Type> as fn())]
    fn test_weighted_hierarchical_bisects_clusters_kmeans_cannot_split(#[case] check: fn()) {
        check();
    }

    #[tokio::test]
    async fn test_build_ivf_model_streaming_dot_coreset_training() {
        const DIMENSION: usize = 4;
        const NUM_PARTITIONS: usize = 257;

        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIMENSION as i32,
            ),
            false,
        )]));
        let values = Float32Array::from_iter_values((0..1_024).flat_map(|row| {
            let angle = TAU * row as f32 / 1_024.0;
            [
                angle.cos(),
                angle.sin(),
                (3.0 * angle).cos(),
                (3.0 * angle).sin(),
            ]
        }));
        let vectors = FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let mut params = IvfBuildParams::new(NUM_PARTITIONS);
        params.sample_rate = 2;
        params.streaming_sample_rate = Some(1);
        params.max_iters = 1;

        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            DIMENSION,
            MetricType::Dot,
            &params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();

        assert_eq!(ivf_model.num_partitions(), NUM_PARTITIONS);
        assert_eq!(ivf_model.dimension(), DIMENSION);
        assert!(
            ivf_model
                .centroids
                .unwrap()
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    /// Regression test for a hang in the streaming *coreset* trainer
    /// (`train_streaming_coreset_ivf_model`, taken when `num_partitions > 256`).
    ///
    /// That function spawns a `progress_worker` task that loops on
    /// `progress_rx.recv()` and only terminates once every clone of the mpsc
    /// sender is dropped. The `on_progress` closure owns a sender clone, and
    /// `WeightedHierarchicalKMeansParams` used to retain an `on_progress` clone
    /// that outlived the `progress_worker.await` at the end of the function.
    /// With a live sender remaining, `recv()` never returned `None`, the worker
    /// never finished, and the trainer hung forever after all compute was done.
    ///
    /// The build is wrapped in a timeout so the regression fails fast rather
    /// than hanging the test process indefinitely.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_ivf_training_terminates() {
        use lance_index::progress::IndexBuildProgress;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Duration;

        #[derive(Debug, Default)]
        struct CountingProgress {
            progress_calls: AtomicU64,
        }

        #[async_trait::async_trait]
        impl IndexBuildProgress for CountingProgress {
            async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
                Ok(())
            }
            async fn stage_progress(&self, _: &str, _: u64) -> Result<()> {
                self.progress_calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            async fn stage_complete(&self, _: &str) -> Result<()> {
                Ok(())
            }
        }

        const SMALL_DIM: usize = 8;

        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>((SMALL_DIM as u32).into()),
            )
            .into_reader_rows(RowCount::from(2048), BatchCount::from(4));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        // > 256 partitions routes through `train_streaming_coreset_ivf_model`.
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 8;
        params.streaming_sample_rate = Some(4);
        params.streaming_refine_passes = 1;
        params.max_iters = 2;

        let progress = Arc::new(CountingProgress::default());

        let ivf_model = tokio::time::timeout(
            Duration::from_secs(120),
            build_ivf_model(
                &dataset,
                "vector",
                SMALL_DIM,
                MetricType::L2,
                &params,
                None,
                progress.clone(),
            ),
        )
        .await
        .expect(
            "streaming coreset IVF training hung: progress worker never terminated after training",
        )
        .unwrap();

        assert_eq!(ivf_model.num_partitions(), 257);
        assert_eq!(ivf_model.dimension(), SMALL_DIM);
        assert_eq!(
            ivf_model.centroids.as_ref().unwrap().value_type(),
            DataType::Float32
        );
        // The progress worker must have processed reports and then joined
        // cleanly (proven by `build_ivf_model` returning at all).
        assert!(
            progress.progress_calls.load(Ordering::Relaxed) > 0,
            "expected the progress worker to receive at least one report"
        );
    }

    /// f16 and f64 columns used to survive the streaming trainer's dtype
    /// normalization unchanged (`convert_to_floating_point` only converts
    /// integer types) and then hit the unconditional Float32 downcast in the
    /// coreset path, panicking the build instead of training.
    ///
    /// The two cases also take different routes into the trainer: a
    /// non-nullable column gets the fixed-range sampler, a nullable one the
    /// resampling path, and each has its own normalization site.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_ivf_training_float16() {
        let values = generate_random_array_with_seed::<Float16Type>(2048 * 8, [22; 32]);
        streaming_coreset_training_completes(values, 8, false).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_ivf_training_float64_nullable() {
        let values = generate_random_array_with_seed::<Float64Type>(2048 * 8, [22; 32]);
        streaming_coreset_training_completes(values, 8, true).await;
    }

    async fn streaming_coreset_training_completes<T: arrow_array::Array + 'static>(
        values: T,
        dimension: usize,
        nullable: bool,
    ) {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let expected_type = values.data_type().clone();
        let fsl = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            fsl.data_type().clone(),
            nullable,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 8;
        params.streaming_sample_rate = Some(4);
        params.streaming_refine_passes = 1;
        params.max_iters = 2;

        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            dimension,
            MetricType::L2,
            &params,
            None,
            noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(ivf_model.num_partitions(), 257);
        assert_eq!(ivf_model.dimension(), dimension);
        let centroids = ivf_model.centroids_array().expect("trained model");
        assert_eq!(centroids.value_type(), expected_type);
        let centroid_values = arrow::compute::cast(centroids.values(), &DataType::Float64).unwrap();
        let centroid_values = centroid_values.as_primitive::<Float64Type>();
        // The generator draws from [0, 1), and a centroid is a weighted mean of
        // training rows, so anything outside that range means the cast produced
        // the wrong values rather than the wrong type.
        assert!(
            centroid_values
                .values()
                .iter()
                .all(|v| (0.0..=1.0).contains(v)),
            "centroid outside the training range: {:?}",
            centroid_values
                .values()
                .iter()
                .find(|v| !(0.0..=1.0).contains(*v)),
        );
    }

    /// Read the f32 values of a training chunk row by row.
    fn f32_rows(array: &FixedSizeListArray) -> Vec<Vec<f32>> {
        let dimension = array.value_length() as usize;
        let values = array.values().as_primitive::<Float32Type>().values();
        values
            .chunks(dimension)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>()
    }

    /// `prepare_streaming_coreset_sample` is the only place the streaming trainers
    /// normalize dtypes, so it has to keep the values, the row boundaries and
    /// the nulls intact for every type it accepts, and reject the rest with an
    /// error rather than let a downstream downcast panic.
    #[test]
    fn test_prepare_streaming_coreset_sample_converts_supported_types() {
        let expected = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];

        let f32_input = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]),
            2,
        )
        .unwrap();
        let out = prepare_streaming_coreset_sample::<Float32Type>(f32_input).unwrap();
        assert_eq!(f32_rows(&out), expected, "Float32 must pass through");

        let f16_input = FixedSizeListArray::try_new_from_values(
            Float16Array::from(vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ]),
            2,
        )
        .unwrap();
        let out = prepare_streaming_coreset_sample::<Float32Type>(f16_input).unwrap();
        assert_eq!(f32_rows(&out), expected, "Float16 must widen exactly");
        assert_eq!(out.value_type(), DataType::Float32);

        let f64_input = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]),
            2,
        )
        .unwrap();
        let out = prepare_streaming_coreset_sample::<Float64Type>(f64_input).unwrap();
        assert_eq!(
            streaming_kmeans_values::<Float64Type>(&out).unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );

        let i8_input =
            FixedSizeListArray::try_new_from_values(Int8Array::from(vec![1i8, 2, 3, 4]), 2)
                .unwrap();
        let out = prepare_streaming_coreset_sample::<Float32Type>(i8_input).unwrap();
        assert_eq!(f32_rows(&out), expected, "Int8 must convert");
    }

    /// The samplers hand over sliced chunks, so the cast has to follow the
    /// slice rather than the whole underlying buffer.
    #[test]
    fn test_prepare_streaming_coreset_sample_follows_a_slice() {
        let input = FixedSizeListArray::try_new_from_values(
            Float16Array::from_iter_values(
                [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]
                    .into_iter()
                    .map(f16::from_f32),
            ),
            2,
        )
        .unwrap();
        let out = prepare_streaming_coreset_sample::<Float32Type>(input.slice(1, 2)).unwrap();
        assert_eq!(
            f32_rows(&out),
            vec![vec![20.0f32, 21.0], vec![30.0, 31.0]],
            "a sliced chunk must map to the rows it points at"
        );
    }

    #[test]
    fn test_prepare_streaming_coreset_sample_keeps_nulls() {
        let values = Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::ZERO,
            f16::ZERO,
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ]);
        let field = Arc::new(Field::new("item", DataType::Float16, false));
        let nulls = NullBuffer::from(vec![true, false, true]);
        let input = FixedSizeListArray::try_new(field, 2, Arc::new(values), Some(nulls)).unwrap();
        let out = prepare_streaming_coreset_sample::<Float32Type>(input).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.null_count(), 1, "the null row must survive the cast");
        assert!(out.is_null(1), "the null must stay on the same row");
    }

    #[test]
    fn test_prepare_streaming_coreset_sample_preserves_large_f64_values() {
        let values = Float64Array::from(vec![1.0, 2.0, 1e39, 4.0, 0.0, 0.0, 5.0, 6.0]);
        let field = Arc::new(Field::new("item", DataType::Float64, false));
        let nulls = NullBuffer::from(vec![true, true, false, true]);
        let input = FixedSizeListArray::try_new(field, 2, Arc::new(values), Some(nulls)).unwrap();
        let out = prepare_streaming_coreset_sample::<Float64Type>(input).unwrap();
        assert_eq!(
            streaming_kmeans_values::<Float64Type>(&out).unwrap(),
            &[1.0, 2.0, 1e39, 4.0, 0.0, 0.0, 5.0, 6.0]
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out.null_count(), 1);
        assert!(out.is_null(2));
    }

    #[test]
    fn test_prepare_streaming_coreset_sample_rejects_unsupported_types() {
        let input =
            FixedSizeListArray::try_new_from_values(UInt16Array::from(vec![1u16, 2, 3, 4]), 2)
                .unwrap();
        let error = prepare_streaming_coreset_sample::<Float32Type>(input).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        assert!(
            error.to_string().contains("UInt16"),
            "the error must name the rejected type: {error}"
        );
    }

    #[rstest]
    #[case::without_raw_refinement(0)]
    #[case::with_raw_refinement(1)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_hamming_coreset_ivf_training(#[case] refine_passes: usize) {
        const DIMENSION: usize = 2;
        const NUM_PARTITIONS: usize = 257;

        let test_dir = TempStrDir::default();
        let uri = format!("{}/hamming", test_dir.as_str());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                DIMENSION as i32,
            ),
            false,
        )]));
        let values = arrow_array::UInt8Array::from(
            (0..600_u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>(),
        );
        let vectors = FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();

        let mut params = IvfBuildParams::new(NUM_PARTITIONS);
        params.sample_rate = 2;
        params.streaming_sample_rate = Some(1);
        params.streaming_refine_passes = refine_passes;
        params.max_iters = 2;

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        for selected_fragments in [None, Some(fragment_ids.as_slice())] {
            let model = build_ivf_model(
                &dataset,
                "vector",
                DIMENSION,
                MetricType::Hamming,
                &params,
                selected_fragments,
                lance_index::progress::noop_progress(),
            )
            .await
            .unwrap();

            let centroids = model.centroids.unwrap();
            assert_eq!(centroids.len(), NUM_PARTITIONS);
            assert_eq!(centroids.value_length(), DIMENSION as i32);
            assert_eq!(centroids.value_type(), DataType::UInt8);
            let values = centroids.values().as_primitive::<UInt8Type>().values();
            assert_eq!(
                values
                    .as_chunks::<DIMENSION>()
                    .0
                    .iter()
                    .collect::<HashSet<_>>()
                    .len(),
                NUM_PARTITIONS
            );
        }
    }

    #[rstest]
    #[case::fixed_l2_no_refine(MetricType::L2, 0, false)]
    #[case::fixed_cosine_refine(MetricType::Cosine, 1, false)]
    #[case::fixed_dot_refine(MetricType::Dot, 1, false)]
    #[case::fragment_dot_refine(MetricType::Dot, 1, true)]
    #[case::fragment_l2_refine(MetricType::L2, 1, true)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_f64_centroids(
        #[case] metric_type: MetricType,
        #[case] refine_passes: usize,
        #[case] use_fragment_ids: bool,
    ) {
        use std::time::Duration;

        const SMALL_DIM: usize = 8;

        let test_dir = TempStrDir::default();
        let uri = format!("{}/ds", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col(
                "vector",
                array::rand_vec::<Float64Type>((SMALL_DIM as u32).into()),
            )
            .into_reader_rows(RowCount::from(1536), BatchCount::from(3));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();
        let fragment_ids = use_fragment_ids.then(|| {
            dataset
                .get_fragments()
                .iter()
                .map(|fragment| fragment.id() as u32)
                .collect::<Vec<_>>()
        });

        // > 256 partitions selects streaming coreset training.
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 4;
        params.streaming_sample_rate = Some(2);
        params.streaming_refine_passes = refine_passes;
        params.max_iters = 1;

        let ivf_model = tokio::time::timeout(
            Duration::from_secs(120),
            build_ivf_model(
                &dataset,
                "vector",
                SMALL_DIM,
                metric_type,
                &params,
                fragment_ids.as_deref(),
                lance_index::progress::noop_progress(),
            ),
        )
        .await
        .expect("streaming coreset f64 training timed out")
        .unwrap();

        assert_eq!(ivf_model.num_partitions(), 257);
        assert_eq!(ivf_model.dimension(), SMALL_DIM);
        let centroids = ivf_model.centroids.as_ref().unwrap();
        assert_eq!(centroids.value_type(), DataType::Float64);
        let centroid_values = centroids.values().as_primitive::<Float64Type>().values();
        assert!(centroid_values.iter().all(|value| value.is_finite()));
        assert!(
            centroid_values
                .iter()
                .any(|value| *value != *value as f32 as f64),
            "expected at least one f64 centroid coordinate to retain precision beyond f32"
        );
    }

    // 257 partitions at sample rate 8 request 2,056 vectors, more than any case
    // holds; a nullable column leaves every other vector null, and the first
    // step of the NaN case reads only vectors the sampler drops as non-finite
    #[rstest]
    #[case::fixed_dot_small_remainder(MetricType::Dot, false, 0, 1_050, None, 4)]
    #[case::fixed_steps_below_partition_count(MetricType::Dot, false, 0, 600, None, 2)]
    #[case::fixed_dot_empty_first_step(MetricType::Dot, false, 262, 1_050, None, 2)]
    #[case::fixed_l2_empty_first_step(MetricType::L2, false, 262, 1_050, None, 2)]
    #[case::nullable_l2_steps_below_partition_count(MetricType::L2, true, 0, 600, None, 4)]
    #[case::nullable_dot_small_remainder(MetricType::Dot, true, 0, 1_050, None, 4)]
    #[case::nullable_dot_boundary(MetricType::Dot, true, 0, 520, None, 2)]
    #[case::nullable_dot_fragments(MetricType::Dot, true, 0, 4_000, Some(vec![0, 1, 3]), 2)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_trains_on_fewer_rows_than_sample(
        #[case] metric_type: MetricType,
        #[case] nullable: bool,
        #[case] nan_rows: usize,
        #[case] num_rows: usize,
        #[case] fragment_ids: Option<Vec<u32>>,
        #[case] streaming_sample_rate: usize,
    ) {
        let test_dir = TempStrDir::default();
        let mut values = generate_random_array_with_seed::<Float32Type>(num_rows * 8, [7; 32])
            .values()
            .to_vec();
        values[..nan_rows * 8].fill(f32::NAN);
        let nulls = nullable.then(|| NullBuffer::from_iter((0..num_rows).map(|row| row % 2 == 0)));
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            8,
            Arc::new(PrimitiveArray::<Float32Type>::from(values)),
            nulls,
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            vectors.data_type().clone(),
            nullable,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &format!("{}/rows", test_dir.as_str()),
            Some(WriteParams {
                max_rows_per_file: 1_000,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 8;
        params.streaming_sample_rate = Some(streaming_sample_rate);
        params.max_iters = 1;

        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            8,
            metric_type,
            &params,
            fragment_ids.as_deref(),
            noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(ivf_model.num_partitions(), 257);
    }

    #[rstest]
    #[case::l2(MetricType::L2)]
    #[case::cosine(MetricType::Cosine)]
    #[tokio::test]
    async fn test_streaming_coreset_f64_index_roundtrip(#[case] metric_type: MetricType) {
        const NUM_PARTITIONS: usize = 257;
        const EXPECTED_ROW: usize = 321;
        const K: usize = 5;

        let test_dir = TempStrDir::default();
        let batch = gen_batch()
            .with_seed(42.into())
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::rand_vec::<Float64Type>(4.into()))
            .into_batch_rows(600.into())
            .unwrap();
        let query = batch["vector"].as_fixed_size_list().value(EXPECTED_ROW);
        let mut dataset = InsertBuilder::new(test_dir.as_str())
            .with_params(&WriteParams {
                max_rows_per_file: 300,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        // Both settings are needed to select streaming coreset training.
        let mut ivf_params = IvfBuildParams::new(NUM_PARTITIONS);
        ivf_params.sample_rate = 2;
        ivf_params.streaming_sample_rate = Some(1);
        ivf_params.max_iters = 1;
        let params = VectorIndexParams::with_ivf_flat_params(metric_type, ivf_params);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        drop(dataset);

        let dataset = Dataset::open(test_dir.as_str()).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let index = dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let model = index.ivf_model();
        assert_eq!(model.num_partitions(), NUM_PARTITIONS);
        let centroids = model.centroids.as_ref().unwrap();
        assert_eq!(centroids.value_type(), DataType::Float64);
        assert!(
            centroids
                .values()
                .as_primitive::<Float64Type>()
                .values()
                .iter()
                .any(|value| *value != *value as f32 as f64)
        );

        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", query.as_ref(), K)
            .unwrap()
            .distance_metric(metric_type)
            .minimum_nprobes(16);
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(plan.contains("ANNIvfPartition"), "{plan}");
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), K);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert_eq!(ids[0], EXPECTED_ROW as u64);

        let ground_truth = dataset
            .scan()
            .nearest("vector", query.as_ref(), K)
            .unwrap()
            .distance_metric(metric_type)
            .use_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let expected_ids = ground_truth["id"].as_primitive::<UInt64Type>().values();
        let recall = ids.iter().filter(|id| expected_ids.contains(id)).count() as f32 / K as f32;
        assert!(recall >= 0.5, "recall: {recall}, row ids: {ids:?}");
    }

    #[rstest]
    #[case::float16_streaming_fixed(DataType::Float16, false, Some(1), 1, "rq")]
    #[case::float16_streaming_nullable(DataType::Float16, true, Some(1), 1, "rq")]
    #[case::float16_streaming_without_refinement(DataType::Float16, false, Some(1), 0, "rq")]
    #[case::float32_streaming(DataType::Float32, false, Some(1), 1, "rq")]
    #[case::float16_in_memory(DataType::Float16, false, None, 0, "rq")]
    // Flat and SQ fail partition assignment, PQ fails residuals, without
    // Float16 centroids from streaming coreset training.
    #[case::float16_streaming_flat(DataType::Float16, false, Some(1), 1, "flat")]
    #[case::float16_streaming_pq(DataType::Float16, false, Some(1), 1, "pq")]
    #[case::float16_streaming_sq(DataType::Float16, false, Some(1), 1, "sq")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_ivf_build_and_search_float_types(
        #[case] value_type: DataType,
        #[case] is_nullable: bool,
        #[case] streaming_sample_rate: Option<usize>,
        #[case] refine_passes: usize,
        #[case] index: &str,
    ) {
        const NUM_ROWS: usize = 600;
        const NUM_PARTITIONS: usize = 257;
        const DIMENSION: usize = 32;
        const EXPECTED_ROW: usize = 17;
        const K: usize = 5;

        let test_dir = TempStrDir::default();
        let batch = gen_batch()
            .with_seed(42.into())
            .col("id", array::step::<UInt64Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>((DIMENSION as u32).into()),
            )
            .into_batch_rows((NUM_ROWS as u64).into())
            .unwrap();
        let vectors = batch["vector"].as_fixed_size_list();
        let values = arrow::compute::cast(vectors.values(), &value_type).unwrap();
        let vectors = FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
        let query = vectors.value(EXPECTED_ROW);
        let batch = batch
            .replace_column_schema_by_name("vector", vectors.data_type().clone(), Arc::new(vectors))
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            batch.schema().field(0).clone(),
            batch.schema().field(1).clone().with_nullable(is_nullable),
        ]));
        let batch = RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap();
        let mut dataset = InsertBuilder::new(test_dir.as_str())
            .with_params(&WriteParams {
                max_rows_per_file: NUM_ROWS / 2,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let mut ivf_params = IvfBuildParams::new(NUM_PARTITIONS);
        ivf_params.sample_rate = 2;
        ivf_params.streaming_sample_rate = streaming_sample_rate;
        ivf_params.streaming_refine_passes = refine_passes;
        ivf_params.max_iters = 1;
        let params = match index {
            "rq" => VectorIndexParams::with_ivf_rq_params(
                MetricType::Cosine,
                ivf_params,
                RQBuildParams::new(1),
            ),
            "flat" => VectorIndexParams::with_ivf_flat_params(MetricType::Cosine, ivf_params),
            "pq" => VectorIndexParams::with_ivf_pq_params(
                MetricType::Cosine,
                ivf_params,
                PQBuildParams::new(4, 8),
            ),
            "sq" => VectorIndexParams::with_ivf_sq_params(
                MetricType::Cosine,
                ivf_params,
                SQBuildParams::default(),
            ),
            _ => unreachable!(),
        };
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        drop(dataset);

        let dataset = Dataset::open(test_dir.as_str()).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let index = dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let model = index.ivf_model();
        assert_eq!(model.num_partitions(), NUM_PARTITIONS);
        assert_eq!(model.centroids_array().unwrap().value_type(), value_type);

        let mut scanner = dataset.scan();
        // Probe and rerank the full fixture to isolate type correctness from
        // partition training and random one-bit quantization quality.
        scanner
            .nearest("vector", query.as_ref(), K)
            .unwrap()
            .distance_metric(MetricType::Cosine)
            .minimum_nprobes(NUM_PARTITIONS)
            .refine(NUM_ROWS.div_ceil(K) as u32);
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(plan.contains("ANNIvfPartition"), "{plan}");
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), K);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert_eq!(ids[0], EXPECTED_ROW as u64);
        assert!(
            result["_distance"]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|distance| distance.is_finite())
        );

        let ground_truth = dataset
            .scan()
            .nearest("vector", query.as_ref(), K)
            .unwrap()
            .distance_metric(MetricType::Cosine)
            .use_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let expected_ids = ground_truth["id"].as_primitive::<UInt64Type>().values();
        let recall = ids.iter().filter(|id| expected_ids.contains(id)).count() as f32 / K as f32;
        assert!(recall >= 0.5, "recall: {recall}, row ids: {ids:?}");
    }

    #[test]
    fn test_fixed_training_ranges_are_sorted_and_bounded() {
        let ranges = generate_fixed_training_ranges(10_000, 1_234, 1_024, 16);
        assert_eq!(ranges.num_rows(), 1_234);
        assert!(ranges.ranges.iter().all(|range| {
            range.start < range.end && range.end <= 10_000 && range_len(range) <= 1_234
        }));
        assert!(
            ranges
                .ranges
                .windows(2)
                .all(|pair| pair[0].end < pair[1].start)
        );

        let all_rows = generate_fixed_training_ranges(128, 256, 1_024, 16);
        assert_eq!(all_rows.ranges, vec![0..128]);
        assert_eq!(all_rows.num_rows(), 128);
    }

    #[test]
    fn test_fixed_training_ranges_chunk_splits_ranges() {
        let ranges = FixedIvfTrainingRanges::new(vec![10..20, 30..45]);
        assert_eq!(ranges.num_rows(), 25);
        assert_eq!(ranges.chunk(0, 5), vec![10..15]);
        assert_eq!(ranges.chunk(5, 12), vec![15..20, 30..37]);
        assert_eq!(ranges.chunk(20, 10), vec![40..45]);
        assert!(ranges.chunk(25, 10).is_empty());
    }

    #[test]
    fn test_split_ranges_by_row_count() {
        assert_eq!(
            split_ranges_by_row_count(&[10..25, 30..33], 8),
            vec![10..18, 18..25, 30..33]
        );
        assert_eq!(
            split_ranges_by_row_count(&[5..8], 0),
            vec![5..6, 6..7, 7..8]
        );
        assert!(split_ranges_by_row_count(&[], 8).is_empty());
    }

    #[test]
    fn test_streaming_ivf_progress_throttle() {
        assert!(should_report_streaming_ivf_progress(1, 64));
        assert!(!should_report_streaming_ivf_progress(63, 64));
        assert!(should_report_streaming_ivf_progress(64, 64));
        assert!(should_report_streaming_ivf_progress(128, 64));
        assert!(should_report_streaming_ivf_progress(2, 0));
    }

    #[test]
    fn test_streaming_coreset_default_rate_is_bounded_by_stream_rate() {
        assert_eq!(default_streaming_coreset_rate(256, 1), 1);
        assert_eq!(default_streaming_coreset_rate(256, 16), 16);
        assert_eq!(default_streaming_coreset_rate(256, 128), 64);
        assert_eq!(default_streaming_coreset_rate(8, 128), 8);
        assert_eq!(streaming_coreset_rate(256, 128, Some(16)), 16);
        assert_eq!(
            streaming_local_coreset_k(1024, 1024 * 128, 16, 2, true),
            1024 * 8
        );
        assert_eq!(
            streaming_local_coreset_k(1024, 1024 * 128, 16, 2, false),
            1024
        );
    }

    #[test]
    fn test_streaming_metric_policy_maps_cosine_and_sizes_local_coresets() {
        assert_eq!(
            streaming_kmeans_metric_policy(DistanceType::Cosine)
                .unwrap()
                .metric_type(),
            DistanceType::L2
        );
        assert_eq!(
            streaming_kmeans_metric_policy(DistanceType::Hamming)
                .unwrap()
                .metric_type(),
            DistanceType::Hamming
        );
        assert_eq!(
            L2_METRIC_POLICY.local_coreset_k(1024, 1024 * 128, 16, 2, true),
            1024 * 8
        );
        assert_eq!(
            DOT_METRIC_POLICY.local_coreset_k(1024, 1024 * 128, 16, 2, true),
            512
        );
        assert!(!L2_METRIC_POLICY.disable_local_hierarchical());
        assert!(DOT_METRIC_POLICY.disable_local_hierarchical());
    }

    fn assert_weighted_coreset_reduction<T: StreamingKMeansArrowType>() {
        let mut coreset = WeightedCoreset::<T>::new(1, 4);
        for value in [0.0, 100.0, 1.0, 101.0] {
            coreset.push(&[T::Native::from_f64(value).unwrap()], 1.0, 0.0);
        }

        coreset.reduce_to_budget(1, 2, &L2_METRIC_POLICY).unwrap();

        assert_eq!(coreset.len(), 2);
        assert!((coreset.values[0].as_() - 0.5).abs() < 1e-6);
        assert!((coreset.values[1].as_() - 100.5).abs() < 1e-6);

        let (data, weights, losses) = coreset.into_fsl_parts(1).unwrap();
        assert_eq!(data.value_type(), T::DATA_TYPE);
        assert_eq!(weights, vec![2.0, 2.0]);
        assert!((losses.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        let values = streaming_kmeans_values::<T>(&data).unwrap();
        let (subset, subset_weights, subset_losses) =
            weighted_subset::<T>(values, &weights, &losses, &[1, 0], 1).unwrap();
        assert_eq!(subset.value_type(), T::DATA_TYPE);
        assert_eq!(subset_weights, vec![2.0, 2.0]);
        assert_eq!(subset_losses, vec![losses[1], losses[0]]);
        assert_eq!(
            streaming_kmeans_values::<T>(&subset)
                .unwrap()
                .iter()
                .map(|value| value.as_())
                .collect::<Vec<_>>(),
            vec![100.5, 0.5]
        );
    }

    #[test]
    fn test_weighted_coreset_reduction_groups_nearby_centroids() {
        assert_weighted_coreset_reduction::<Float32Type>();
        assert_weighted_coreset_reduction::<Float64Type>();
    }

    #[test]
    fn test_dot_coreset_reduction_has_zero_merge_residual() {
        let mut coreset = WeightedCoreset::<Float32Type>::new(1, 2);
        coreset.push(&[2.0], 2.0, 0.0);
        coreset.push(&[4.0], 1.0, 0.0);

        coreset.reduce_to_budget(1, 1, &DOT_METRIC_POLICY).unwrap();

        assert_eq!(coreset.len(), 1);
        assert!((coreset.values[0] - 8.0 / 3.0).abs() < 1e-6);
        assert_eq!(coreset.weights, vec![3.0]);
        assert_eq!(coreset.losses, vec![0.0]);
    }

    // One cluster for each of 300 rows over 150 distinct vectors, and a single
    // row under Dot, whose local k is zero
    #[rstest]
    #[case::repeated_rows(300, 150, 300, &L2_METRIC_POLICY)]
    #[case::single_dot_row(1, 1, 0, &DOT_METRIC_POLICY)]
    fn test_local_coreset_keeps_rows_that_fit_its_budget(
        #[case] num_rows: usize,
        #[case] distinct_rows: usize,
        #[case] local_k: usize,
        #[case] metric_policy: &dyn StreamingFloatKMeansMetricPolicy<Float32Type>,
    ) {
        let values = (0..num_rows)
            .flat_map(|row| [(row % distinct_rows) as f32, 0.0])
            .collect::<Vec<_>>();
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(values.clone(), 2).unwrap();
        let mut coreset = WeightedCoreset::<Float32Type>::new(2, num_rows);

        append_local_coreset(
            &mut coreset,
            &data,
            metric_policy,
            local_k,
            1,
            Arc::new(|_, _| {}),
        )
        .unwrap();

        assert_eq!(coreset.values, values);
        assert_eq!(coreset.weights, vec![1.0; num_rows]);
        assert_eq!(coreset.losses, vec![0.0; num_rows]);
    }

    #[test]
    fn test_weighted_dot_summary_preserves_assignment_loss() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![2.0, 4.0], 1).unwrap();
        let weights = vec![2.0, 1.0];
        let losses = vec![0.0, 0.0];

        let result = assign_weighted_points::<Float32Type>(
            &data,
            &weights,
            &losses,
            &[3.0],
            &DOT_METRIC_POLICY,
        )
        .unwrap();

        let raw_loss = 2.0 * (1.0 - 2.0 * 3.0) + (1.0 - 4.0 * 3.0);
        assert!((result.loss - raw_loss).abs() < 1e-6);
        assert!((result.centroids[0] - 8.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_convergence_scale_is_sign_independent() {
        let metric_policy = &DOT_METRIC_POLICY;
        assert!(metric_policy.has_converged(-10.0, -10.0005));
        assert!(metric_policy.has_converged(10.0, 10.0005));
        assert!(!metric_policy.has_converged(-10.0, -10.01));
        assert!(!metric_policy.has_converged(10.0, 10.01));
        assert!(metric_policy.has_converged(0.0, 0.0));
        assert!(!metric_policy.has_converged(1.0, 0.0));
    }

    #[test]
    fn test_streaming_dot_training_dispatches_supported_types() {
        let f16_data = FixedSizeListArray::try_new_from_values(
            Float16Array::from_iter_values([1.0, 2.0, -1.0, -2.0].into_iter().map(f16::from_f32)),
            1,
        )
        .unwrap();
        let f32_data =
            streaming_kmeans_fsl_from_values::<Float32Type>(vec![1.0, 2.0, -1.0, -2.0], 1).unwrap();
        let f64_data = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![1.0, 2.0, -1.0, -2.0]),
            1,
        )
        .unwrap();
        let int8_data =
            FixedSizeListArray::try_new_from_values(Int8Array::from(vec![1, 2, -1, -2]), 1)
                .unwrap();

        for data in [&f16_data, &f32_data, &f64_data, &int8_data] {
            let model = train_ivf_kmeans_step_arrow_array_no_loss(
                None,
                data,
                KMeansStepOptions {
                    dimension: 1,
                    metric_type: DistanceType::Dot,
                    num_partitions: 2,
                    sample_rate: 2,
                    max_iters: 1,
                    disable_hierarchical: true,
                    on_progress: Arc::new(|_, _| {}),
                },
            )
            .unwrap();
            assert_eq!(model.centroids.len(), 2);
        }
    }

    #[test]
    fn test_dot_raw_refinement_uses_dot_assignments() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![2.0, -2.0], 1).unwrap();
        let centroids =
            streaming_kmeans_fsl_from_values::<Float32Type>(vec![1.0, -1.0], 1).unwrap();
        let mut cluster_sums = vec![0.0; 2];
        let mut cluster_weights = vec![0.0; 2];

        let loss = accumulate_refine_assignments::<Float32Type>(
            &data,
            &centroids,
            &DOT_METRIC_POLICY,
            &mut cluster_sums,
            &mut cluster_weights,
        )
        .unwrap();

        assert_eq!(cluster_sums, vec![2.0, -2.0]);
        assert_eq!(cluster_weights, vec![1.0, 1.0]);
        assert_eq!(loss, -2.0);
    }

    #[test]
    fn test_weighted_dot_rejects_nonfinite_distance() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![f32::MAX], 1).unwrap();
        let error = assign_weighted_points::<Float32Type>(
            &data,
            &[1.0],
            &[0.0],
            &[f32::MAX],
            &DOT_METRIC_POLICY,
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("nonfinite distance"));
        assert!(error.to_string().contains(&DistanceType::Dot.to_string()));
    }

    #[test]
    fn test_dot_split_priority_is_nonnegative_dispersion() {
        let data = vec![1.0, 3.0];
        let priority =
            <DotMetricPolicy as StreamingFloatKMeansMetricPolicy<Float32Type>>::split_priority(
                &DOT_METRIC_POLICY,
                &data,
                &[1.0, 1.0],
                &[0, 1],
                &[2.0],
                1,
                -100.0,
            );

        assert_eq!(priority, 2.0);
    }

    #[test]
    fn test_squared_l2_accumulates_in_f64() {
        let distance = squared_l2(&[f32::MAX, f32::MAX], &[0.0, 0.0]);

        assert!(distance.is_finite());
        assert!(distance > f64::from(f32::MAX));
    }

    #[test]
    fn test_weighted_assignment_ignores_zero_weight_rows() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![0.0, 10.0], 1).unwrap();
        let result = assign_weighted_points::<Float32Type>(
            &data,
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 10.0],
            &L2_METRIC_POLICY,
        )
        .unwrap();
        assert_eq!(result.cluster_weights, vec![1.0, 0.0]);
        assert_eq!(result.centroids, vec![0.0, 10.0]);
        assert_eq!(result.loss, 0.0);
    }

    #[test]
    fn test_weighted_hierarchical_rejects_empty_and_mismatched_inputs() {
        let progress = Arc::new(|_: u32, _: u32| {});
        let params = WeightedHierarchicalKMeansParams {
            dimension: 1,
            target_k: 2,
            metric_policy: &L2_METRIC_POLICY,
            max_iters: 1,
            on_progress: progress,
        };
        let empty = streaming_kmeans_fsl_from_values::<Float32Type>(Vec::new(), 1).unwrap();
        assert!(
            train_weighted_hierarchical_kmeans::<Float32Type>(&empty, &[], &[], &params).is_err()
        );
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![0.0, 1.0], 1).unwrap();
        let error =
            train_weighted_hierarchical_kmeans::<Float32Type>(&data, &[1.0], &[0.0, 0.0], &params)
                .unwrap_err();
        assert!(error.to_string().contains("input lengths do not match"));
    }

    #[rstest]
    #[case::zero_loss(17, 0.0)]
    #[case::lossy_coreset(32, 1_000.0)]
    #[case::exhausted_splits(33, 1_000.0)]
    fn test_weighted_hierarchical_splits_until_target_k(
        #[case] target_k: usize,
        #[case] base_loss: f64,
    ) {
        let values = (0..16)
            .flat_map(|value| [value as f32, value as f32 + 0.1])
            .collect::<Vec<_>>();
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(values, 1).unwrap();
        // A singleton summary can retain a large loss from the raw vectors it
        // represents, even though that summary cannot be split any further.
        let mut losses = vec![0.0; data.len()];
        losses[0] = base_loss;
        let progress = Arc::new(|_: u32, _: u32| {});
        let params = WeightedHierarchicalKMeansParams {
            dimension: 1,
            target_k,
            metric_policy: &L2_METRIC_POLICY,
            max_iters: 3,
            on_progress: progress,
        };
        let result = train_weighted_hierarchical_kmeans::<Float32Type>(
            &data,
            &vec![1.0; 32],
            &losses,
            &params,
        );
        if target_k > data.len() {
            let Err(error) = result else {
                panic!(
                    "a coreset of {} rows cannot serve {target_k} partitions",
                    data.len()
                );
            };
            assert!(
                error
                    .to_string()
                    .contains("smaller than the partition count"),
                "unexpected error: {error}"
            );
            return;
        }
        let result = result.unwrap();
        assert_eq!(result.len(), target_k);
        let values = result.values().as_primitive::<Float32Type>().values();
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        let distinct_centroids = 1 + sorted
            .windows(2)
            .filter(|pair| (pair[0] - pair[1]).abs() > 0.001)
            .count();
        assert_eq!(distinct_centroids, target_k.min(data.len()));
    }

    #[test]
    fn test_weighted_update_all_points_single_cluster_and_duplicates() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![3.0, 3.0, 3.0], 1).unwrap();
        let result = assign_weighted_points::<Float32Type>(
            &data,
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 10.0],
            &L2_METRIC_POLICY,
        )
        .unwrap();
        assert_eq!(result.cluster_weights, vec![6.0, 0.0]);
        assert_eq!(result.centroids, vec![3.0, 10.0]);
        assert_eq!(result.membership, vec![Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn test_weighted_update_tie_breaks_to_first_centroid() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![0.0], 1).unwrap();
        let result = assign_weighted_points::<Float32Type>(
            &data,
            &[1.0],
            &[0.0],
            &[-1.0, 1.0],
            &L2_METRIC_POLICY,
        )
        .unwrap();
        assert_eq!(result.membership, vec![Some(0)]);
        assert_eq!(result.cluster_weights, vec![1.0, 0.0]);
    }

    #[test]
    fn test_streaming_partition_boundary_and_sample_rate_helpers() {
        assert_eq!(streaming_local_coreset_k(256, 256, 16, 1, false), 256);
        assert_eq!(streaming_local_coreset_k(257, 257, 16, 1, false), 257);
        assert_eq!(streaming_coreset_rate(256, 1, None), 1);
        assert_eq!(streaming_coreset_rate(256, 256, None), 64);
        assert_eq!(streaming_coreset_rate(256, 256, Some(16)), 16);
        assert_eq!(streaming_coreset_rate(1, 256, None), 1);
    }

    #[tokio::test]
    async fn test_streaming_coreset_rejects_insufficient_sample() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/insufficient", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::rand_vec::<Float32Type>(2.into()))
            .into_reader_rows(RowCount::from(8), BatchCount::from(1));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 1;
        params.streaming_sample_rate = Some(1);
        params.max_iters = 1;
        let error = build_ivf_model(
            &dataset,
            "vector",
            2,
            MetricType::L2,
            &params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("cannot train 257 centroids"),
            "{error}"
        );
    }

    // Disjoint steps only need 257 vectors between them, which an all-null
    // column lacks; multivector steps draw independently, so each of them
    // must hold 257 on its own
    #[rstest]
    #[case::all_null(false, "sampled 0 rows")]
    #[case::multivector(true, "sampled 200 rows")]
    #[tokio::test]
    async fn test_streaming_coreset_rejects_too_few_vectors(
        #[case] multivector: bool,
        #[case] sampled: &str,
    ) {
        let test_dir = TempStrDir::default();
        let values = generate_random_array_with_seed::<Float32Type>(200 * 8, [3; 32]);
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let column: ArrayRef = if multivector {
            let vectors = FixedSizeListArray::try_new(item, 8, Arc::new(values), None).unwrap();
            Arc::new(arrow_array::ListArray::new(
                Arc::new(Field::new("item", vectors.data_type().clone(), true)),
                arrow_buffer::OffsetBuffer::from_lengths(repeat_n(2, 100)),
                Arc::new(vectors),
                None,
            ))
        } else {
            let nulls = Some(NullBuffer::new_null(200));
            Arc::new(FixedSizeListArray::try_new(item, 8, Arc::new(values), nulls).unwrap())
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            column.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &format!("{}/few", test_dir.as_str()),
            None,
        )
        .await
        .unwrap();
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 8;
        params.streaming_sample_rate = Some(2);
        params.max_iters = 1;

        let error = build_ivf_model(
            &dataset,
            "vector",
            8,
            MetricType::Dot,
            &params,
            None,
            noop_progress(),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("Not enough training vectors"), "{error}");
        assert!(error.contains(sampled), "{error}");
    }

    #[rstest]
    #[case::fixed_no_refine(false, 0)]
    #[case::fixed_refine(false, 1)]
    #[case::fragments_no_refine(true, 0)]
    #[case::fragments_refine(true, 1)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_fixed_and_fragment_refinement(
        #[case] use_fragments: bool,
        #[case] refine_passes: usize,
    ) {
        const DIMENSION: usize = 1;
        const NUM_PARTITIONS: usize = 257;
        let test_dir = TempStrDir::default();
        let uri = format!("{}/streaming", test_dir.as_str());
        let reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>((DIMENSION as u32).into()),
            )
            .into_reader_rows(RowCount::from(1024), BatchCount::from(2));
        let dataset = Dataset::write(reader, &uri, None).await.unwrap();
        let fragment_ids = use_fragments.then(|| {
            dataset
                .get_fragments()
                .iter()
                .map(|fragment| fragment.id() as u32)
                .collect::<Vec<_>>()
        });
        let mut params = IvfBuildParams::new(NUM_PARTITIONS);
        params.sample_rate = 4;
        params.streaming_sample_rate = Some(2);
        params.streaming_refine_passes = refine_passes;
        params.max_iters = 1;
        let model = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            build_ivf_model(
                &dataset,
                "vector",
                DIMENSION,
                MetricType::L2,
                &params,
                fragment_ids.as_deref(),
                lance_index::progress::noop_progress(),
            ),
        )
        .await
        .expect("streaming refinement timed out")
        .unwrap();
        assert_eq!(model.num_partitions(), NUM_PARTITIONS);
        assert_eq!(model.dimension(), DIMENSION);
        let centroids = model.centroids.unwrap();
        assert_eq!(centroids.len(), NUM_PARTITIONS);
        assert!(
            centroids
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[tokio::test]
    async fn test_streaming_coreset_dimension_one_and_empty_dataset_errors() {
        let test_dir = TempStrDir::default();
        let empty_uri = format!("{}/empty", test_dir.as_str());
        let empty_reader = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::rand_vec::<Float32Type>(1.into()))
            .into_reader_rows(RowCount::from(0), BatchCount::from(1));
        let empty = Dataset::write(empty_reader, &empty_uri, None)
            .await
            .unwrap();
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 1;
        params.streaming_sample_rate = Some(1);
        let error = build_ivf_model(
            &empty,
            "vector",
            1,
            MetricType::L2,
            &params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap_err();
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn test_dot_assignment_accepts_negative_distances() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![2.0, -2.0], 1).unwrap();
        let result = assign_weighted_points::<Float32Type>(
            &data,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[1.0, -1.0],
            &DOT_METRIC_POLICY,
        )
        .unwrap();
        assert_eq!(result.membership, vec![Some(0), Some(1)]);
        assert!(result.loss < 0.0);
        assert_eq!(result.centroids, vec![2.0, -2.0]);
    }

    #[test]
    fn test_dot_refinement_preserves_empty_centroid() {
        let data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![2.0, 2.0], 1).unwrap();
        let initial =
            streaming_kmeans_fsl_from_values::<Float32Type>(vec![10.0, -10.0], 1).unwrap();
        let result = refine_weighted_kmeans::<Float32Type>(
            &data,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &initial,
            &DOT_METRIC_POLICY,
            1,
            Arc::new(|_, _| {}),
        )
        .unwrap();
        assert_eq!(result.centroids, vec![2.0, -10.0]);
        assert_eq!(result.cluster_weights, vec![2.0, 0.0]);
    }

    #[test]
    fn test_weighted_kmeanspp_initialization_selects_distant_centroids() {
        let values = vec![0.0, 0.1, 100.0, 101.0];
        let weights = vec![1.0; 4];
        let centroids = initialize_weighted_centroids::<Float32Type>(&values, 1, 2, 4, &weights);

        assert_eq!(centroids.len(), 2);
        assert!(
            (centroids[0] - centroids[1]).abs() > 10.0,
            "weighted kmeans++ should seed distant coreset regions, got {:?}",
            centroids
        );
    }

    #[test]
    fn test_prepare_streaming_coreset_sample_selects_working_type() {
        let f16_data = FixedSizeListArray::try_new_from_values(
            Float16Array::from_iter_values([1.25, -2.5].into_iter().map(f16::from_f32)),
            1,
        )
        .unwrap();
        let prepared_f16 = prepare_streaming_coreset_sample::<Float32Type>(f16_data).unwrap();
        assert_eq!(prepared_f16.value_type(), DataType::Float32);
        assert_eq!(
            prepared_f16.values().as_primitive::<Float32Type>().values(),
            &[1.25, -2.5]
        );

        let int8_data =
            FixedSizeListArray::try_new_from_values(Int8Array::from(vec![1, -2]), 1).unwrap();
        let prepared_int8 = prepare_streaming_coreset_sample::<Float32Type>(int8_data).unwrap();
        assert_eq!(prepared_int8.value_type(), DataType::Float32);
        assert_eq!(
            prepared_int8
                .values()
                .as_primitive::<Float32Type>()
                .values(),
            &[1.0, -2.0]
        );

        let f64_data =
            streaming_kmeans_fsl_from_values::<Float64Type>(vec![1.0 + f64::EPSILON, 2.0], 1)
                .unwrap();
        let prepared_f64 = prepare_streaming_coreset_sample::<Float64Type>(f64_data).unwrap();
        assert_eq!(prepared_f64.value_type(), DataType::Float64);
        assert_eq!(
            prepared_f64.values().as_primitive::<Float64Type>().values()[0],
            1.0 + f64::EPSILON
        );

        let f32_data = streaming_kmeans_fsl_from_values::<Float32Type>(vec![1.0], 1).unwrap();
        let prepared_f32 =
            prepare_streaming_coreset_sample::<Float32Type>(f32_data.clone()).unwrap();
        assert_eq!(prepared_f32.value_type(), DataType::Float32);
        assert_eq!(
            prepared_f32.values().as_primitive::<Float32Type>().values(),
            &[1.0]
        );

        let error = prepare_streaming_coreset_sample::<Float64Type>(f32_data).unwrap_err();
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("expected Float64 input values, got Float32"),
            "{error}"
        );
    }

    fn assert_weighted_assignment_preserves_empty_centroid<T: StreamingKMeansArrowType>() {
        let data = streaming_kmeans_fsl_from_values::<T>(
            vec![
                T::Native::from_f64(0.0).unwrap(),
                T::Native::from_f64(10.0).unwrap(),
            ],
            1,
        )
        .unwrap();
        let centroids = vec![
            T::Native::from_f64(0.0).unwrap(),
            T::Native::from_f64(10.0).unwrap(),
            T::Native::from_f64(20.0).unwrap(),
        ];
        let result = assign_weighted_points::<T>(
            &data,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &centroids,
            &L2_METRIC_POLICY,
        )
        .unwrap();

        assert_eq!(result.cluster_weights, vec![1.0, 1.0, 0.0]);
        assert_eq!(result.centroids[2].as_(), 20.0);
    }

    #[test]
    fn test_weighted_assignment_preserves_empty_centroid_for_both_types() {
        assert_weighted_assignment_preserves_empty_centroid::<Float32Type>();
        assert_weighted_assignment_preserves_empty_centroid::<Float64Type>();
    }

    #[test]
    fn test_weighted_f64_centroids_retain_precision() {
        const BASE: f64 = 16_777_216.0;
        const EXPECTED_MEAN: f64 = 16_777_217.0;

        let data =
            streaming_kmeans_fsl_from_values::<Float64Type>(vec![BASE, BASE + 2.0], 1).unwrap();
        let assigned = assign_weighted_points::<Float64Type>(
            &data,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[BASE],
            &L2_METRIC_POLICY,
        )
        .unwrap();
        assert_eq!(assigned.centroids, vec![EXPECTED_MEAN]);
        assert_ne!(assigned.centroids[0] as f32 as f64, EXPECTED_MEAN);

        let mut coreset = WeightedCoreset::<Float64Type>::new(1, 2);
        coreset.push(&[BASE], 1.0, 0.0);
        coreset.push(&[BASE + 2.0], 1.0, 0.0);
        coreset.reduce_to_budget(1, 1, &L2_METRIC_POLICY).unwrap();
        assert_eq!(coreset.values, vec![EXPECTED_MEAN]);
        assert_eq!(coreset.weights, vec![2.0]);
        assert_eq!(coreset.losses, vec![2.0]);
    }

    #[test]
    fn test_weighted_f64_coreset_preserves_extreme_magnitudes() {
        let subnormal = f64::MIN_POSITIVE;
        let mut tiny = WeightedCoreset::<Float64Type>::new(1, 2);
        tiny.push(&[subnormal], 1.0, 0.0);
        tiny.push(&[subnormal * 2.0], 1.0, 0.0);
        tiny.reduce_to_budget(1, 1, &L2_METRIC_POLICY).unwrap();
        assert_eq!(tiny.values, vec![subnormal * 1.5]);
        assert_ne!(tiny.values[0], tiny.values[0] as f32 as f64);

        let large = 1.0e100_f64;
        let delta = 2.0e84_f64;
        let mut huge = WeightedCoreset::<Float64Type>::new(1, 2);
        huge.push(&[large], 1.0, 0.0);
        huge.push(&[large + delta], 1.0, 0.0);
        huge.reduce_to_budget(1, 1, &L2_METRIC_POLICY).unwrap();
        assert!(huge.values[0].is_finite());
        assert!(huge.losses[0].is_finite());
        assert_ne!(huge.values[0], huge.values[0] as f32 as f64);
    }

    #[tokio::test]
    async fn test_fixed_cosine_sampler_preserves_f64_normalization() {
        let test_dir = TempStrDir::default();
        let uri = format!("{}/cosine-f64", test_dir.as_str());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
            false,
        )]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![3.0, 4.0, 5.0, 12.0]),
            2,
        )
        .unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            None,
        )
        .await
        .unwrap();
        let sampler = FixedIvfTrainingSampler::try_new(&dataset, "vector")
            .unwrap()
            .unwrap();
        let (sample, metric) = sampler
            .sample_ranges(&[0..2], MetricType::Cosine)
            .await
            .unwrap();
        assert_eq!(metric, MetricType::L2);
        assert_eq!(sample.value_type(), DataType::Float64);
        let values = sample.values().as_primitive::<Float64Type>().values();
        for vector in values.as_chunks::<2>().0 {
            assert!((vector.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1e-14);
        }
        assert!((values[0] - 0.6).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_f64_refinement_sources_retain_exact_mean() {
        const BASE: f64 = 16_777_216.0;
        const EXPECTED_MEAN: f64 = 16_777_216.5;
        let test_dir = TempStrDir::default();
        let uri = format!("{}/refine-f64", test_dir.as_str());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 1),
            false,
        )]));
        let vectors =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![BASE, BASE + 1.0]), 1)
                .unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            None,
        )
        .await
        .unwrap();
        let initial = streaming_kmeans_fsl_from_values::<Float64Type>(vec![BASE], 1).unwrap();
        let sampler = FixedIvfTrainingSampler::try_new(&dataset, "vector")
            .unwrap()
            .unwrap();
        let ranges = FixedIvfTrainingRanges::new(vec![0..2]);
        let fixed = refine_streaming_kmeans_with_sampler::<Float64Type>(
            &sampler,
            MetricType::L2,
            2,
            &ranges,
            &initial,
            1,
            Arc::new(|_, _| {}),
        )
        .await
        .unwrap();
        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        let resampled = refine_streaming_kmeans_with_resampling::<Float64Type>(
            &dataset,
            "vector",
            MetricType::L2,
            2,
            2,
            1,
            &initial,
            Some(&fragment_ids),
            1,
            Arc::new(|_, _| {}),
        )
        .await
        .unwrap();
        for refined in [fixed, resampled] {
            assert_eq!(refined.value_type(), DataType::Float64);
            let value = refined.values().as_primitive::<Float64Type>().value(0);
            assert_eq!(value, EXPECTED_MEAN);
            assert_ne!(value, value as f32 as f64);
        }
    }

    #[rstest]
    #[case::f16("f16", DataType::Float16)]
    #[case::f32("f32", DataType::Float32)]
    #[case::f64("f64", DataType::Float64)]
    #[case::int8("int8", DataType::Float32)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_streaming_coreset_output_type_contract(
        #[case] input_type: &str,
        #[case] expected_output: DataType,
    ) {
        const DIMENSION: usize = 2;
        const ROWS: usize = 600;
        let value_count = ROWS * DIMENSION;
        let (value_type, values): (DataType, ArrayRef) = match input_type {
            "f16" => (
                DataType::Float16,
                Arc::new(Float16Array::from_iter_values(
                    (0..value_count).map(|value| f16::from_f32(value as f32 / 100.0)),
                )),
            ),
            "f32" => (
                DataType::Float32,
                Arc::new(Float32Array::from_iter_values(
                    (0..value_count).map(|value| value as f32 / 100.0),
                )),
            ),
            "f64" => (
                DataType::Float64,
                Arc::new(Float64Array::from_iter_values(
                    (0..value_count).map(|value| value as f64 / 100.0 + f64::EPSILON),
                )),
            ),
            "int8" => (
                DataType::Int8,
                Arc::new(Int8Array::from_iter_values((0..ROWS).flat_map(|row| {
                    [(row & 0xff) as u8 as i8, (row >> 8) as u8 as i8]
                }))),
            ),
            _ => unreachable!(),
        };
        let test_dir = TempStrDir::default();
        let uri = format!("{}/type-{input_type}", test_dir.as_str());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", value_type, true)),
                DIMENSION as i32,
            ),
            false,
        )]));
        let vectors = FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            None,
        )
        .await
        .unwrap();
        let mut params = IvfBuildParams::new(257);
        params.sample_rate = 2;
        params.streaming_sample_rate = Some(1);
        params.max_iters = 1;
        let model = build_ivf_model(
            &dataset,
            "vector",
            DIMENSION,
            MetricType::L2,
            &params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(model.centroids.unwrap().value_type(), expected_output);
    }

    #[tokio::test]
    async fn test_create_ivf_pq_f16() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams::new(4, 8),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(
                "vector",
                &Float32Array::from_iter_values(repeat_n(0.5, DIM)),
                5,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let batch = &results[0];
        assert_eq!(
            batch.schema(),
            Arc::new(Schema::new(vec![
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        DIM as i32,
                    ),
                    true,
                ),
                Field::new("_distance", DataType::Float32, true)
            ]))
        );
    }

    #[tokio::test]
    async fn test_create_ivf_pq_f16_with_codebook() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let codebook = Arc::new(generate_random_array_with_seed::<Float16Type>(
            256 * DIM,
            [22; 32],
        ));
        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams::with_codebook(4, 8, codebook),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(
                "vector",
                &Float32Array::from_iter_values(repeat_n(0.5, DIM)),
                5,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let batch = &results[0];
        assert_eq!(
            batch.schema(),
            Arc::new(Schema::new(vec![
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        DIM as i32,
                    ),
                    true,
                ),
                Field::new("_distance", DataType::Float32, true)
            ]))
        );
    }

    /// A codebook whose sub-vector width disagrees with the column used to be
    /// accepted: the PQ transform derives the width from the column, so an
    /// oversized codebook had its sub-vector boundaries read at the wrong
    /// offsets and produced a searchable but wrong index, with no error.
    ///
    /// The two index file versions reach the codebook through different
    /// writers, so both are covered.
    #[rstest]
    #[case::v3(IndexFileVersion::V3)]
    #[case::legacy(IndexFileVersion::Legacy)]
    #[tokio::test]
    async fn test_create_ivf_pq_rejects_mismatched_codebook(#[case] version: IndexFileVersion) {
        const DIM: usize = 32;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));
        let arr = generate_random_array_with_seed::<Float32Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        // Twice the required 256 * DIM. Divisible by the column dimension, so
        // nothing downstream complains on its own.
        let codebook = Arc::new(generate_random_array_with_seed::<Float32Type>(
            256 * DIM * 2,
            [22; 32],
        ));
        let mut params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            fast_ivf_params(2),
            PQBuildParams::with_codebook(4, 8, codebook),
        );
        params.version = version;
        let error = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .expect_err("a codebook that does not match the column must be rejected");
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains(&(256 * DIM * 2).to_string())
                && message.contains(&(256 * DIM).to_string()),
            "the error must give the supplied and the required size: {message}"
        );
    }

    /// The codebook length matches, but its values are f32 while the column is
    /// f16. Every create path used to accept it and panic on the first batch
    /// when quantization read the codebook as f16.
    #[rstest]
    #[case::v3(IndexFileVersion::V3)]
    #[case::legacy(IndexFileVersion::Legacy)]
    #[tokio::test]
    async fn test_create_ivf_pq_rejects_codebook_of_another_value_type(
        #[case] version: IndexFileVersion,
    ) {
        const DIM: usize = 32;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));
        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let codebook = Arc::new(generate_random_array_with_seed::<Float32Type>(
            256 * DIM,
            [22; 32],
        ));
        let mut params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            fast_ivf_params(2),
            PQBuildParams::with_codebook(4, 8, codebook),
        );
        params.version = version;
        let error = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .expect_err("a codebook of another value type must be rejected");
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        assert!(
            error.to_string().contains(
                "PQ codebook value type Float32 does not match the vector value type Float16"
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_create_ivf_flat_f16() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let query = Float16Array::from_iter_values(repeat_n(f16::from_f32(0.5), DIM));
        let results = dataset
            .scan()
            .nearest("vector", &query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let schema = results[0].schema();
        let field = schema.field(0);
        let DataType::FixedSizeList(item, _) = field.data_type() else {
            panic!("vector column should remain fixed size list");
        };
        assert_eq!(item.data_type(), &DataType::Float16);
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_flat_f16() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_hnsw(
            MetricType::L2,
            IvfBuildParams::new(2),
            HnswBuildParams::default(),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let query = Float16Array::from_iter_values(repeat_n(f16::from_f32(0.5), DIM));
        let results = dataset
            .scan()
            .nearest("vector", &query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let schema = results[0].schema();
        let field = schema.field(0);
        let DataType::FixedSizeList(item, _) = field.data_type() else {
            panic!("vector column should remain fixed size list");
        };
        assert_eq!(item.data_type(), &DataType::Float16);
    }

    #[tokio::test]
    async fn test_create_ivf_pq_with_invalid_num_sub_vectors() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float32Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(256),
            PQBuildParams::new(6, 8),
        );
        let res = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await;
        match &res {
            Err(Error::InvalidInput { source, .. }) => {
                assert!(
                    source
                        .to_string()
                        .contains("num_sub_vectors must divide vector dimension"),
                    "{:?}",
                    res
                );
            }
            _ => panic!("Expected InvalidInput error: {:?}", res),
        }
    }

    fn ground_truth(
        fsl: &FixedSizeListArray,
        query: &[f32],
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u32)> {
        let dim = fsl.value_length() as usize;
        let mut dists = fsl
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .chunks(dim)
            .enumerate()
            .map(|(i, vec)| {
                let dist = distance_type.try_func().unwrap()(query, vec);
                (dist, i as u32)
            })
            .collect_vec();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_pq() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let nlist = 2;
        let (mut dataset, vector_array) =
            generate_test_dataset_with_rows(test_uri, 0.0..1.0, 512).await;

        // The live build samples IVF training rows and initializes both k-means
        // models from OS entropy, so this fixture's recall is a random variable
        // whose tail crosses the assertion. Train both models here from fixed
        // rows with the production trainers, then build the index with them.
        // Partitioning, PQ encoding, HNSW construction and search are unchanged.
        let flat = vector_array.values().as_primitive::<Float32Type>();
        let mut train_ivf_params = fast_ivf_params(nlist);
        let training_rows = nlist * train_ivf_params.sample_rate;
        let training_sample =
            FixedSizeListArray::try_new_from_values(flat.slice(0, training_rows * DIM), DIM as i32)
                .unwrap();
        train_ivf_params.centroids = Some(Arc::new(
            FixedSizeListArray::try_new_from_values(flat.slice(0, nlist * DIM), DIM as i32)
                .unwrap(),
        ));
        train_ivf_params.retrain = true;
        let ivf_model = train_ivf_model_from_sample(
            training_sample,
            MetricType::L2,
            &train_ivf_params,
            noop_progress(),
        )
        .await
        .unwrap();
        let ivf_centroids = Arc::new(ivf_model.centroids.unwrap());

        let transformer = lance_index::vector::ivf::new_ivf_transformer(
            ivf_centroids.as_ref().clone(),
            DistanceType::L2,
            vec![],
        );
        let residuals = transformer.compute_residual(&vector_array).unwrap();
        let initial_codebook = FixedSizeListArray::try_new_from_values(
            residuals
                .values()
                .as_primitive::<Float32Type>()
                .slice(0, 256 * DIM),
            DIM as i32,
        )
        .unwrap();
        let mut train_pq_params = fast_pq_params(4, 8);
        train_pq_params.codebook = Some(Arc::new(initial_codebook));
        let trained_pq = train_pq_params.build(&residuals, DistanceType::L2).unwrap();

        let mut ivf_params = fast_ivf_params(nlist);
        ivf_params.centroids = Some(ivf_centroids);
        let mut pq_params = fast_pq_params(4, 8);
        pq_params.codebook = Some(Arc::new(
            trained_pq
                .codebook
                .values()
                .as_primitive::<Float32Type>()
                .clone(),
        ));
        let hnsw_params = fast_hnsw_params();
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            pq_params,
        );

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let query = vector_array.value(0);
        let query = query.as_primitive::<Float32Type>();
        let k = 20;
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", query, k)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(64)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(k, results.num_rows());

        let row_ids = results
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap() as u32)
            .collect::<Vec<_>>();
        let dists = results
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();

        let results = dists.into_iter().zip(row_ids.into_iter()).collect_vec();
        let gt = ground_truth(&vector_array, query.values(), k, DistanceType::L2);

        let results_set = results.iter().map(|r| r.1).collect::<HashSet<_>>();
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();
        assert_eq!(results_set.len(), k, "search returned duplicate row ids");
        assert!(
            results.iter().all(|(distance, _)| distance.is_finite()),
            "search returned a non-finite distance: {results:?}"
        );
        assert!(
            results.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "search distances are not sorted: {results:?}"
        );

        let recall = results_set.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= 0.5,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_with_empty_partition() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // the generate_test_dataset function generates a dataset with 1000 vectors,
        // so 1001 partitions will have at least one empty partition
        let nlist = 1001;
        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(nlist * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params =
            IvfBuildParams::try_with_centroids(nlist, Arc::new(ivf_centroids)).unwrap();

        let distance_type = DistanceType::L2;
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let query = vector_array.value(0);
        let query = query.as_primitive::<Float32Type>();
        let k = 100;
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", query, k)
            .unwrap()
            .minimum_nprobes(nlist)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(k, results[0].num_rows());

        let row_ids = results[0]
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap() as u32)
            .collect::<Vec<_>>();
        let dists = results[0]
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();

        let results = dists.into_iter().zip(row_ids.into_iter()).collect_vec();
        let gt = ground_truth(&vector_array, query.values(), k, distance_type);

        let results_set = results.iter().map(|r| r.1).collect::<HashSet<_>>();
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = results_set.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= 0.9,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    #[tokio::test]
    async fn test_check_cosine_normalization() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        const DIM: usize = 32;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_range::<Float32Type>(1000 * DIM, 1000.0..1001.0);
        let fsl = FixedSizeListArray::try_new_from_values(arr.clone(), DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 4, MetricType::Cosine, 50);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let idx = dataset
            .open_generic_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let ivf_idx = idx.as_any().downcast_ref::<v2::IvfPq>().unwrap();

        assert!(
            ivf_idx
                .ivf_model()
                .centroids
                .as_ref()
                .unwrap()
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        );

        // PQ code is on residual space
        let pq_store = ivf_idx.load_partition_storage(0, None).await.unwrap();
        pq_store
            .codebook()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| assert!((-1.0..=1.0).contains(v), "Got {}", v));

        let dataset = Dataset::open(test_uri).await.unwrap();

        let mut correct_times = 0;
        for query_id in 0..10 {
            let query = &arr.slice(query_id * DIM, DIM);
            let results = dataset
                .scan()
                .with_row_id()
                .nearest("vector", query, 1)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(results.num_rows(), 1);
            let row_id = results
                .column_by_name("_rowid")
                .unwrap()
                .as_primitive::<UInt64Type>()
                .value(0);
            if row_id == (query_id as u64) {
                correct_times += 1;
            }
        }

        assert!(correct_times >= 9, "correct: {}", correct_times);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_build_ivf_model_progress_callback() {
        use lance_index::progress::IndexBuildProgress;
        use tokio::sync::Mutex;

        #[derive(Debug)]
        struct RecordingProgress {
            calls: Arc<Mutex<Vec<(String, u64)>>>,
        }

        #[async_trait::async_trait]
        impl IndexBuildProgress for RecordingProgress {
            async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
                Ok(())
            }
            async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()> {
                self.calls.lock().await.push((stage.to_string(), completed));
                Ok(())
            }
            async fn stage_complete(&self, _: &str) -> Result<()> {
                Ok(())
            }
        }

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (dataset, _) = generate_test_dataset(test_uri, 1000.0..1100.0).await;

        let ivf_params = IvfBuildParams::new(2);
        let calls: Arc<Mutex<Vec<(String, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let progress: Arc<dyn IndexBuildProgress> = Arc::new(RecordingProgress {
            calls: calls.clone(),
        });

        let ivf_model = build_ivf_model(
            &dataset,
            "vector",
            DIM,
            MetricType::L2,
            &ivf_params,
            None,
            progress,
        )
        .await
        .unwrap();
        assert_eq!(2, ivf_model.num_partitions());

        // Let spawned progress tasks complete.
        tokio::task::yield_now().await;

        let recorded = calls.lock().await;
        assert!(
            !recorded.is_empty(),
            "Expected progress callbacks to be called"
        );
        // All calls should be for train_ivf stage
        for (stage, _) in recorded.iter() {
            assert_eq!(stage, "train_ivf");
        }
        // Completed values should be monotonically increasing
        for window in recorded.windows(2) {
            assert!(
                window[1].1 >= window[0].1,
                "Expected monotonically increasing progress: {} >= {}",
                window[1].1,
                window[0].1,
            );
        }
    }

    #[tokio::test]
    async fn test_prewarm_ivf_legacy() {
        use lance_io::assert_io_eq;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let dim = DIM as i32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        )]));
        let vectors = generate_random_array(512 * DIM);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, dim).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let nlist = 4;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(nlist),
            PQBuildParams::default(),
        )
        .version(IndexFileVersion::Legacy)
        .clone();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("my_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reset IO stats after index creation
        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        // Prewarm should perform IO to load all partitions into cache
        dataset.prewarm_index("my_idx").await.unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert!(
            stats.read_iops > 0,
            "prewarm should have read from disk, but read_iops was 0"
        );

        // Can query index without IO
        let q = Float32Array::from_iter_values(repeat_n(0.0, DIM));
        dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "query should not perform IO after prewarm"
        );

        // Second prewarm should not need IO (already cached)
        dataset.prewarm_index("my_idx").await.unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
    }

    /// Count this process's descriptors for index files under `prefix`.
    ///
    /// Deleted files still match: their `/proc` link keeps the old path with a
    /// ` (deleted)` suffix.
    #[cfg(target_os = "linux")]
    fn count_open_index_fds(prefix: &str) -> usize {
        std::fs::read_dir("/proc/self/fd")
            .expect("read /proc/self/fd")
            .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
            .filter(|target| {
                let target = target.to_string_lossy();
                target.starts_with(prefix) && target.contains("_indices")
            })
            .count()
    }

    /// A cached legacy index must not keep a descriptor open for the file it
    /// was loaded from. Creating indexes in a loop mints a new UUID each time;
    /// before the fix the session index cache pinned one descriptor per UUID,
    /// which is what let descriptor counts grow with every optimize/rewrite.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_legacy_index_cache_does_not_pin_file_descriptors() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let dim = DIM as i32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        )]));
        let vectors = generate_random_array(512 * DIM);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, dim).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        )
        .version(IndexFileVersion::Legacy)
        .clone();
        let query = Float32Array::from_iter_values(repeat_n(0.0, DIM));

        for round in 0..4 {
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some("my_idx".to_owned()),
                    &params,
                    true,
                )
                .await
                .unwrap();

            // Querying opens the index, which caches it for the session.
            dataset
                .scan()
                .nearest("vector", &query, 10)
                .unwrap()
                .project(&["_rowid"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let open_fds = count_open_index_fds(test_dir.as_str());
            assert_eq!(
                open_fds, 0,
                "round {round}: cached legacy index pinned {open_fds} descriptors"
            );
        }
    }

    #[tokio::test]
    async fn test_prewarm_ivf_legacy_multiple_deltas() {
        use lance_io::assert_io_eq;

        let test_dir = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        // Trigger migration to repair legacy corrupt fragment bitmaps.
        let mut dataset = Dataset::open(test_uri).await.unwrap();
        dataset.index_statistics("vector_idx").await.unwrap();
        dataset.checkout_latest().await.unwrap();

        // Reopen dataset to avoid carrying index state in-memory from migration.
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 2, "expected two index deltas for vector_idx");
        let unique_uuids: HashSet<_> = indices.iter().map(|meta| meta.uuid).collect();
        assert_eq!(unique_uuids.len(), 2, "expected two unique index UUIDs");

        let sample_batch = dataset
            .scan()
            .limit(Some(1), None)
            .unwrap()
            .project(&["vector"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = sample_batch["vector"]
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .value(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .clone();

        // Reset IO stats after migration and sampling.
        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        // Prewarm should perform IO to load all index deltas into cache.
        dataset.prewarm_index("vector_idx").await.unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert!(
            stats.read_iops > 0,
            "prewarm should have read from disk, but read_iops was 0"
        );

        // Query should not perform index IO after prewarm of all deltas.
        dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "query should not perform IO after prewarm"
        );

        // Second prewarm should not need IO (already cached).
        dataset.prewarm_index("vector_idx").await.unwrap();
        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
    }
}
