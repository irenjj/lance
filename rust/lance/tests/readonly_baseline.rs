// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{num::NonZeroU64, path::Path, sync::Arc};

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StructArray, UInt64Array,
    cast::AsArray, record_batch, types::Float32Type,
};
use arrow_schema::{Field, Schema as ArrowSchema};
use datafusion_catalog::TableProvider;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_expr::{col, lit};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{collect, displayable};
use futures::{StreamExt, TryStreamExt};
use lance::blob::{BlobDescriptorArrayBuilder, BlobRange};
use lance::datafusion::LanceTableProvider;
use lance::dataset::builder::DatasetBuilder;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::{Dataset, Error, Result};
use lance_core::cache::LanceCache;
use lance_core::datatypes::BlobKind;
use lance_core::datatypes::{BlobHandling, Schema};
use lance_core::utils::{address::RowAddress, deletion::DeletionVector};
use lance_core::{ROW_ADDR, ROW_ID};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_file::versions::create_writer;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bitmap::BitmapIndexPlugin;
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::inverted::builder::InvertedIndexBuilder;
use lance_index::scalar::inverted::query::{BooleanQuery, MatchQuery, PhraseQuery};
use lance_index::scalar::registry::{BasicTrainer, VALUE_COLUMN_NAME};
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::scalar::{index_files_to_table, lance_format::LanceIndexStore};
use lance_index::vector::DISTANCE_TYPE_KEY;
use lance_index::vector::flat::{
    index::{FlatIndex, FlatQuantizer},
    storage::FLAT_COLUMN,
};
use lance_index::vector::ivf::storage::{IVF_METADATA_KEY, IvfModel};
use lance_index::vector::quantizer::Quantization as _;
use lance_index::vector::shared::{SupportedIvfIndexType, write_unified_ivf_and_index_metadata};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, VECTOR_INDEX_VERSION, pb};
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use lance_table::feature_flags::{FLAG_FRAGMENT_REUSE_INDEX, apply_feature_flags};
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::pb::fragment_reuse_index_details::{
    OrderedCompaction, Transition, transition,
};
use lance_table::format::{DataFile, IndexFile, IndexMetadata, Manifest};
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table::io::deletion::write_deletion_file;
use lance_table::system_index::frag_reuse::{FRAG_REUSE_INDEX_NAME, FragReuseGroup};
use lance_table::transaction::Operation;
use lance_table_runtime::Table;
use lance_table_runtime::blob::locate::StoredBlob;
use lance_table_runtime::frag_reuse::{build_new_frag_reuse_index, frag_reuse_entry_appending};
use lance_table_runtime::write::{InsertBuilder, WriteMode, WriteParams};
use lance_testing::datafusion::TestSession;
use prost::Message;
use rand::{Rng, SeedableRng, rngs::StdRng};
use roaring::RoaringBitmap;
use roaring::RoaringTreemap;
use tempfile::TempDir;
use uuid::Uuid;

const FORMAT: ConcreteFileVersion = ConcreteFileVersion::V2_2;
const IDS: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const TEXTS: [&str; 8] = [
    "mars moon moon moon",
    "venus",
    "mars red red",
    "cold moon",
    "mars moon",
    "venus",
    "mars red moon",
    "earth",
];

fn rows() -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(57);
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        IDS.map(|id| Some([Some(id as f32 * 10.0), Some(rng.random_range(0.0..1.0))])),
        2,
    );
    let scalars =
        record_batch!(("id", Int32, IDS.to_vec()), ("text", Utf8, TEXTS.to_vec())).unwrap();
    let mut fields = scalars.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "vector",
        vectors.data_type().clone(),
        true,
    )));
    let mut columns = scalars.columns().to_vec();
    columns.push(Arc::new(vectors));
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
}

fn batch_stream(batch: RecordBatch) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        batch.schema(),
        futures::stream::iter([Ok(batch)]),
    ))
}

fn row_addresses(table: &Table) -> UInt64Array {
    table
        .manifest()
        .fragments
        .iter()
        .flat_map(|fragment| {
            (0..fragment.physical_rows.unwrap() as u32)
                .map(|offset| u64::from(RowAddress::new_from_parts(fragment.id as u32, offset)))
        })
        .collect::<Vec<_>>()
        .into()
}

fn batch_ids(batch: &RecordBatch) -> &[i32] {
    batch["id"]
        .as_primitive::<arrow_array::types::Int32Type>()
        .values()
}

async fn create_table(uri: &str, batch: RecordBatch, max_rows: usize) -> Result<Table> {
    InsertBuilder::new(uri)
        .with_params(&WriteParams {
            max_rows_per_file: max_rows,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute(vec![batch])
        .await
}

fn index_metadata(
    table: &Table,
    uuid: Uuid,
    column: &str,
    index_version: u32,
    details: prost_types::Any,
    coverage: impl IntoIterator<Item = u32>,
    files: Vec<IndexFile>,
) -> IndexMetadata {
    IndexMetadata {
        uuid,
        name: format!("{column}_idx"),
        fields: vec![table.schema().field_id(column).unwrap()],
        covering_fields: vec![],
        dataset_version: table.manifest().version,
        fragment_bitmap: Some(coverage.into_iter().collect()),
        index_details: Some(Arc::new(details)),
        index_version: index_version as i32,
        created_at: None,
        base_id: None,
        files: Some(files),
        clustering_id: None,
        metadata: Default::default(),
    }
}

async fn write_ivf_flat(
    table: &Table,
    uuid: Uuid,
    vectors: &FixedSizeListArray,
    row_ids: &UInt64Array,
) -> Result<Vec<IndexFile>> {
    let (store, _) = ObjectStore::from_uri(table.uri()).await?;
    let directory = table.indices_dir().join(uuid.to_string());
    let quantizer = FlatQuantizer::new(2, DistanceType::L2);
    let mut ivf = IvfModel::new(vectors.slice(0, 1), None);
    ivf.add_partition(vectors.len() as u32);
    let values = RecordBatch::try_from_iter([
        (ROW_ID, Arc::new(row_ids.clone()) as _),
        (FLAT_COLUMN, Arc::new(vectors.clone()) as _),
    ])?;

    let mut auxiliary = create_writer(
        FORMAT,
        store
            .create(&directory.clone().join(INDEX_AUXILIARY_FILE_NAME))
            .await?,
        Schema::try_from(values.schema().as_ref())?,
        Default::default(),
    )?;
    auxiliary.write_batch(&values).await?;
    let ivf_buffer = auxiliary
        .add_global_buffer(pb::Ivf::try_from(&ivf)?.encode_to_vec().into())
        .await?;
    auxiliary.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer.to_string());
    auxiliary.add_schema_metadata(DISTANCE_TYPE_KEY, DistanceType::L2.to_string());
    let flat_metadata = serde_json::to_string(&quantizer.metadata(None))?;
    auxiliary.add_schema_metadata(
        STORAGE_METADATA_KEY,
        serde_json::to_string(&[flat_metadata])?,
    );

    let mut root = create_writer(
        FORMAT,
        store.create(&directory.join(INDEX_FILE_NAME)).await?,
        Schema::try_from(FlatIndex::schema().as_ref())?,
        Default::default(),
    )?;
    let mut root_ivf = IvfModel::new(vectors.slice(0, 1), None);
    root_ivf.add_partition(0);
    write_unified_ivf_and_index_metadata(
        &mut root,
        &root_ivf,
        DistanceType::L2,
        SupportedIvfIndexType::IvfFlat,
    )
    .await?;
    root.add_schema_metadata(FlatIndex::metadata_key(), r#"[""]"#.to_owned());

    Ok(vec![
        IndexFile {
            path: INDEX_AUXILIARY_FILE_NAME.into(),
            size_bytes: auxiliary.finish().await?.size_bytes,
            file_metadata_size_bytes: None,
        },
        IndexFile {
            path: INDEX_FILE_NAME.into(),
            size_bytes: root.finish().await?.size_bytes,
            file_metadata_size_bytes: None,
        },
    ])
}

async fn build_indices(
    table: &Table,
    batch: &RecordBatch,
    include_fts: bool,
) -> Result<Vec<IndexMetadata>> {
    let (store, _) = ObjectStore::from_uri(table.uri()).await?;
    let row_ids = row_addresses(table);
    let fragments: Vec<_> = table
        .manifest()
        .fragments
        .iter()
        .map(|f| f.id as u32)
        .collect();
    let cache = Arc::new(LanceCache::with_capacity(4 * 1024 * 1024));
    let uuid = Uuid::from_u128(1);
    let vector_files =
        write_ivf_flat(table, uuid, batch["vector"].as_fixed_size_list(), &row_ids).await?;
    let vector_details = pb::VectorIndexDetails {
        metric_type: pb::VectorMetricType::L2.into(),
        ..Default::default()
    };

    let bitmap_id = Uuid::from_u128(2);
    let bitmap_store = LanceIndexStore::with_format_version(
        store.clone(),
        table.indices_dir().join(bitmap_id.to_string()),
        cache.clone(),
        FORMAT,
    );
    let keys = RecordBatch::try_from_iter([
        (VALUE_COLUMN_NAME, batch["id"].clone()),
        (ROW_ADDR, Arc::new(row_ids.clone()) as _),
    ])?;
    let trainer = BitmapIndexPlugin;
    let request = trainer.new_training_request("", keys.schema().field(0))?;
    let bitmap = trainer
        .train_index(
            batch_stream(keys),
            &bitmap_store,
            request,
            None,
            lance_index::progress::noop_progress(),
        )
        .await?;
    let mut indices = vec![
        index_metadata(
            table,
            uuid,
            "vector",
            VECTOR_INDEX_VERSION,
            prost_types::Any::from_msg(&vector_details)?,
            fragments.iter().copied(),
            vector_files,
        ),
        index_metadata(
            table,
            bitmap_id,
            "id",
            bitmap.index_version,
            bitmap.index_details,
            fragments.iter().copied(),
            index_files_to_table(bitmap.files),
        ),
    ];

    if include_fts {
        let fts_id = Uuid::from_u128(3);
        let fts_store = LanceIndexStore::with_format_version(
            store,
            table.indices_dir().join(fts_id.to_string()),
            cache,
            FORMAT,
        );
        let indexed_rows = table.manifest().fragments[0].physical_rows.unwrap();
        let documents = RecordBatch::try_from_iter([
            ("text", batch["text"].slice(0, indexed_rows)),
            (ROW_ID, Arc::new(row_ids.slice(0, indexed_rows)) as _),
        ])?;
        let params = InvertedIndexParams::default().with_position(true);
        let index_version = params.resolved_format_version().index_version();
        let details = lance_index::pbold::InvertedIndexDetails::try_from(&params)?;
        let mut builder = InvertedIndexBuilder::new(params);
        let files = builder
            .update(batch_stream(documents), &fts_store, None)
            .await?;
        indices.push(index_metadata(
            table,
            fts_id,
            "text",
            index_version,
            prost_types::Any::from_msg(&details)?,
            [fragments[0]],
            index_files_to_table(files),
        ));
    }
    Ok(indices)
}

async fn publish(table: &Table, mut manifest: Manifest, indices: Vec<IndexMetadata>) -> Result<()> {
    let (store, base) = ObjectStore::from_uri(table.uri()).await?;
    apply_feature_flags(&mut manifest, false, false)?;
    let path = table
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, manifest.version);
    write_manifest_file_to_path(&store, &mut manifest, Some(indices), &path, None).await?;
    Ok(())
}

async fn indexed_table(dir: &TempDir) -> Result<Dataset> {
    let uri = dir.path().to_str().unwrap();
    let batch = rows();
    let table = create_table(uri, batch.clone(), 4).await?;
    assert_eq!(table.manifest().fragments.len(), 2);
    let indices = build_indices(&table, &batch, true).await?;
    let manifest = Manifest::new_from_previous(
        table.manifest(),
        table.schema().clone(),
        table.manifest().fragments.clone(),
    );
    publish(&table, manifest, indices).await?;
    Dataset::open(uri).await
}

#[tokio::test]
async fn dataset_and_provider_pushdown() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let base = create_table(dir.path().to_str().unwrap(), rows(), 4).await?;
    let dataset = DatasetBuilder::from_uri(base.uri())
        .with_version(1)
        .load()
        .await?;
    assert_eq!(dataset.count_fragments(), 2);
    assert_eq!(
        dataset
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        vec!["id", "text", "vector"]
    );
    let fragment = &dataset.get_fragments()[1];
    assert_eq!(fragment.metadata().physical_rows, Some(4));
    assert_eq!(fragment.metadata().files.len(), 1);
    let projection = dataset.schema().project(&["id"])?;
    let batches: Vec<_> = fragment
        .read_physical_slice(1..3, &projection, 1)
        .await?
        .buffered(2)
        .try_collect()
        .await?;
    let sliced: Vec<_> = batches
        .iter()
        .flat_map(|batch| batch_ids(batch).iter().copied())
        .collect();
    assert_eq!(sliced, [5, 6]);

    let mut scanner = dataset.scan();
    let batch = scanner
        .project(&["id"])?
        .filter("id >= 2")?
        .limit(Some(2), None)?
        .try_into_batch()
        .await?;
    assert_eq!(batch_ids(&batch), [2, 3]);

    let provider = LanceTableProvider::new(Arc::new(dataset), false, false);
    let session = TestSession::default();
    let plan = provider
        .scan(&session, Some(&[1]), &[col("id").gt_eq(lit(2i32))], Some(2))
        .await?;
    assert_eq!(plan.schema().field(0).name(), "text");
    assert_eq!(plan.schema().fields().len(), 1);
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("id") && plan_display.contains("2"),
        "{plan_display}"
    );
    let actual = collect(plan, datafusion_session::Session::task_ctx(&session)).await?;
    let text: Vec<_> = actual
        .iter()
        .flat_map(|batch| batch["text"].as_string::<i32>().iter().flatten())
        .collect();
    assert_eq!(text, [TEXTS[2], TEXTS[3]]);
    Ok(())
}

#[tokio::test]
async fn blob_v2_descriptor_and_range_read() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let table_path = dir.path().join("blob.lance");
    let external_path = dir.path().join("payload.bin");
    let offline_path = dir.path().join("payload.offline");
    let bytes: Vec<u8> = (0..=u8::MAX).cycle().take(1024 * 1024).collect();
    std::fs::write(&external_path, &bytes)?;
    let uri = url::Url::from_file_path(&external_path)
        .unwrap()
        .to_string();
    let offset = 113;
    let mut blobs = BlobDescriptorArrayBuilder::new("payload");
    blobs.push_external(
        uri.clone(),
        Some(BlobRange {
            offset,
            size: bytes.len() as u64 - offset,
        }),
    )?;
    blobs.push_inline(b"small inline blob".as_slice())?;
    let (field, array) = blobs.finish()?.into_parts();
    let schema = ArrowSchema::new(vec![
        Field::new("id", arrow_schema::DataType::Int32, false),
        field,
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int32Array::from(vec![0, 1])), array],
    )?;
    let path = table_path.to_str().unwrap();
    InsertBuilder::new(path)
        .with_params(
            &WriteParams::with_storage_version(LanceFileVersion::V2_2)
                .with_allow_external_blob_outside_bases(true),
        )
        .execute(vec![batch])
        .await?;

    // The descriptor path must work while the payload itself is inaccessible.
    std::fs::rename(&external_path, &offline_path)?;
    let dataset = Dataset::open(path).await?;
    let mut scanner = dataset.scan();
    let descriptors = scanner
        .project(&["payload"])?
        .blob_handling(BlobHandling::BlobsDescriptions)
        .try_into_batch()
        .await?;
    let descriptors = descriptors["payload"]
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert!(matches!(
        StoredBlob::parse(descriptors, 0)?,
        Some(StoredBlob::External { uri: stored_uri, position, size, .. })
            if stored_uri == uri && position == offset && size == bytes.len() as u64 - offset
    ));

    std::fs::rename(&offline_path, &external_path)?;
    let files = Arc::new(dataset)
        .take_blobs_by_indices(&[0], "payload")
        .await?;
    let file = files[0].as_ref().expect("payload should exist");
    assert_eq!(file.kind(), BlobKind::External);
    assert_eq!(file.size(), bytes.len() as u64 - offset);
    assert_eq!(file.position(), offset);
    assert_eq!(file.uri(), Some(uri.as_str()));
    assert_eq!(file.range_submission_count(), 0);
    assert_eq!(
        file.read_range(137..191).await?.as_ref(),
        &bytes[offset as usize + 137..offset as usize + 191]
    );
    assert_eq!(file.range_submission_count(), 1);
    Ok(())
}

#[tokio::test]
async fn ivf_ann_uses_scalar_prefilter_and_take() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let dataset = indexed_table(&dir).await?;
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("id_idx").await?)?;
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_indexed_rows"], 8);
    let query = Float32Array::from(vec![0.0, 0.0]);
    let mut scan = dataset.scan();
    scan.project(&["id", "text"])?
        .filter("id IN (4, 5, 6, 7)")?
        .nearest("vector", &query, 2)?
        .prefilter(true);
    let plan = scan.explain_plan(false).await?;
    assert!(
        plan.contains("ANNIvf")
            && plan.contains("ScalarIndex")
            && plan.contains("source=stream(_rowid)"),
        "{plan}"
    );
    let batch = scan.try_into_batch().await?;
    // Exact ground truth for L2 with the query and scalar predicate above.
    assert_eq!(batch_ids(&batch), [4, 5]);
    assert_eq!(batch["text"].as_string::<i32>().value(0), TEXTS[4]);
    assert_eq!(batch["text"].as_string::<i32>().value(1), TEXTS[5]);
    Ok(())
}

#[tokio::test]
async fn inverted_bm25_and_compound_residual() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let dataset = indexed_table(&dir).await?;
    let mut scan = dataset.scan();
    let terms = FullTextSearchQuery::new("mars".to_owned()).with_column("text".to_owned())?;
    let matches = scan
        .project(&["id"])?
        .full_text_search(terms)?
        .try_into_batch()
        .await?;
    let mut ids = batch_ids(&matches).to_vec();
    ids.sort_unstable();
    assert_eq!(ids, [0, 2, 4, 6]);
    assert!(
        matches[SCORE_COL]
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .all(|score| score.is_finite() && *score > 0.0)
    );

    let leaf = |text: &str| MatchQuery::new(text.to_owned()).with_column(Some("text".to_owned()));
    let boolean = BooleanQuery::new([])
        .with_must(leaf("mars").into())
        .with_must(leaf("moon").into())
        .with_must_not(leaf("earth").into());
    let mut scan = dataset.scan();
    scan.project(&["id"])?
        .full_text_search(FullTextSearchQuery::new_query(boolean.into()).limit(Some(4)))?;
    let plan = scan.explain_plan(false).await?;
    assert!(plan.contains("HybridCompound"), "{plan}");
    let mut ids = batch_ids(&scan.try_into_batch().await?).to_vec();
    ids.sort_unstable();
    assert_eq!(ids, [0, 4, 6]);

    let phrase = PhraseQuery::new("mars moon".to_owned()).with_column(Some("text".to_owned()));
    let mut scan = dataset.scan();
    scan.project(&["id"])?
        .full_text_search(FullTextSearchQuery::new_query(phrase.into()))?;
    let mut ids = batch_ids(&scan.try_into_batch().await?).to_vec();
    ids.sort_unstable();
    assert_eq!(ids, [0, 4]);
    Ok(())
}

#[rstest::rstest]
#[case::v0(false)]
#[case::v1(true)]
#[tokio::test]
async fn compacted_snapshot_replays_overlay_and_deletion(#[case] tagged: bool) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = dir.path().to_str().unwrap();
    let batch = rows();
    let table = create_table(uri, batch.clone(), 4).await?;
    let mut indices = build_indices(&table, &batch, false).await?;
    let (store, base) = ObjectStore::from_uri(uri).await?;
    let transaction = InsertBuilder::new(&table)
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![batch])
        .await?;
    let Operation::Append { mut fragments } = transaction.operation else {
        panic!("fixture append did not produce fragments");
    };
    assert_eq!(fragments.len(), 1);
    let compacted = &mut fragments[0];
    compacted.id = 2;
    let mut changed = Vec::new();
    row_addresses(&table)
        .values()
        .iter()
        .copied()
        .collect::<RoaringTreemap>()
        .serialize_into(&mut changed)?;
    let group = FragReuseGroup {
        changed_row_addrs: changed,
        old_frags: table.manifest().fragments.iter().map(Into::into).collect(),
        new_frags: vec![(&*compacted).into()],
    };
    let mut history = if tagged {
        let transition = Transition {
            sources: group.old_frags.iter().map(Into::into).collect(),
            destinations: group.new_frags.iter().map(Into::into).collect(),
            mapping: Some(transition::Mapping::OrderedCompaction(OrderedCompaction {
                changed_row_addrs: group.changed_row_addrs,
            })),
        };
        frag_reuse_entry_appending(&table, vec![transition]).await?
    } else {
        build_new_frag_reuse_index(&table, vec![group], RoaringBitmap::from_iter([2]), 2).await?
    };
    history.dataset_version = 2;
    assert_eq!(history.name, FRAG_REUSE_INDEX_NAME);
    indices.push(history);
    compacted.deletion_file = write_deletion_file(
        &base,
        compacted.id,
        2,
        &DeletionVector::from_iter([3]),
        &store,
        0,
    )
    .await?;
    assert!(
        compacted
            .deletion_file
            .as_ref()
            .is_some_and(|f| !f.is_inline())
    );

    let overlay = record_batch!(("id", Int32, [101])).unwrap();
    let path = "fixture-overlay.lance";
    let field_id = table.schema().field_id("id")?;
    let schema = table.schema().project(&["id"])?;
    let mut writer = create_writer(
        FORMAT,
        store.create(&table.data_dir().join(path)).await?,
        schema,
        Default::default(),
    )?;
    writer.write_batch(&overlay).await?;
    let size = writer.finish().await?.size_bytes;
    compacted.overlays.push(DataOverlayFile {
        data_file: DataFile::new(
            path,
            vec![field_id],
            vec![0],
            FORMAT,
            NonZeroU64::new(size),
            None,
        ),
        coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([1]))),
        committed_version: 2,
    });
    let mut manifest = Manifest::new_from_previous(
        table.manifest(),
        table.schema().clone(),
        Arc::new(fragments),
    );
    manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    publish(&table, manifest, indices).await?;

    let dataset = Dataset::open(uri).await?;
    let indices = dataset.load_indices().await?;
    for name in ["id_idx", "vector_idx"] {
        let index = indices.iter().find(|index| index.name == name).unwrap();
        assert_eq!(index.fragment_bitmap, Some(RoaringBitmap::from_iter([2])));
    }
    assert_eq!(dataset.count_rows(None).await?, 7);
    assert_eq!(dataset.count_deleted_rows().await?, 1);
    let logical = dataset.scan().project(&["id"])?.try_into_batch().await?;
    assert_eq!(batch_ids(&logical), [0, 101, 2, 4, 5, 6, 7]);
    let mut scalar = dataset.scan();
    scalar.project(&["id"])?.filter("id IN (1, 3, 4, 101)")?;
    let plan = scalar.explain_plan(false).await?;
    assert!(plan.contains("ScalarIndex"), "{plan}");
    let mut ids = batch_ids(&scalar.try_into_batch().await?).to_vec();
    ids.sort_unstable();
    assert_eq!(ids, [4, 101]);

    // A warm open must apply the same mapping and deletion filtering as the cold open.
    for _ in 0..2 {
        let mut scan = dataset.scan();
        scan.project(&["id", "text"])?
            .filter("id >= 100")?
            .nearest("vector", &Float32Array::from(vec![10.0, 0.0]), 1)?
            .prefilter(true);
        assert!(scan.explain_plan(false).await?.contains("ANNIvf"));
        let batch = scan.try_into_batch().await?;
        assert_eq!(batch_ids(&batch), [101]);
        assert_eq!(batch["text"].as_string::<i32>().value(0), TEXTS[1]);
    }
    Ok(())
}

fn copy_tree(source: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let output = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &output)?;
        } else {
            std::fs::copy(entry.path(), output)?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn historical_deletion_blob_and_legacy_vector_details() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let deletion = dir.path().join("deletions.lance");
    copy_tree(&data.join("v0.7.5/with_deletions"), &deletion)?;
    let dataset = Dataset::open(deletion.to_str().unwrap()).await?;
    assert_eq!(dataset.count_rows(None).await?, 90);
    assert_eq!(dataset.count_deleted_rows().await?, 10);
    assert_eq!(dataset.scan().try_into_batch().await?.num_rows(), 90);

    let blob_path = dir.path().join("legacy-blob.lance");
    copy_tree(
        &data.join("v12.0.0/blobs/legacy_fragment.lance"),
        &blob_path,
    )?;
    let legacy_blob = Arc::new(Dataset::open(blob_path.to_str().unwrap()).await?);
    assert!(!legacy_blob.schema().field("blob").unwrap().is_blob_v2());
    let blobs = legacy_blob
        .take_blobs_by_indices(&[0, 1, 2, 3], "blob")
        .await?;
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await?.as_ref(),
        b"legacy bytes"
    );
    assert!(blobs[1].is_none());
    assert_eq!(blobs[2].as_ref().unwrap().size(), 0);
    assert_eq!(
        blobs[3].as_ref().unwrap().read().await?.as_ref(),
        b"fragment bytes"
    );

    let indexed = dir.path().join("legacy-vector.lance");
    copy_tree(&data.join("v0.18.2/ivf_hnsw_pq"), &indexed)?;
    let dataset = Dataset::open(indexed.to_str().unwrap()).await?;
    let indices = dataset.load_indices().await?;
    let metadata = indices
        .iter()
        .find(|m| {
            m.index_details
                .as_ref()
                .is_some_and(|d| d.type_url.ends_with("VectorIndexDetails"))
        })
        .expect("legacy vector details must be inferred");
    let field = dataset.schema().field_path(metadata.fields[0])?;
    let index = dataset
        .open_vector_index(&field, &metadata.uuid, &NoOpMetricsCollector)
        .await?;
    assert!(index.num_rows() > 0);
    Ok(())
}

#[tokio::test]
async fn mem_wal_index_open_fails_closed() -> Result<()> {
    use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
    use lance_table::format::pb::MemWalIndexDetails;

    let dir = tempfile::tempdir()?;
    let table = create_table(dir.path().to_str().unwrap(), rows(), IDS.len()).await?;
    let uuid = Uuid::from_u128(4);
    let mut metadata = index_metadata(
        &table,
        uuid,
        "id",
        0,
        prost_types::Any::from_msg(&MemWalIndexDetails::default())?,
        [],
        Vec::new(),
    );
    metadata.name = MEM_WAL_INDEX_NAME.to_owned();
    metadata.fields.clear();
    let manifest = Manifest::new_from_previous(
        table.manifest(),
        table.schema().clone(),
        table.manifest().fragments.clone(),
    );
    publish(&table, manifest, vec![metadata]).await?;
    let dataset = Dataset::open(table.uri()).await?;
    let Err(error) = dataset
        .open_generic_index("id", &uuid, &NoOpMetricsCollector)
        .await
    else {
        panic!("MemWAL must be rejected by the read-only reader");
    };
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("MemWAL"));
    Ok(())
}
