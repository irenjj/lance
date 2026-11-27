// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::RecordBatch;
use futures::Stream;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::deepsize::DeepSizeOf;

use crate::session::index_caches::DSIndexCache;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Projectable, Projection};
use lance_core::traits::DatasetTakeRows;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::reader::FileReaderOptions;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, StorageOptions, StorageOptionsAccessor, WrappingObjectStore,
};
use lance_table::format::{DataFile, Fragment, IndexMetadata, Manifest};
use lance_table::io::commit::{CommitHandler, CommitLock, ManifestLocation};

use lance_table_runtime::commit::detect_overlapping_fragments;
pub(crate) use lance_table_runtime::paths::{DATA_DIR, INDICES_DIR, TRANSACTIONS_DIR};
use lance_table_runtime::{Table, TableState, TableView};
pub use lance_table_runtime::{Version, VersionRef, VersionTransaction};
use object_store::path::Path;
use rand::{Rng, seq::index::sample as sample_indices};
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{Instrument, debug_span, field::Empty, instrument};

pub(crate) mod blob;
pub(crate) use lance_table_runtime::branch_location;
pub mod builder;
pub mod deletions;
pub mod files;
pub mod fragment;
pub(crate) mod fragment_slice;
pub mod index;
pub(crate) mod overlay;
pub use lance_table_runtime::refs;
pub mod rowids;
pub mod scanner;
pub mod statistics;
mod take;
/// Transaction definitions for updating datasets
///
/// Prior to creating a new manifest, a transaction must be created representing
/// the changes being made to the dataset. By representing them as incremental
/// changes, we can detect whether concurrent operations are compatible with
/// one another. We can also rebuild manifests when retrying committing a
/// manifest.
///
/// The definitions live in [`lance_table::transaction`]: building a manifest from
/// a transaction reads and writes only table metadata, so it belongs at the table
/// layer. This module re-exports them at the path callers have always used.
///
/// For more details please refer to the
/// [Transaction Specification](https://lance.org/format/table/transaction/#transaction-types).
pub mod transaction {
    pub use lance_table::transaction::{
        DataOverlayGroup, DataReplacementGroup, Operation, ReadVersionState, RewriteGroup,
        RewrittenIndex, TaggedRewriteAssembly, Transaction, TransactionBuilder, UpdateMap,
        UpdateMapEntry, UpdateMode, UpdatedFragmentOffsets, translate_config_updates,
        translate_schema_metadata_updates, validate_operation,
    };
}
pub mod udtf;
mod utils;
pub(crate) mod versions;

pub(crate) use take::row_offsets_to_row_addresses;

use self::builder::DatasetBuilder;
use self::fragment::FileFragment;
use self::refs::Refs;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::statistics::DatasetStatistics;
use self::transaction::Transaction;
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::refs::{BranchContents, BranchIdentifier, Branches, Tags};
use crate::datatypes::Schema;
use crate::session::Session;
use crate::{Error, Result};
pub use blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobRange, ReadBlobRangesBuilder,
    ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
};
pub use lance_core::ROW_ID;
pub use take::TakeBuilder;

// Full samples already have their final ordering and need no randomness.
fn sample_row_offsets(rng: &mut impl Rng, num_rows: usize, n: usize) -> Vec<u64> {
    let sample_size = n.min(num_rows);
    if sample_size == 0 {
        return Vec::new();
    }
    if sample_size == num_rows {
        return (0..num_rows as u64).collect();
    }
    let mut offsets: Vec<u64> = sample_indices(rng, num_rows, sample_size)
        .into_iter()
        .map(|offset| offset as u64)
        .collect();
    offsets.sort_unstable();
    offsets
}

// We default to 6GB for the index cache, since indices are often large but
// worth caching.
pub const DEFAULT_INDEX_CACHE_SIZE: usize = 6 * 1024 * 1024 * 1024;
// Default to 1 GiB for the metadata cache. Column metadata can be like 40MB,
// so this should be enough for a few hundred columns. Other metadata is much
// smaller.
pub use lance_table_runtime::session::DEFAULT_METADATA_CACHE_SIZE;

/// Lance Dataset
///
/// The manifest and refs are available directly on a checked-out handle.
///
/// ```
/// # use lance::Dataset;
/// # use std::sync::Arc;
/// # fn inspect(mut dataset: Dataset) {
/// let Dataset { manifest, refs, .. } = &dataset;
/// let _version = manifest.version;
/// let _tags = refs.tags();
/// Arc::make_mut(&mut dataset.manifest).tag = Some("local".into());
/// # }
/// ```
#[derive(Clone)]
pub struct Dataset {
    pub(crate) table_state: TableState,
    pub manifest: Arc<Manifest>,
    pub refs: Refs,
    pub(crate) session: Arc<Session>,
    pub(crate) index_cache: Arc<DSIndexCache>,
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("uri", &self.table_state.uri)
            .field("base", &self.table_state.base)
            .field("version", &self.manifest.version)
            .field("cache_num_items", &self.session.approx_num_items())
            .field(
                "base_store_params",
                &self.table_state.base_store_params.is_some(),
            )
            .finish()
    }
}

/// Customize read behavior of a dataset.
#[derive(Clone, Debug)]
pub struct ReadParams {
    /// Size of the index cache in bytes. This cache stores index data in memory
    /// for faster lookups. The default is 6 GiB.
    pub index_cache_size_bytes: usize,

    /// Size of the metadata cache in bytes. This cache stores metadata in memory
    /// for faster open table and scans. The default is 1 GiB.
    pub metadata_cache_size_bytes: usize,

    /// If present, dataset will use this shared [`Session`] instead creating a new one.
    ///
    /// This is useful for sharing the same session across multiple datasets.
    pub session: Option<Arc<Session>>,

    pub store_options: Option<ObjectStoreParams>,

    /// If present, dataset will use this to resolve the latest version
    ///
    /// Lance needs to be able to make atomic updates to the manifest.  This involves
    /// coordination between readers and writers and we can usually rely on the filesystem
    /// to do this coordination for us.
    ///
    /// Some file systems (e.g. S3) do not support atomic operations.  In this case, for
    /// safety, we recommend an external commit mechanism (such as dynamodb) and, on the
    /// read path, we need to reach out to that external mechanism to figure out the latest
    /// version of the dataset.
    ///
    /// If this is not set then a default behavior is chosen that is appropriate for the
    /// filesystem.
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,

    /// File reader options to use when reading data files.
    ///
    /// This allows control over features like caching repetition indices and validation.
    /// Options set here act as dataset-level defaults and can be overridden on a
    /// per-scan basis via [`Scanner::batch_size_bytes`](crate::dataset::scanner::Scanner::batch_size_bytes) or
    /// [`Scanner::with_file_reader_options`](crate::dataset::scanner::Scanner::with_file_reader_options).
    pub file_reader_options: Option<FileReaderOptions>,
}

impl ReadParams {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `index_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn index_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 20 * 1024 * 1024; // 20 MiB per entry
        self.index_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    pub fn index_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.index_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `metadata_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn metadata_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 10 * 1024 * 1024; // 10 MiB per entry
        self.metadata_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// Set the cache size for the file metadata in bytes.
    pub fn metadata_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.metadata_cache_size_bytes = cache_size;
        self
    }

    /// Set a shared session for the datasets.
    pub fn session(&mut self, session: Arc<Session>) -> &mut Self {
        self.session = Some(session);
        self
    }

    /// Use the explicit locking to resolve the latest version
    pub fn set_commit_lock<T: CommitLock + Send + Sync + 'static>(&mut self, lock: Arc<T>) {
        self.commit_handler = Some(Arc::new(lock));
    }

    /// Set the file reader options.
    pub fn file_reader_options(&mut self, options: FileReaderOptions) -> &mut Self {
        self.file_reader_options = Some(options);
        self
    }
}

impl Default for ReadParams {
    fn default() -> Self {
        Self {
            index_cache_size_bytes: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size_bytes: DEFAULT_METADATA_CACHE_SIZE,
            session: None,
            store_options: None,
            commit_handler: None,
            file_reader_options: None,
        }
    }
}

/// Runtime-only configuration applied to an already-open [`Dataset`] handle.
///
/// Every field is optional so a caller can change one part of the handle without
/// rebuilding the manifest, object stores, or inherited base bindings.
#[derive(Default)]
pub struct DatasetRuntimeConfig {
    /// New index cache capacity. Ignored when `session` is also set.
    pub index_cache_size_bytes: Option<usize>,
    /// New metadata cache capacity. Ignored when `session` is also set.
    pub metadata_cache_size_bytes: Option<usize>,
    /// New primary object-store block-size hint.
    pub block_size: Option<usize>,
    /// Replacement session. Wins over the cache sizes above.
    pub session: Option<Arc<Session>>,
    /// Replacement commit handler.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,
    /// Replacement storage-options accessor for the primary store. The primary
    /// store is rebuilt through the session registry so subsequent primary I/O
    /// consults this accessor.
    pub storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    /// Explicit per-base bindings to merge. Caller entries win over inherited ones.
    pub base_store_params: HashMap<String, ObjectStoreParams>,
}

#[derive(Debug, Clone)]
pub enum ProjectionRequest {
    Schema(Arc<Schema>),
    Sql(Vec<(String, String)>),
}

impl ProjectionRequest {
    /// Project `columns`, by name or nested path, out of `dataset_schema`.
    ///
    /// # Panics
    ///
    /// If a column, or a nested path below it, is not in the schema. Use
    /// [`Self::try_from_columns`] for column names that come from a user.
    pub fn from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Self {
        Self::try_from_columns(columns, dataset_schema).unwrap()
    }

    /// Project `columns`, by name or nested path, out of `dataset_schema`, or
    /// return [`Error::FieldNotFound`] for a column the schema does not have.
    ///
    /// ```
    /// # use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    /// # use lance::dataset::ProjectionRequest;
    /// # use lance_core::datatypes::Schema;
    /// let schema = Schema::try_from(&ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]))?;
    /// assert!(ProjectionRequest::try_from_columns(["id"], &schema).is_ok());
    /// assert!(ProjectionRequest::try_from_columns(["missing"], &schema).is_err());
    /// # Ok::<(), lance::Error>(())
    /// ```
    pub fn try_from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Result<Self> {
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let schema = dataset_schema.project_preserve_system_columns(&columns)?;
        Ok(Self::Schema(Arc::new(schema)))
    }

    pub fn from_schema(schema: Schema) -> Self {
        Self::Schema(Arc::new(schema))
    }

    /// Provide a list of projection with SQL transform.
    ///
    /// # Parameters
    /// - `columns`: A list of tuples where the first element is resulted column name and the second
    ///   element is the SQL expression.
    pub fn from_sql(
        columns: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self::Sql(
            columns
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect(),
        )
    }

    pub fn into_projection_plan(self, dataset: Arc<Dataset>) -> Result<ProjectionPlan> {
        match self {
            Self::Schema(schema) => {
                // The schema might contain system columns (_rowid, _rowaddr) which are not
                // in the dataset schema. We handle these specially in ProjectionPlan::from_schema.
                let system_columns_present = schema
                    .fields
                    .iter()
                    .any(|f| lance_core::is_system_column(&f.name));

                if system_columns_present {
                    // If system columns are present, we can't use project_by_schema directly
                    // Just pass the schema to ProjectionPlan::from_schema which handles it
                    ProjectionPlan::from_schema(dataset, schema.as_ref())
                } else {
                    // No system columns, use normal path with validation
                    let projection = dataset.schema().project_by_schema(
                        schema.as_ref(),
                        OnMissing::Error,
                        OnTypeMismatch::Error,
                    )?;
                    ProjectionPlan::from_schema(dataset, &projection)
                }
            }
            Self::Sql(columns) => ProjectionPlan::from_expressions(dataset, &columns),
        }
    }
}

impl From<Arc<Schema>> for ProjectionRequest {
    fn from(schema: Arc<Schema>) -> Self {
        Self::Schema(schema)
    }
}

impl From<Schema> for ProjectionRequest {
    fn from(schema: Schema) -> Self {
        Self::from(Arc::new(schema))
    }
}

impl Dataset {
    pub(crate) fn from_table(table: Table, session: Arc<Session>) -> Self {
        let index_cache = Arc::new(session.index_cache.for_dataset(table.uri()));
        Self::from_table_with_index_cache(table, session, index_cache)
    }

    pub(crate) fn from_committed_table(table: Table, session: Arc<Session>) -> Self {
        let index_cache = Arc::new(DSIndexCache(table.index_cache().clone()));
        Self::from_table_with_index_cache(table, session, index_cache)
    }

    pub(crate) fn from_table_with_index_cache(
        table: Table,
        session: Arc<Session>,
        index_cache: Arc<DSIndexCache>,
    ) -> Self {
        let (table_state, manifest, refs) = table.into_parts();
        Self {
            table_state,
            manifest,
            refs,
            session,
            index_cache,
        }
    }

    pub(crate) fn table_view(&self) -> TableView<'_> {
        TableView::new(&self.table_state, &self.manifest, &self.refs)
    }

    /// Open an existing dataset.
    ///
    /// See also [DatasetBuilder].
    #[instrument]
    pub async fn open(uri: &str) -> Result<Self> {
        DatasetBuilder::from_uri(uri).load().await
    }

    /// Check out a dataset version with a ref.
    ///
    /// In reference contexts, `"main"` is an alias for the default branch and
    /// is equivalent to `None`.
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Self> {
        match self.table_view().checkout_version(version).await? {
            Some(table) => Ok(Self::from_table(table, self.session.clone())),
            None => Ok(self.clone()),
        }
    }

    pub fn tags(&self) -> Tags<'_> {
        self.refs.tags()
    }

    /// A handle for cheap, index-derived statistics about this dataset (e.g. a
    /// column's global value range) that never scan data.
    pub fn statistics(&self) -> DatasetStatistics<'_> {
        DatasetStatistics::new(self)
    }

    pub fn branches(&self) -> Branches<'_> {
        self.refs.branches()
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        let (manifest, manifest_location) = self.latest_manifest().await?;
        self.set_manifest(manifest, manifest_location);
        Ok(())
    }

    /// Replace the manifest, refreshing derived state. Base stores are kept
    /// when `base_paths` is unchanged.
    fn set_manifest(&mut self, manifest: Arc<Manifest>, manifest_location: ManifestLocation) {
        self.table_state
            .set_manifest(&mut self.manifest, manifest, manifest_location);
    }

    /// Check out the latest version of the branch.
    ///
    /// Use `"main"` to check out the latest version of the default branch.
    pub async fn checkout_branch(&self, branch: &str) -> Result<Self> {
        self.checkout_version((branch, None)).await
    }

    pub async fn list_branches(&self) -> Result<HashMap<String, BranchContents>> {
        self.branches().list().await
    }

    /// Get the fully qualified URI of this dataset.
    pub fn uri(&self) -> &str {
        &self.table_state.uri
    }

    /// The commit handler this dataset commits through. Lets a caller wrap it
    /// for a snapshot via [`DatasetBuilder::with_commit_handler`], for example
    /// to fence that commit to an exact version, without reopening the
    /// dataset or knowing how the handler was chosen.
    pub fn commit_handler(&self) -> Arc<dyn CommitHandler> {
        self.table_state.commit_handler.clone()
    }

    pub fn branch_location(&self) -> BranchLocation {
        self.table_view().branch_location()
    }

    pub async fn branch_identifier(&self) -> Result<BranchIdentifier> {
        self.table_view().branch_identifier().await
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        self.table_view().manifest()
    }

    pub fn manifest_location(&self) -> &ManifestLocation {
        self.table_view().manifest_location()
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        self.table_view().latest_manifest().await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        self.table_view().read_transaction().await
    }

    /// Read the transaction (if any) and commit timestamp of a version of the
    /// dataset. `version` is a version number on this dataset's current branch.
    ///
    /// When `version` is the version this handle is checked out at, the
    /// in-memory manifest is used and the transaction is read directly from
    /// storage, so no manifest download is needed. Otherwise the version's
    /// manifest is read transiently: no historical `Dataset` is constructed,
    /// no `IndexSection` is decoded, and no session cache is written, so
    /// scanning many historical versions does not fill the shared caches. A
    /// manifest that the session cache already holds (for example after
    /// `checkout_version`) is reused instead of being downloaded again.
    ///
    /// Neither path consults or populates the session's transaction cache
    /// (unlike [`Self::read_transaction`]): that cache is keyed only by
    /// version number, which is not safe to reuse across a dataset dropped
    /// and recreated at the same URI within the same session (see #8916).
    ///
    /// Returns an error if the version does not exist (for example, if it has
    /// been cleaned up).
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let record = dataset.read_version_transaction(5).await?;
    /// let committed_at = record.timestamp;
    /// let operation = record.transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_version_transaction(&self, version: u64) -> Result<VersionTransaction> {
        self.table_view().read_version_transaction(version).await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    ///
    /// Does not populate the session caches; see
    /// [`Self::read_version_transaction`].
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let transaction = dataset.read_transaction_by_version(5).await?;
    /// let operation = transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        self.table_view().read_transaction_by_version(version).await
    }

    /// List transactions for the dataset, up to a maximum number.
    ///
    /// This method iterates through dataset versions, starting from the current version,
    /// and collects the transaction for each version. It stops when either `recent_transactions`
    /// is reached or there are no more versions.
    ///
    /// # Arguments
    ///
    /// * `recent_transactions` - Maximum number of transactions to return
    ///
    /// # Returns
    ///
    /// A vector of optional transactions. Each element corresponds to a version,
    /// and may be None if no transaction file exists for that version.
    pub async fn get_transactions(
        &self,
        recent_transactions: usize,
    ) -> Result<Vec<Option<Transaction>>> {
        self.table_view()
            .get_transactions(recent_transactions)
            .await
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(Arc::new(self.clone()))
    }

    /// Count the number of rows in the dataset.
    ///
    /// It offers a fast path of counting rows by just computing via metadata.
    #[instrument(skip_all)]
    pub async fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        // TODO: consolidate the count_rows into Scanner plan.
        if let Some(filter) = filter {
            let mut scanner = self.scan();
            scanner.filter(&filter)?;
            Ok(scanner
                .project::<String>(&[])?
                .with_row_id() // TODO: fix scan plan to not require row_id for count_rows.
                .count_rows()
                .await? as usize)
        } else {
            self.count_all_rows().await
        }
    }

    pub(crate) async fn count_all_rows(&self) -> Result<usize> {
        self.table_view().to_table().count_rows().await
    }

    /// Take rows by indices.
    #[instrument(skip_all, fields(num_rows=row_indices.len()))]
    pub async fn take(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        let span = debug_span!(
            "take_io",
            num_rows = row_indices.len(),
            fragments_selected = Empty,
            logical_ranges = Empty,
            logical_requested_bytes = Empty,
            logical_union_bytes = Empty,
            physical_iops = Empty,
            scheduler_requests = Empty,
            physical_bytes = Empty,
            cloud_reader_attempts = Empty,
            cloud_reader_failed_attempts = Empty,
            cloud_reader_body_bytes = Empty,
            peak_active_reads = Empty,
            max_scheduler_queue_time_ns = Empty,
            max_child_read_time_ns = Empty,
            elapsed_ns = Empty,
            outcome = Empty,
        );
        let summary_span = (!span.is_disabled()).then(|| span.clone());
        take::take(self, row_indices, projection.into(), summary_span)
            .instrument(span)
            .await
    }

    /// Take rows with a maximum number of active fragment-reader data reads in the primary base.
    ///
    /// Current-format primary-base data and overlay files opened through the
    /// fragment reader share one scheduler with this request ceiling, and
    /// fragment-reader fanout is capped at `min(store_parallelism, io_parallelism)`.
    /// This does not increase fragment-reader fanout above the store default, so
    /// actual concurrency may be lower. Legacy v1 files, additional-base
    /// schedulers, and auxiliary metadata or lineage reads (such as deletion
    /// vectors and spilled row-ID or row-version sequences) keep their existing
    /// schedulers. It must be greater than zero.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let rows = dataset
    ///     .take_with_io_parallelism(&[0, 1, 2], dataset.schema().clone(), 64)
    ///     .await?;
    /// # let _ = rows;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip_all, fields(num_rows=row_indices.len(), io_parallelism = io_parallelism))]
    pub async fn take_with_io_parallelism(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
        io_parallelism: u32,
    ) -> Result<RecordBatch> {
        let span = debug_span!(
            "take_io",
            num_rows = row_indices.len(),
            io_parallelism,
            fragments_selected = Empty,
            logical_ranges = Empty,
            logical_requested_bytes = Empty,
            logical_union_bytes = Empty,
            physical_iops = Empty,
            scheduler_requests = Empty,
            physical_bytes = Empty,
            cloud_reader_attempts = Empty,
            cloud_reader_failed_attempts = Empty,
            cloud_reader_body_bytes = Empty,
            peak_active_reads = Empty,
            max_scheduler_queue_time_ns = Empty,
            max_child_read_time_ns = Empty,
            elapsed_ns = Empty,
            outcome = Empty,
        );
        let summary_span = (!span.is_disabled()).then(|| span.clone());
        take::take_with_io_parallelism(
            self,
            row_indices,
            projection.into(),
            Some(io_parallelism),
            summary_span,
        )
        .instrument(span)
        .await
    }

    /// Read selected rows by their internal row IDs.
    ///
    /// ```rust
    /// use lance::{Dataset, Result, dataset::ProjectionRequest};
    ///
    /// async fn read_rows(dataset: &Dataset, row_ids: &[u64]) -> Result<()> {
    ///     let schema = dataset.schema().clone();
    ///     let rows = dataset.take_rows(row_ids, schema).await?;
    ///     let projection = ProjectionRequest::from_sql([("identity", "id * 2")]);
    ///     let computed = dataset.take_rows(row_ids, projection).await?;
    ///     assert_eq!(rows.num_rows(), computed.num_rows());
    ///     Ok(())
    /// }
    /// ```
    pub async fn take_rows(
        &self,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        Arc::new(self.clone())
            .take_builder(row_ids, projection)?
            .execute()
            .await
    }

    pub fn take_builder(
        self: &Arc<Self>,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<TakeBuilder> {
        TakeBuilder::try_new_from_ids(self.clone(), row_ids.to_vec(), projection.into())
    }

    /// Take [BlobFile] by row IDs.
    ///
    /// The returned vector has one element per row ID. Null blob values are
    /// represented as `None`; valid empty blobs return a `BlobFile` with size
    /// zero.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset.take_blobs(&[42], "images").await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs(
        self: &Arc<Self>,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs(row_ids, column)
            .await
    }

    /// Take [BlobFile] by row addresses.
    ///
    /// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
    /// Use this method when you already have row addresses, for example from
    /// a scan with `with_row_address()`. For row IDs (stable identifiers), use
    /// [`Self::take_blobs`]. For row indices (offsets), use
    /// [`Self::take_blobs_by_indices`]. The result has the same null and empty
    /// blob representation as [`Self::take_blobs`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>, row_address: u64) -> Result<()> {
    /// let blobs = dataset
    ///     .take_blobs_by_addresses(&[row_address], "images")
    ///     .await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_addresses(
        self: &Arc<Self>,
        row_addrs: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs_by_addresses(row_addrs, column)
            .await
    }

    /// Scan every row of this dataset with its blob v2 columns as values another
    /// dataset can be written from: the payload bytes of every blob this dataset
    /// stores, and an External blob's absolute URI and byte range, at any depth.
    ///
    /// Each batch holds the payloads of its blobs in memory.
    pub async fn scan_blobs_as_logical(self: &Arc<Self>) -> Result<DatasetRecordBatchStream> {
        let mut scan = self.scan();
        scan.with_row_address();
        let batches = scan.try_into_stream().await?;
        let plan = Arc::new(lance_table_runtime::blob::BlobV2BatchRewritePlan::try_new(
            self.schema(),
            lance_io::stream::RecordBatchStream::schema(&batches).as_ref(),
            false,
            false,
        )?);
        let schema = plan.output_schema().clone();
        let dataset = self.clone();
        let batches = batches
            .and_then(move |batch| {
                let (plan, dataset) = (plan.clone(), dataset.clone());
                async move {
                    plan.transform_batch(&dataset.table_view().to_table(), batch)
                        .await
                }
            })
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)));
        Ok(DatasetRecordBatchStream::new(Box::pin(
            datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, batches),
        )))
    }

    /// Take [BlobFile] by row indices (offsets in the dataset).
    ///
    /// The result has the same null and empty blob representation as
    /// [`Self::take_blobs`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset.take_blobs_by_indices(&[0], "images").await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_indices(
        self: &Arc<Self>,
        row_indices: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs_by_indices(row_indices, column)
            .await
    }

    /// Create a planned blob reader for a blob column.
    ///
    /// This API complements [`Self::take_blobs`]. `take_blobs` returns
    /// [`BlobFile`] handles for caller-driven random access, while
    /// `read_blobs` builds a streaming read plan for sequential or batched blob
    /// retrieval. Every selected row produces one result: null blob values have
    /// `ReadBlob::data` set to `None`, while valid empty blobs contain an empty
    /// buffer.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use futures::TryStreamExt;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset
    ///     .read_blobs("images")?
    ///     .with_row_indices(vec![0, 1, 2])
    ///     .execute()
    ///     .await?;
    /// # let _ = blobs;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blobs(self: &Arc<Self>, column: impl AsRef<str>) -> Result<ReadBlobsBuilder> {
        self.table_view().to_table().read_blobs(column)
    }

    /// Create a planned reader for row-specific blob-local byte ranges.
    ///
    /// Each [`BlobRangeRequest`] contains both its row selector and byte range,
    /// so requests can be repeated or reordered without coordinating parallel
    /// selector and range lists. Every request produces one result. A null blob
    /// has `ReadBlobRange::data` set to `None`; an empty range on a non-null blob
    /// contains an empty buffer.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use lance::dataset::{BlobRangeRequest, Dataset};
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let ranges = dataset
    ///     .read_blob_ranges("images")?
    ///     .with_row_indices([
    ///         BlobRangeRequest::new(7, 0, 1024),
    ///         BlobRangeRequest::new(7, 4096, 1024),
    ///     ])
    ///     .execute()
    ///     .await?;
    /// # let _ = ranges;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blob_ranges(
        self: &Arc<Self>,
        column: impl AsRef<str>,
    ) -> Result<ReadBlobRangesBuilder> {
        self.table_view().to_table().read_blob_ranges(column)
    }

    /// Get a stream of batches based on iterator of ranges of row numbers.
    ///
    /// `batch_readahead` must be greater than zero; otherwise the stream yields an invalid-input error.
    ///
    /// This is an experimental API. It may change at any time.
    pub fn take_scan(
        &self,
        row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
        projection: Arc<Schema>,
        batch_readahead: usize,
    ) -> DatasetRecordBatchStream {
        take::take_scan(self, row_ranges, projection, batch_readahead)
    }

    /// Randomly sample up to `n` live rows uniformly without replacement.
    ///
    /// Returns all available live rows if `n` exceeds their count.
    ///
    /// If `fragment_ids` is provided, sampling is limited to rows from those
    /// fragments in the current dataset version.
    ///
    /// The returned rows are in ascending logical row-offset order within the
    /// dataset or selected fragments, allowing efficient sorted reads.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let batch = dataset.sample(10, dataset.schema(), None).await?;
    /// assert!(batch.num_rows() <= 10);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sample(
        &self,
        n: usize,
        projection: &Schema,
        fragment_ids: Option<&[u32]>,
    ) -> Result<RecordBatch> {
        match fragment_ids {
            None => {
                let num_rows = self.count_rows(None).await?;
                let offsets = sample_row_offsets(&mut rand::rng(), num_rows, n);
                self.take(&offsets, projection.clone()).await
            }
            Some(fragment_ids) => {
                if fragment_ids.is_empty() {
                    return Err(Error::invalid_input(
                        "Dataset::sample does not accept an empty fragment_ids list".to_string(),
                    ));
                }

                let selected_fragments = self.get_fragments_from_ids(fragment_ids)?;

                let num_rows = stream::iter(selected_fragments.iter().cloned())
                    .map(|fragment| async move { fragment.count_rows(None).await })
                    .buffer_unordered(16)
                    .try_fold(0_u64, |acc, rows| async move {
                        acc.checked_add(rows as u64).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "Dataset::sample selected live row count overflow: {acc} + {rows}"
                            ))
                        })
                    })
                    .await?;

                let num_rows = usize::try_from(num_rows).map_err(|_| {
                    Error::invalid_input(format!(
                        "Dataset::sample selected live row count {num_rows} exceeds usize::MAX ({})",
                        usize::MAX
                    ))
                })?;
                let offsets = sample_row_offsets(&mut rand::rng(), num_rows, n);

                let row_addrs = row_offsets_to_row_addresses(&selected_fragments, &offsets).await?;
                let dataset = Arc::new(self.clone());
                let projection = Arc::new(
                    ProjectionRequest::from(projection.clone())
                        .into_projection_plan(dataset.clone())?,
                );
                TakeBuilder::try_new_from_addresses(dataset, row_addrs, projection)?
                    .execute()
                    .await
            }
        }
    }

    pub async fn count_deleted_rows(&self) -> Result<usize> {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.count_deletions().await })
            .buffer_unordered(self.table_state.object_store.io_parallelism())
            .try_fold(0, |acc, x| futures::future::ready(Ok(acc + x)))
            .await
    }

    /// Clone this dataset with a different object store binding.
    ///
    /// The returned dataset shares metadata, session state, and caches with the
    /// original dataset, but its primary object store — and the dataset-level
    /// default params used for base paths that have no explicit binding — are
    /// replaced by the supplied ones. Explicit per-base bindings still win: a base
    /// path bound through [`DatasetBuilder::with_base_store_params`], or inherited
    /// from the dataset a shallow clone was taken from, keeps reading through its
    /// own store.
    pub fn with_object_store(
        &self,
        object_store: Arc<ObjectStore>,
        store_params: Option<ObjectStoreParams>,
    ) -> Self {
        let mut cloned = self.clone();
        cloned
            .table_state
            .set_object_store(object_store, store_params);
        cloned
    }

    /// Apply runtime configuration to this dataset handle.
    ///
    /// The returned handle keeps the loaded manifest and every store binding that
    /// `config` does not replace. An explicit session wins over cache sizes, and
    /// cache sizes create a new session on the same store registry so custom
    /// object-store providers keep resolving. When `block_size` is set, the
    /// primary store keeps its existing backend and only changes its read hint.
    /// Replacing `storage_options_accessor` rebuilds the primary store through
    /// the session's registry, because the accessor is consumed when the store is
    /// constructed and must reach the backend that serves primary I/O.
    pub async fn reconfigure(&self, config: DatasetRuntimeConfig) -> Result<Self> {
        let mut ds = self.clone();

        let session = match config.session {
            Some(session) => Some(session),
            None if config.index_cache_size_bytes.is_some()
                || config.metadata_cache_size_bytes.is_some() =>
            {
                Some(Arc::new(Session::new(
                    config
                        .index_cache_size_bytes
                        .unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
                    config
                        .metadata_cache_size_bytes
                        .unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
                    ds.session.store_registry(),
                )))
            }
            None => None,
        };
        if let Some(session) = session {
            ds.index_cache = Arc::new(session.index_cache.for_dataset(&ds.table_state.uri));
            ds.table_state.metadata_cache =
                Arc::new(session.metadata_cache().for_dataset(ds.uri()));
            ds.table_state.set_session(session.table_session.clone());
            ds.session = session;
        }

        let mut store_params = ds.table_state.store_params.as_deref().cloned();
        let accessor_replaced = config.storage_options_accessor.is_some();
        if let Some(accessor) = config.storage_options_accessor {
            let params = store_params.get_or_insert_default();
            params.storage_options_accessor = Some(accessor);
        }
        if let Some(block_size) = config.block_size {
            let params = store_params.get_or_insert_default();
            params.block_size = Some(block_size);
        }

        if accessor_replaced {
            // A store consumes its accessor at construction time, so replacing it
            // in `store_params` alone leaves the opened backend on the previous
            // (possibly expiring) credentials. Rebuild through the handle's
            // session registry so the new accessor, and every other params-bound
            // setting, reach the backend that serves primary metadata, refs, and
            // commits. Explicit base bindings stay untouched; `with_object_store`
            // drops the derived fallback cache so bases are re-resolved from their
            // own bindings, or from the new default params when they have none,
            // instead of keeping the replaced stores.
            let params = store_params.clone().unwrap_or_default();
            let (object_store, _) = ObjectStore::from_uri_and_params(
                ds.session.store_registry(),
                &ds.table_state.uri,
                &params,
            )
            .await?;
            ds = ds.with_object_store(object_store, Some(params));
        } else if let Some(block_size) = config.block_size {
            let params = store_params.unwrap_or_default();
            ds = ds.with_object_store(
                Arc::new(ds.table_state.object_store.with_block_size(block_size)),
                Some(params),
            );
        } else if store_params.is_some() {
            ds.table_state.store_params = store_params.map(Box::new);
        }

        let commit_handler_replaced = config.commit_handler.is_some();
        if let Some(commit_handler) = config.commit_handler {
            ds.table_state.commit_handler = commit_handler;
        }
        if accessor_replaced || config.block_size.is_some() || commit_handler_replaced {
            // Refs binds the primary store and commit handler when it is built.
            ds.refs = Refs::new(
                ds.table_state.object_store.clone(),
                ds.table_state.commit_handler.clone(),
                ds.branch_location(),
            );
        }

        if !config.base_store_params.is_empty() {
            let mut base_store_params = ds
                .table_state
                .base_store_params
                .as_deref()
                .cloned()
                .unwrap_or_default();
            base_store_params.extend(config.base_store_params);
            ds.table_state.base_store_params = Some(Arc::new(base_store_params));
            ds.table_state.clear_base_object_stores();
        }

        Ok(ds)
    }

    /// Clone this dataset with extra object store wrappers applied to all read stores.
    ///
    /// The returned dataset uses the wrappers for the already-open primary object
    /// store and appends the same wrappers to the dataset-level and base-specific
    /// object store params used when additional base stores are opened later.
    pub fn with_object_store_wrappers(
        &self,
        wrappers: impl IntoIterator<Item = Arc<dyn WrappingObjectStore>>,
    ) -> Self {
        let table = self.table_view().with_object_store_wrappers(wrappers);
        Self::from_table_with_index_cache(table, self.session.clone(), self.index_cache.clone())
    }

    /// Returns the initial storage options used when opening this dataset, if any.
    ///
    /// This returns the static initial options without triggering any refresh.
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    #[deprecated(since = "0.25.0", note = "Use initial_storage_options() instead")]
    pub fn storage_options(&self) -> Option<&HashMap<String, String>> {
        self.initial_storage_options()
    }

    /// Returns the initial storage options without triggering any refresh.
    ///
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    pub fn initial_storage_options(&self) -> Option<&HashMap<String, String>> {
        self.table_state.initial_storage_options()
    }

    /// Returns the storage options provider used when opening this dataset, if any.
    pub fn storage_options_provider(
        &self,
    ) -> Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>> {
        self.table_state.storage_options_provider()
    }

    /// Returns the unified storage options accessor for this dataset, if any.
    ///
    /// The accessor handles both static and dynamic storage options with automatic
    /// caching and refresh. Use [`StorageOptionsAccessor::get_storage_options`] to
    /// get the latest options.
    pub fn storage_options_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.table_state.storage_options_accessor()
    }

    /// Returns the latest (possibly refreshed) storage options.
    ///
    /// If a dynamic storage options provider is configured, this will return
    /// the cached options if still valid, or fetch fresh options if expired.
    ///
    /// For the initial static options without refresh, use [`Self::storage_options`].
    ///
    /// # Returns
    ///
    /// - `Ok(Some(options))` - Storage options are available (static or refreshed)
    /// - `Ok(None)` - No storage options were configured for this dataset
    /// - `Err(...)` - Error occurred while fetching/refreshing options from provider
    pub async fn latest_storage_options(&self) -> Result<Option<StorageOptions>> {
        self.table_state.latest_storage_options().await
    }

    pub fn data_dir(&self) -> Path {
        self.table_state.data_dir()
    }

    pub fn indices_dir(&self) -> Path {
        self.table_state.indices_dir()
    }

    pub fn transactions_dir(&self) -> Path {
        self.table_state.transactions_dir()
    }

    pub fn deletions_dir(&self) -> Path {
        self.table_state.deletions_dir()
    }

    pub fn versions_dir(&self) -> Path {
        self.table_state.versions_dir()
    }

    pub(crate) fn data_file_dir(&self, data_file: &DataFile) -> Result<Path> {
        self.data_file_dir_for_base(data_file.base_id)
    }

    /// Resolve the object store and complete object key of a manifest data file.
    ///
    /// Uses the file's registered base when present, including dataset-root bases
    /// used by shallow clones. Otherwise uses this dataset's `data/` directory.
    /// This does not read the file, so it also works for missing objects.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn inspect(dataset: &Dataset) -> Result<()> {
    /// for fragment in dataset.iter_fragments() {
    ///     for file in &fragment.files {
    ///         let (store, path) = dataset.resolve_data_file_location(file).await?;
    ///         let exists = store.exists(&path).await?;
    /// #       let _ = exists;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn resolve_data_file_location(
        &self,
        data_file: &DataFile,
    ) -> Result<(Arc<ObjectStore>, Path)> {
        let path = self.data_file_dir(data_file)?.join(data_file.path.as_str());
        let store = self.object_store_for_data_file(data_file).await?;
        Ok((store, path))
    }

    /// Resolve the data directory for a given base_id.
    ///
    /// If `base_id` is `None`, returns the default data directory.
    pub(crate) fn data_file_dir_for_base(&self, base_id: Option<u32>) -> Result<Path> {
        self.table_view().data_file_dir_for_base(base_id)
    }

    /// Resolve the object store for the primary dataset or an additional base.
    ///
    /// Pass `None` to get the primary dataset object store. Pass `Some(base_id)`
    /// when resolving a file whose metadata references an additional base.
    pub async fn object_store(&self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        self.table_view().object_store(base_id).await
    }

    /// The `ObjectStoreParams` this dataset was opened with, or `None` when
    /// opened without explicit params. Lets a caller re-open a derived path
    /// (e.g. a MemWAL SSTable) with the same store this dataset used.
    pub fn store_params(&self) -> Option<&ObjectStoreParams> {
        self.table_state.store_params.as_deref()
    }

    /// Dataset-level file-reader defaults supplied through [`ReadParams`].
    ///
    /// `None` means no explicit options were supplied and readers use their
    /// defaults. Clone these options when creating a per-read override so that
    /// unrelated settings are preserved without changing the shared dataset.
    ///
    /// ```
    /// # use lance::Dataset;
    /// # use lance::dataset::scanner::Scanner;
    /// fn scan_with_byte_limit(dataset: &Dataset) -> Scanner {
    ///     let mut options = dataset.file_reader_options().cloned().unwrap_or_default();
    ///     options.batch_size_bytes = Some(1024 * 1024);
    ///     let mut scanner = dataset.scan();
    ///     scanner.with_file_reader_options(options);
    ///     scanner
    /// }
    /// ```
    pub fn file_reader_options(&self) -> Option<&FileReaderOptions> {
        self.table_state.file_reader_options.as_deref()
    }

    /// Return a copy of this dataset with updated dataset-level file-reader
    /// defaults.
    ///
    /// The returned handle shares the manifest, object stores, session, and
    /// caches with this dataset. Only the defaults used by subsequent scans
    /// change; see [`Self::file_reader_options`].
    pub fn with_file_reader_options(&self, file_reader_options: Option<FileReaderOptions>) -> Self {
        let mut cloned = self.clone();
        cloned.table_state.file_reader_options = file_reader_options.map(Arc::new);
        cloned
    }

    pub(crate) async fn object_store_for_data_file(
        &self,
        data_file: &DataFile,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(data_file.base_id).await
    }

    pub(crate) async fn object_store_for_index(
        &self,
        index: &IndexMetadata,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(index.base_id).await
    }

    /// Get the indices directory for a specific index, considering its base_id
    pub(crate) fn indice_files_dir(&self, index: &IndexMetadata) -> Result<Path> {
        self.table_view().indice_files_dir(index)
    }

    pub fn session(&self) -> Arc<Session> {
        self.session.clone()
    }

    /// Get the currently checked-out version id.
    ///
    /// This is a cheap accessor that reads the id directly from the loaded
    /// manifest without constructing the full [Version] summary.
    pub fn version_id(&self) -> u64 {
        self.table_view().version_id()
    }

    /// Get the currently checked-out version details.
    ///
    /// This constructs a full [Version], including summary metadata derived
    /// from the loaded manifest fragments.
    pub fn version(&self) -> Version {
        self.table_view().version()
    }

    /// Get the number of entries currently in the index cache.
    pub async fn index_cache_entry_count(&self) -> usize {
        self.session.index_cache.size().await
    }

    /// Get cache hit ratio.
    pub async fn index_cache_hit_rate(&self) -> f32 {
        let stats = self.session.index_cache_stats().await;
        stats.hit_ratio()
    }

    pub fn cache_size_bytes(&self) -> u64 {
        self.session.deep_size_of() as u64
    }

    /// Get all versions, in ascending order.
    ///
    /// Every manifest in the history is read. Use [`Self::versions_with_options`] to
    /// read only a page, or [`Self::count_versions`] to avoid reading manifests at all.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        self.table_state.versions().await
    }

    /// Get one page of versions in the current version history.
    ///
    /// Versions are ordered ascending by version, or descending when `descending`
    /// is true. `offset` is the number of versions to skip and `limit` is the
    /// maximum number to return, so `versions_with_options(0, None, false)` is the
    /// whole history. `None` means no limit, `Some(0)` returns an empty page without
    /// listing or reading any manifest, and an `offset` past the end of the history
    /// returns an empty page. Detached versions are never part of the history.
    ///
    /// Only the manifests in the returned page are read, concurrently within the
    /// store's effective IO window.
    ///
    /// With `descending` on a store whose listing is lexically ordered and whose
    /// manifests are V2 paths, listing stops after `offset + limit` manifest entries
    /// have been observed, so a page near the newest version does not enumerate the
    /// history. Every other case (ascending pages, V1 manifest paths, or a listing
    /// that is not lexically ordered) enumerates every manifest location before
    /// selecting the page. These are listing-entry bounds: a store may still
    /// materialize more keys, or issue more requests, than the entries it yields.
    pub async fn versions_with_options(
        &self,
        offset: usize,
        limit: Option<usize>,
        descending: bool,
    ) -> Result<Vec<Version>> {
        self.table_state
            .versions_with_options(offset, limit, descending)
            .await
    }

    /// Get the number of versions in the current version history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest.
    pub async fn count_versions(&self) -> Result<u64> {
        self.table_state.count_versions().await
    }

    /// List lightweight references to all attached versions in the current branch's history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest. The references are sorted by version in ascending order.
    /// Detached manifests are excluded; see [`Self::list_detached_manifests`].
    ///
    /// Use [`Self::latest_version_id`] instead when only the latest version is needed.
    pub async fn version_refs(&self) -> Result<Vec<VersionRef>> {
        self.table_state.version_refs().await
    }

    /// List all detached manifest locations.
    ///
    /// Detached manifests are versions that are not part of the main version history.
    /// They are created by `commit_detached` and can be used for staging changes.
    ///
    /// To read transaction properties from a detached manifest:
    /// ```ignore
    /// let detached = dataset.list_detached_manifests().await?;
    /// for location in detached {
    ///     let ds = dataset.checkout_version(location.version).await?;
    ///     let tx = ds.read_transaction().await?;
    ///     // Access tx.transaction_properties
    /// }
    /// ```
    pub async fn list_detached_manifests(&self) -> Result<Vec<ManifestLocation>> {
        self.table_state.list_detached_manifests().await
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        self.table_state.latest_version_id().await
    }

    /// Return whether the dataset has a newer committed version.
    pub async fn is_stale(&self) -> Result<bool> {
        self.table_view().is_stale().await
    }

    /// Return whether the immediate attached successor manifest exists.
    ///
    /// This is a fast contiguous-history probe. It does not resolve the latest
    /// version and may return `false` if intermediate manifests have been
    /// removed. Callers that need a general freshness check should use
    /// [`Self::is_stale`].
    #[doc(hidden)]
    pub async fn has_successor_version(&self) -> Result<bool> {
        self.table_view().has_successor_version().await
    }

    pub fn count_fragments(&self) -> usize {
        self.table_view().count_fragments()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        self.table_view().schema()
    }

    /// Similar to [Self::schema], but only returns fields that are not marked as blob columns
    /// Creates a new empty projection into the dataset schema
    pub fn empty_projection(self: &Arc<Self>) -> Projection {
        Projection::empty(self.clone())
    }

    /// Creates a projection that includes all columns in the dataset
    pub fn full_projection(self: &Arc<Self>) -> Projection {
        Projection::full(self.clone())
    }

    /// Get fragments.
    pub fn get_fragments(&self) -> Vec<FileFragment> {
        let dataset = Arc::new(self.clone());
        let table = Arc::new(self.table_view().to_table());
        self.iter_fragments()
            .map(|fragment| {
                FileFragment::from_table(dataset.clone(), table.clone(), fragment.clone())
            })
            .collect()
    }

    /// Iterate over manifest fragments without allocating [`FileFragment`] wrappers.
    pub fn iter_fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.table_view().iter_fragments()
    }

    pub fn get_fragment(&self, fragment_id: usize) -> Option<FileFragment> {
        let metadata = self.find_fragment(fragment_id as u64)?.clone();
        Some(FileFragment::new(Arc::new(self.clone()), metadata))
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        self.table_view().fragments()
    }

    /// Ids of the fragments in this version.
    pub fn fragment_bitmap(&self) -> &Arc<RoaringBitmap> {
        self.table_view().fragment_bitmap()
    }

    pub(crate) fn normalize_fragment_ids(fragment_ids: &[u32]) -> Vec<u32> {
        let mut ids = fragment_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub(crate) fn get_fragments_from_ids(&self, fragment_ids: &[u32]) -> Result<Vec<FileFragment>> {
        let ordered_ids = Self::normalize_fragment_ids(fragment_ids);
        let fragments = self.get_frags_from_ordered_ids(&ordered_ids);
        if let Some(missing_id) = fragments
            .iter()
            .zip(ordered_ids.iter())
            .find_map(|(fragment, fragment_id)| fragment.is_none().then_some(*fragment_id))
        {
            return Err(Error::invalid_input(format!(
                "Unknown fragment id {missing_id} in fragment filter; not part of the current dataset version"
            )));
        }

        Ok(fragments.into_iter().flatten().collect())
    }

    pub(crate) fn get_fragment_metadata_from_ids(
        &self,
        fragment_ids: &[u32],
    ) -> Result<Vec<Fragment>> {
        Ok(self
            .get_fragments_from_ids(fragment_ids)?
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect())
    }

    pub(crate) async fn count_rows_in_fragments(&self, fragment_ids: &[u32]) -> Result<usize> {
        let fragments = self.get_fragments_from_ids(fragment_ids)?;
        self.count_rows_in_resolved_fragments(fragments).await
    }

    async fn count_rows_in_resolved_fragments(
        &self,
        fragments: Vec<FileFragment>,
    ) -> Result<usize> {
        let counts = stream::iter(fragments)
            .map(|fragment| async move { fragment.count_rows(None).await })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(counts.iter().sum())
    }

    /// Resolves fragments for the given ids without scanning the manifest.
    ///
    /// The ids do not need to be sorted or deduplicated. Each id is resolved
    /// independently via the fragment bitmap.
    pub fn get_frags_from_ordered_ids(&self, ordered_ids: &[u32]) -> Vec<Option<FileFragment>> {
        let dataset = Arc::new(self.clone());
        let table = Arc::new(self.table_view().to_table());
        ordered_ids
            .iter()
            .map(|id| {
                if !self.table_state.fragment_bitmap.contains(*id) {
                    return None;
                }
                let fragment_index = self.table_state.fragment_bitmap.rank(*id) as usize - 1;
                let fragment = self.manifest.fragments.get(fragment_index)?;
                debug_assert_eq!(
                    fragment.id, *id as u64,
                    "fragment_bitmap rank({id}) resolved to fragment {}, but fragment_bitmap and manifest.fragments are expected to stay in sync",
                    fragment.id
                );
                Some(FileFragment::from_table(dataset.clone(), table.clone(), fragment.clone()))
            })
            .collect()
    }

    /// Look up the fragment with `id` in the manifest.
    ///
    /// `Manifest::fragments` is kept sorted by id, so this binary searches
    /// rather than scanning. Two kinds of manifest predate that invariant and
    /// are still readable: those written before fragments were forced into id
    /// order (Lance 0.10 and earlier), and those with duplicate fragment ids
    /// (Lance 0.16 and earlier). Neither is rejected on read, so the search
    /// result is checked and a scan takes over when it does not match --
    /// returning some other fragment's data would be silent corruption.
    pub(crate) fn find_fragment(&self, id: u64) -> Option<&Fragment> {
        self.table_view().find_fragment(id)
    }

    // This method filters deleted items from `addr_or_ids` using `addrs` as a reference

    /// Gets the number of files that are so small they don't even have a full
    /// group. These are considered too small because reading many of them is
    /// much less efficient than reading a single file because the separate files
    /// split up what would otherwise be single IO requests into multiple.
    pub async fn num_small_files(&self, max_rows_per_group: usize) -> usize {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.physical_rows().await })
            .buffered(self.table_state.object_store.io_parallelism())
            .try_filter(|row_count| futures::future::ready(*row_count < max_rows_per_group))
            .count()
            .await
    }

    pub async fn validate(&self) -> Result<()> {
        // All fragments have unique ids
        let id_counts =
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id)
                .fold(HashMap::new(), |mut acc, id| {
                    *acc.entry(id).or_insert(0) += 1;
                    acc
                });
        for (id, count) in id_counts {
            if count > 1 {
                return Err(Error::corrupt_file(
                    self.table_state.base.clone(),
                    format!(
                        "Duplicate fragment id {} found in dataset {:?}",
                        id, self.table_state.base
                    ),
                ));
            }
        }

        // Fragments are sorted in increasing fragment id order
        self.manifest
            .fragments
            .iter()
            .map(|f| f.id)
            .try_fold(0, |prev, id| {
                if id < prev {
                    Err(Error::corrupt_file(self.table_state.base.clone(), format!(
                        "Fragment ids are not sorted in increasing fragment-id order. Found {} after {} in dataset {:?}",
                        id, prev, self.table_state.base
                    )))
                } else {
                    Ok(id)
                }
            })?;

        // All fragments have equal lengths
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.validate().await })
            .buffer_unordered(self.table_state.object_store.io_parallelism())
            .try_collect::<Vec<()>>()
            .await?;

        rowids::validate_stable_row_ids(self).await?;

        // Validate indices. Over the complete list: these checks are about what
        // the manifest says, not about what this build can use, and duplicate
        // uuids or overlapping coverage are no less corrupt for involving an
        // index this build has no reader for. `migrate_indices` already runs the
        // same overlap check over the complete list on every commit.
        let indices = crate::index::loading::load_all_indices(self).await?;
        self.validate_indices(&indices)?;

        Ok(())
    }

    fn validate_indices(&self, indices: &[IndexMetadata]) -> Result<()> {
        // Make sure there are no duplicate ids
        let mut index_ids = HashSet::new();
        for index in indices.iter() {
            if !index_ids.insert(&index.uuid) {
                return Err(Error::corrupt_file(
                    self.table_state.manifest_location.path.clone(),
                    format!(
                        "Duplicate index id {} found in dataset {:?}",
                        index.uuid, self.table_state.base
                    ),
                ));
            }
        }

        // For each index name, make sure there is no overlap in fragment bitmaps
        if let Err(err) = detect_overlapping_fragments(indices) {
            let mut message = "Overlapping fragments detected in dataset.".to_string();
            for (index_name, overlapping_frags) in err.bad_indices {
                message.push_str(&format!(
                    "\nIndex {:?} has overlapping fragments: {:?}",
                    index_name, overlapping_frags
                ));
            }
            return Err(Error::corrupt_file(
                self.table_state.manifest_location.path.clone(),
                message,
            ));
        };

        Ok(())
    }
}

/// # Schema Evolution
///
/// Lance datasets support evolving the schema. Several operations are
/// supported that mirror common SQL operations:
///
/// - [Self::add_columns()]: Add new columns to the dataset, similar to `ALTER TABLE ADD COLUMN`.
/// - [Self::drop_columns()]: Drop columns from the dataset, similar to `ALTER TABLE DROP COLUMN`.
/// - [Self::alter_columns()]: Modify columns in the dataset, changing their name, type, or nullability.
///   Similar to `ALTER TABLE ALTER COLUMN`.
///
/// In addition, one operation is unique to Lance: [`merge`](Self::merge). This
/// operation allows inserting precomputed data into the dataset.
///
/// Because these operations change the schema of the dataset, they will conflict
/// with most other concurrent operations. Therefore, they should be performed
/// when no other write operations are being run.
impl Dataset {}

/// # Dataset metadata APIs
///
/// There are four kinds of metadata on datasets:
///
///  - **Schema metadata**: metadata about the data itself.
///  - **Field metadata**: metadata about the dataset itself.
///  - **Dataset metadata**: metadata about the dataset. For example, this could
///    store a created_at date.
///  - **Dataset config**: configuration values controlling how engines should
///    manage the dataset. This configures things like auto-cleanup.
///
/// You can get
impl Dataset {
    /// Get dataset metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.manifest.table_metadata
    }

    /// Get the dataset config from manifest
    pub fn config(&self) -> &HashMap<String, String> {
        &self.manifest.config
    }
}

#[async_trait::async_trait]
impl DatasetTakeRows for Dataset {
    fn schema(&self) -> &Schema {
        Self::schema(self)
    }

    async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        Self::take_rows(self, row_ids, projection.clone()).await
    }
}

impl Projectable for Dataset {
    fn schema(&self) -> &Schema {
        self.schema()
    }
}
