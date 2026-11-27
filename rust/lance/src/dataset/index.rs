// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{Dataset, Result};

#[async_trait]
pub trait LanceIndexStoreExt {
    /// Open an index store for an existing index (might be relative or absolute)
    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self>
    where
        Self: Sized;
}

/// Select the exact file version used for index files in this dataset version.
///
/// Index files should never use the legacy format. If the dataset uses legacy
/// format, V2_0 is selected explicitly by the dataset composition table.
pub(crate) fn dataset_format_version(dataset: &Dataset) -> ConcreteFileVersion {
    match dataset.manifest.data_storage_format.lance_file_format() {
        ConcreteFileVersion::V1 => ConcreteFileVersion::V2_0,
        version => version,
    }
}

#[async_trait]
impl LanceIndexStoreExt for LanceIndexStore {
    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self> {
        let index_dir = dataset
            .indice_files_dir(index)?
            .join(index.uuid.to_string());
        let cache = dataset
            .table_state
            .metadata_cache
            .file_metadata_cache(&index_dir);
        let format_version = dataset_format_version(dataset);
        let object_store = dataset.object_store_for_index(index).await?;
        let store =
            Self::with_format_version(object_store, index_dir, Arc::new(cache), format_version);
        Ok(store.with_file_sizes(index.file_size_map()))
    }
}
