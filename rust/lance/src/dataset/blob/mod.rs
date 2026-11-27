// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(feature = "write")]
pub mod clone;

#[cfg(all(test, feature = "write"))]
mod compaction;

#[cfg(feature = "write")]
mod managed;

mod materialize;

#[cfg(feature = "write")]
use lance_table_runtime::blob::locate;
pub use lance_table_runtime::blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobRange, ReadBlobRangesBuilder,
    ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
};
#[cfg(feature = "write")]
pub use lance_table_runtime::blob::{
    BlobPayloadInventory, BlobPreprocessor, prepared_blob_batch_to_descriptors, prepared_blob_ids,
    prepared_sidecar_bytes,
};
#[cfg(feature = "write")]
pub(super) use lance_table_runtime::blob::{
    ExternalBaseResolver, SidecarRewrite, field_contains_blob, validate_external_blob_references,
};
#[cfg(feature = "write")]
pub(super) use locate::data_file_key_from_path;
#[cfg(feature = "write")]
pub use managed::preserve_managed_descriptors;
#[cfg(feature = "write")]
pub(super) use managed::{managed_paths, may_be_same_store};
pub use materialize::{
    BlobMaterializationAdmission, BlobMaterializationContext, MaterializedBlobBatch,
    blob_v2_descriptor_schema, materialize_blob_v2_binary_batch,
    materialize_blob_v2_binary_batch_with_admission, public_blob_v2_binary_output_schema,
    schema_has_blob_v2_binary_view, split_batch_by_bytes,
};

#[cfg(feature = "write")]
mod rewrite;

#[cfg(all(test, feature = "write"))]
mod rewrite_tests;

#[cfg(all(test, feature = "write"))]
pub mod test_util;

#[cfg(all(test, feature = "write"))]
mod write;

#[cfg(all(test, feature = "write"))]
mod file_tests;

#[cfg(all(test, feature = "write"))]
mod locate_tests;

#[cfg(all(test, feature = "write"))]
mod read_tests;
#[cfg(feature = "write")]
pub use rewrite::{BlobV2BatchRewritePlan, field_contains_blob_v2, transform_blob_v2_batch};
