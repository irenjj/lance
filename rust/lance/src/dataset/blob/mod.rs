// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod materialize;

pub use lance_table_runtime::blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobRange, ReadBlobRangesBuilder,
    ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
};
pub use materialize::{
    BlobMaterializationAdmission, BlobMaterializationContext, MaterializedBlobBatch,
    blob_v2_descriptor_schema, materialize_blob_v2_binary_batch,
    materialize_blob_v2_binary_batch_with_admission, public_blob_v2_binary_output_schema,
    schema_has_blob_v2_binary_view, split_batch_by_bytes,
};
