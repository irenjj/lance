// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// The streaming IVF training stack (see index::vector::ivf), awaited whole by
// its training tests, has a future type that nests past the default limit of
// 128. `recursion_limit` is a crate-level inner attribute, so it necessarily
// applies to every build of this crate, not only the test build that currently
// exercises that future. benches/streaming_ivf_training.rs raises the limit for
// the same stack.
#![recursion_limit = "256"]
// `redundant_clone` analyzes MIR for every function and costs ~35s on the test
// build of this crate, which is the critical path of a workspace clippy run.
// The non-test build still runs it, so library code stays covered.
#![cfg_attr(test, allow(clippy::redundant_clone))]

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.
//!
//!
//! # Open and scan a dataset
//!
//! ```rust
//! use lance::{Dataset, Result};
//!
//! async fn inspect(uri: &str) -> Result<()> {
//!     let dataset = Dataset::open(uri).await?;
//!     let mut scan = dataset.scan();
//!     scan.limit(Some(10), None)?;
//!     let batch = scan.try_into_batch().await?;
//!     assert!(batch.num_rows() <= 10);
//!     Ok(())
//! }
//! ```

use arrow_schema::DataType;
use dataset::builder::DatasetBuilder;
pub use lance_core::datatypes;
pub use lance_core::{Error, Result, cache, memory};
use std::sync::LazyLock;

pub mod arrow;
pub mod blob;
pub mod datafusion;
pub mod dataset;
pub mod index;
pub mod io;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod session;
pub mod table;
pub mod utils;

pub mod pb {
    #![allow(clippy::use_self)]
    include!("pb/lance.pb.rs");
}

/// Protobuf transport used by cache diagnostics language bindings.
///
/// These messages are runtime API payloads and are not part of the persisted
/// Lance file format.
pub mod cache_diagnostics_pb {
    include!("pb/lance.cache.rs");
}

pub use blob::{
    BlobArrayBuilder, BlobDescriptor, BlobDescriptorArrayBuilder, BlobDescriptorColumn,
    BlobFieldOptions, BlobRange, DedicatedBlobWriter, ManagedBlobWriter, PackedBlobWriter,
    blob_field, blob_field_with_options,
};
pub use dataset::Dataset;
use lance_index::vector::DIST_COL;

/// Creates and loads a [`Dataset`] from the given path.
/// Infers the storage backend to use from the scheme in the given table path.
///
/// For more advanced configurations use [`DatasetBuilder`].
pub async fn open_dataset<T: AsRef<str>>(table_uri: T) -> Result<Dataset> {
    DatasetBuilder::from_uri(table_uri.as_ref()).load().await
}

pub static DIST_FIELD: LazyLock<arrow_schema::Field> =
    LazyLock::new(|| arrow_schema::Field::new(DIST_COL, DataType::Float32, true));

/// Re-exports of 3rd party dependencies used in lance public APIs
///
/// Users that only use these dependencies for the sake of communicating with
/// Lance APIs can use these re-exports to ensure they are always pinned to the
/// same version that lance is using.
pub mod deps {
    pub use arrow_array;
    pub use arrow_schema;
}
