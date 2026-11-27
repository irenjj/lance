// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read-only secondary index APIs for Lance snapshots.
//!
//! Index algorithms and storage live in `lance-index`. This module resolves
//! snapshot coverage, opens physical segments, and reports index statistics.

#[cfg(feature = "write")]
use lance_index::IndexType;
use lance_index::scalar::registry::ScalarIndexPlugin;

mod api;

#[cfg(feature = "write")]
pub(crate) mod append;

#[cfg(feature = "write")]
mod create;

pub(crate) mod field_dependencies;
#[cfg(feature = "write")]
pub mod frag_reuse;
#[cfg(not(feature = "write"))]
#[path = "index/frag_reuse_open.rs"]
pub(crate) mod frag_reuse;
pub mod frag_reuse_reader;
mod frag_reuse_remapping;
mod frag_reuse_v0_cache;
#[cfg_attr(not(feature = "write"), path = "index/frag_reuse_visibility.rs")]
pub(crate) mod frag_reuse_with_stable_row_ids;
mod legacy_details;

#[cfg(feature = "write")]
mod lifecycle;

pub(crate) mod loading;

#[cfg(feature = "write")]
pub mod mem_wal;

#[cfg(feature = "write")]
pub(crate) mod memory_progress;

#[cfg(feature = "write")]
pub mod optimize;

pub mod prefilter;
mod prewarm;
#[cfg(not(feature = "write"))]
mod read;

#[cfg(feature = "write")]
pub(crate) mod provenance;

#[cfg(feature = "write")]
pub(crate) mod remap;

pub mod scalar;
pub(crate) mod scalar_logical;
pub(crate) mod segment;
pub(crate) mod segment_coverage;
mod statistics;
pub mod vector;
pub(crate) mod visibility;

pub use crate::index::api::{
    DatasetIndexExt, IndexSegment, IndexSegmentStatistics, IntoIndexSegment,
};
pub use crate::index::loading::DatasetIndexInternalExt;
pub use crate::index::prefilter::{FilterLoader, PreFilter};
pub use crate::index::vector::{LogicalIvfView, LogicalVectorIndex};
#[cfg(feature = "write")]
pub use create::CreateIndexBuilder;
pub use lance_index::IndexDescription;
pub use lance_index::IndexParams;
pub use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};

/// Register a statically linked scalar index plugin before using any datasets.
///
/// See [`lance_index::registry::register_scalar_index_plugin`] for the startup
/// registration and collision rules.
///
/// ```
/// # use lance_index::scalar::registry::ScalarIndexPlugin;
/// # fn register<D, P>() -> lance::Result<()>
/// # where D: prost::Message + prost::Name,
/// #       P: ScalarIndexPlugin + Default + 'static,
/// # {
/// lance::index::register_scalar_index_plugin::<D, P>()?;
/// # Ok(())
/// # }
/// ```
pub fn register_scalar_index_plugin<
    DetailsType: prost::Message + prost::Name,
    PluginType: ScalarIndexPlugin + Default + 'static,
>() -> crate::Result<()> {
    lance_index::registry::register_scalar_index_plugin_with_registry::<DetailsType, PluginType>(
        Some(|| &scalar::SCALAR_INDEX_PLUGIN_REGISTRY),
    )
}

#[cfg(feature = "write")]
/// Builds an index.
#[async_trait::async_trait]
pub trait IndexBuilder {
    /// Return the concrete index type.
    fn index_type() -> IndexType;

    /// Build the index.
    async fn build(&self) -> crate::Result<()>;
}

#[cfg(all(test, feature = "write"))]
mod tests;
