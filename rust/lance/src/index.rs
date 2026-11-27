// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read-only secondary index APIs for Lance snapshots.
//!
//! Index algorithms and storage live in `lance-index`. This module resolves
//! snapshot coverage, opens physical segments, and reports index statistics.

use lance_index::scalar::registry::ScalarIndexPlugin;

mod api;
pub(crate) mod field_dependencies;
#[path = "index/frag_reuse_open.rs"]
pub(crate) mod frag_reuse;
pub mod frag_reuse_reader;
mod frag_reuse_remapping;
mod frag_reuse_v0_cache;
#[path = "index/frag_reuse_visibility.rs"]
mod frag_reuse_with_stable_row_ids;
mod legacy_details;
pub(crate) mod loading;
pub mod prefilter;
mod prewarm;
mod read;
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
