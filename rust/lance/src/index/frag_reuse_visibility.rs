// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Readers apply the fragment reuse index to every index they load, which would
//! rewrite stable row ids as if they were addresses. So on a stable-row-id table
//! it may only coexist with indices proven to store row addresses.

use lance_core::{Error, Result};
use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
use lance_index::registry::display_type_from_url;
use lance_table::format::{IndexMetadata, Manifest, pb as table_pb};
use lance_table::system_index::frag_reuse::is_frag_reuse_index_entry;
use prost::Name;
use prost_types::Any;

use super::scalar::IndexDetails;
use super::segment::{index_details_have_reader, index_is_usable, unsupported_index_version};

fn details_type_name(details: &Any) -> &str {
    details
        .type_url
        .rsplit_once('/')
        .map_or(details.type_url.as_str(), |(_, name)| name)
}

fn is_mem_wal_index_entry(index: &IndexMetadata) -> bool {
    index.name == MEM_WAL_INDEX_NAME
        && index.index_details.as_ref().is_some_and(|details| {
            details_type_name(details)
                .eq_ignore_ascii_case(&table_pb::MemWalIndexDetails::full_name())
        })
}

pub use lance_table_runtime::frag_reuse::has_frag_reuse_with_stable_row_ids;

/// `supports_batch_row_id_remapping` is only a proxy for applying the legacy
/// fragment reuse index on load, so every admitted type needs a query test on
/// that path.
fn address_index_incompatibility(index: &IndexMetadata) -> Option<String> {
    let Some(details) = index.index_details.as_ref() else {
        return Some("a legacy index without type details, which stores stable row IDs".into());
    };
    let type_name = details_type_name(details);
    let display_type = display_type_from_url(&details.type_url);
    if !index_details_have_reader(&details.type_url) {
        return Some(format!(
            "an index of type `{type_name}`, which is not known to this build of Lance"
        ));
    }
    if type_name.eq_ignore_ascii_case(&lance_index::pb::JsonIndexDetails::full_name()) {
        return Some("a JSON index, which does not apply the fragment reuse index".into());
    }
    if let Some(max_supported_version) = unsupported_index_version(index) {
        return Some(format!(
            "a {display_type} index at version {}, which is not known to this build of Lance \
             (it supports up to version {max_supported_version})",
            index.index_version
        ));
    }
    if !index.results_are_row_addrs() {
        return Some(format!(
            "a {display_type} index, which stores stable row IDs"
        ));
    }
    let applies_frag_reuse = IndexDetails(details.clone())
        .get_plugin()
        .is_ok_and(|plugin| plugin.supports_batch_row_id_remapping());
    if !applies_frag_reuse {
        return Some(format!(
            "a {display_type} index, which does not apply the fragment reuse index"
        ));
    }
    None
}

fn ordinary_index_incompatibility(index: &IndexMetadata) -> Option<String> {
    if is_frag_reuse_index_entry(index) || is_mem_wal_index_entry(index) {
        return None;
    }
    address_index_incompatibility(index)
}

fn incompatible_indices(indices: &[IndexMetadata]) -> Vec<(&IndexMetadata, String)> {
    indices
        .iter()
        .filter_map(|index| ordinary_index_incompatibility(index).map(|reason| (index, reason)))
        .collect()
}

/// Only meaningful where [`has_frag_reuse_with_stable_row_ids`] holds.
pub fn is_hidden_by_frag_reuse(index: &IndexMetadata) -> bool {
    ordinary_index_incompatibility(index).is_some()
}

pub fn warn_about_indices_hidden_by_frag_reuse(manifest: &Manifest, indices: &[IndexMetadata]) {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return;
    }
    for (index, reason) in incompatible_indices(indices) {
        // Unreadable indices are already reported by `warn_about_unsupported_indices`.
        if index_is_usable(index) {
            log::warn!(
                "Index {} is not used: the dataset uses stable row IDs and carries a fragment \
                 reuse index, and the index is {reason}",
                index.name,
            );
        }
    }
}

/// Backstop for callers that load an index from the full list, not the usable
/// view that already hides incompatible indices.
pub fn ensure_frag_reuse_applies(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    index: &IndexMetadata,
) -> Result<()> {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return Ok(());
    }
    match ordinary_index_incompatibility(index) {
        None => Ok(()),
        Some(reason) => Err(Error::invalid_input(format!(
            "Refusing to apply the fragment reuse index to index `{}` ({}): the dataset uses \
             stable row IDs and the index is {reason}",
            index.name, index.uuid
        ))),
    }
}
