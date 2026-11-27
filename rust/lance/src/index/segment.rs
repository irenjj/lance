// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Validation and capability rules for physical index segments.
//!
//! These predicates describe the persisted `IndexMetadata` contract. They do
//! not execute index algorithms; the concrete readers and mergers live under
//! their index-type modules.

// These helpers stay crate-visible so callers do not depend on private parameter types.
#![allow(clippy::redundant_pub_crate)]

use lance_index::registry::{
    plugin_name_from_details_url, scalar_index_details_are_known, scalar_index_version,
};
use lance_index::{INDEX_FILE_NAME, IndexType, pb::VectorIndexDetails};
use lance_table::format::IndexMetadata;
use lance_table::system_index::is_system_index;
use prost::Name;

use crate::Result;
use crate::index::scalar::IndexDetails;

/// Ask the scalar index plugin whether the parameters of `incoming` segments
/// let them join `existing` ones (for example, MinHash signatures are only
/// comparable across segments built with the same hashing parameters).
/// Complements [`validate_segment_index_details`], which only checks that
/// segments share a details type. Vector indices and details without a plugin
/// are not checked.
pub(crate) fn validate_segment_params_compatible(
    existing: &[IndexMetadata],
    incoming: &[IndexMetadata],
) -> Result<()> {
    let Some(reference) = incoming
        .iter()
        .chain(existing.iter())
        .find_map(|segment| segment.index_details.as_ref())
    else {
        return Ok(());
    };
    let details = IndexDetails(reference.clone());
    let Ok(plugin) = details.get_plugin() else {
        return Ok(());
    };
    let existing_details: Vec<&prost_types::Any> = existing
        .iter()
        .filter_map(|segment| segment.index_details.as_deref())
        .collect();
    let incoming_details: Vec<&prost_types::Any> = incoming
        .iter()
        .filter_map(|segment| segment.index_details.as_deref())
        .collect();
    plugin.validate_new_segments_against_existing(&existing_details, &incoming_details)
}

/// Detect vector segments while preserving the legacy pre-details fallback.
///
/// Older vector segments may not have `VectorIndexDetails` in the manifest, so
/// we also recognize them by the legacy monolithic index file name.
pub(crate) fn segment_has_vector_details(segment: &IndexMetadata) -> bool {
    segment.index_details.as_ref().map_or_else(
        || {
            segment
                .files
                .as_ref()
                .is_some_and(|files| files.iter().any(|file| file.path == INDEX_FILE_NAME))
        },
        |details| details.type_url.ends_with("VectorIndexDetails"),
    )
}

/// Whether this build has a reader for the index's declared type.
///
/// Segments without details predate type URLs and remain readable through the
/// legacy file-based detection in the index open paths.
pub(crate) fn index_type_is_known(index: &IndexMetadata) -> bool {
    is_system_index(index)
        || index
            .index_details
            .as_ref()
            .is_none_or(|details| index_details_have_reader(&details.type_url))
}

/// Whether this build has a reader for the complete declared type name.
pub(crate) fn index_details_have_reader(type_url: &str) -> bool {
    let Some((_, details_type_name)) = type_url.rsplit_once('/') else {
        return false;
    };
    if details_type_name.eq_ignore_ascii_case(&VectorIndexDetails::full_name())
        // MemWAL flush used this exact pre-`pb` package name.
        || details_type_name.eq_ignore_ascii_case("lance.index.VectorIndexDetails")
    {
        return true;
    }
    if scalar_index_details_are_known(type_url) {
        return true;
    }
    false
}

/// `None` when this build supports the index's version, otherwise the highest
/// version it does support.
///
/// Only a version bump of a type this build already has a plugin for is caught.
/// Reader availability is checked separately by [`index_is_usable`], because
/// an unknown type has no meaningful maximum version in this build.
pub(crate) fn unsupported_index_version(index: &IndexMetadata) -> Option<u32> {
    let max_supported_version = index
        .index_details
        .as_ref()
        .map(|details| details_index_version(&details.type_url).unwrap_or(i32::MAX as u32))
        .unwrap_or_default();
    (index.index_version > max_supported_version as i32).then_some(max_supported_version)
}

fn details_index_version(type_url: &str) -> Option<u32> {
    // The details do not include the vector subtype, so use the highest
    // supported version of any vector index.
    if type_url.ends_with(VectorIndexDetails::NAME) {
        return Some(IndexType::max_vector_version());
    }
    let plugin_name = plugin_name_from_details_url(type_url);
    scalar_index_version(&plugin_name)
}

pub(crate) fn index_has_supported_field_dependencies(index: &IndexMetadata) -> bool {
    is_system_index(index)
        || index
            .fields
            .iter()
            .chain(&index.covering_fields)
            .all(|field_id| *field_id >= 0)
}

/// Whether this build may expose an index through the usable-index view.
///
/// System indices have dedicated readers rather than scalar plugins. Ordinary
/// indices need both a reader for their exact declared type and a supported
/// format version. Their field dependencies must belong to the user schema:
/// reserved negative IDs require dedicated support, not schema lookups.
pub(crate) fn index_is_usable(index: &IndexMetadata) -> bool {
    index_type_is_known(index)
        && unsupported_index_version(index).is_none()
        && index_has_supported_field_dependencies(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::scalar::SCALAR_INDEX_PLUGIN_REGISTRY;
    use lance_index::pb::RTreeIndexDetails;

    #[test]
    fn test_index_details_reader_matches_complete_type_name_case_insensitively() {
        for type_url in [
            "/lance.index.pb.VectorIndexDetails",
            "type.googleapis.com/LANCE.INDEX.PB.VECTORINDEXDETAILS",
            "type.googleapis.com/lance.index.VectorIndexDetails",
            "type.googleapis.com/LANCE.TABLE.BTREEINDEXDETAILS",
            "type.googleapis.com/lance.index.pb.BTreeIndexDetails",
        ] {
            assert!(
                index_details_have_reader(type_url),
                "expected a reader for {type_url}"
            );
        }

        for type_url in [
            "type.googleapis.com/example.MyVectorIndexDetails",
            "type.googleapis.com/example.BTreeIndexDetails",
            "VectorIndexDetails",
        ] {
            assert!(
                !index_details_have_reader(type_url),
                "unexpected reader for {type_url}"
            );
        }
    }

    #[test]
    fn test_rtree_manifest_metadata_matches_registry() {
        let details = prost_types::Any::from_msg(&RTreeIndexDetails::default()).unwrap();
        let registry = &*SCALAR_INDEX_PLUGIN_REGISTRY;
        assert!(!index_details_have_reader(&details.type_url));
        assert_eq!(
            details_index_version(&details.type_url),
            registry
                .get_plugin_by_details(&details)
                .ok()
                .map(|plugin| plugin.version()),
        );
    }
}
