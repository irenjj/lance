// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_execution::SendableRecordBatchStream;
use futures::FutureExt;
use itertools::Itertools;
use lance_index::{
    FtsPrewarmResult, IndexCriteria, IndexDescription, PrewarmOptions,
    frag_reuse::FRAG_REUSE_INDEX_NAME, is_system_index, mem_wal::MEM_WAL_INDEX_NAME,
};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::index::api::{DatasetIndexExt, IndexSegmentStatistics};
use crate::index::field_dependencies::described_field_ids;
use crate::index::frag_reuse_reader;
use crate::index::frag_reuse_with_stable_row_ids::{
    has_frag_reuse_with_stable_row_ids, is_hidden_by_frag_reuse,
};
use crate::index::loading::{DatasetIndexInternalExt, load_all_indices};
use crate::index::prewarm::{filter_index_segments_by_ids, prewarm_index_segments_by_metadata};
use crate::index::scalar::{IndexDetails, index_matches_criteria};
use crate::index::segment::index_is_usable;
use crate::index::statistics::{
    IndexDescriptionImpl, collect_index_segment_statistics, index_statistics_frag_reuse,
    index_statistics_scalar, serialize_regular_index_statistics,
};
use crate::index::vector::details::needs_vector_details_inference;
use crate::session::index_caches::{
    DerivedIndexListingKey, QueryIndexListing, QueryIndexListingKey,
};
use crate::{Dataset, Error, Result};

impl Dataset {
    pub(crate) async fn derived_index_listing(&self) -> Result<DerivedIndexListing> {
        let indices = load_all_indices(self).await?;
        self.derive_index_listing(indices).await
    }

    /// [`Self::derived_index_listing`] for a raw listing the caller already
    /// loaded, so a caller that must inspect the raw listing first does not
    /// load it twice.
    async fn derive_index_listing(
        &self,
        indices: Arc<Vec<IndexMetadata>>,
    ) -> Result<DerivedIndexListing> {
        // Readers apply the fragment reuse index to every index they open.
        let indices = if has_frag_reuse_with_stable_row_ids(&self.manifest, &indices)
            && indices.iter().any(is_hidden_by_frag_reuse)
        {
            Arc::new(
                indices
                    .iter()
                    .filter(|idx| !is_hidden_by_frag_reuse(idx))
                    .cloned()
                    .collect(),
            )
        } else {
            indices
        };
        let tagged = indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .map(|fri| fri.index_version);
        match tagged {
            // Legacy FRI index version 0 already had its fragment coverage
            // remapped in load_all_indices().
            None | Some(0) => Ok(DerivedIndexListing {
                indices: filter_unusable(indices),
                unverifiable: BTreeSet::new(),
            }),
            Some(1) => {
                let fri = indices
                    .iter()
                    .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                    .expect("the v1 FRI entry was just located");
                let (derived, unverifiable) =
                    frag_reuse_reader::load_indices(self, fri, &indices).await?;
                Ok(DerivedIndexListing {
                    indices: derived,
                    unverifiable,
                })
            }
            Some(version) => Err(Error::not_supported(format!(
                "FRI index_version {version} is unsupported. Please upgrade to a newer version",
            ))),
        }
    }
}

/// The index listing a snapshot's readers see, with what the caller needs to
/// decide whether to cache it.
pub struct DerivedIndexListing {
    pub indices: Arc<Vec<IndexMetadata>>,
    /// Logical indices with at least one segment whose format marker could not
    /// be read; empty when every segment was verified. Those indices' listings
    /// hold for this call alone.
    pub unverifiable: BTreeSet<String>,
}

/// Drop the segments no reader can open, keeping the input allocation when
/// every segment is usable.
fn filter_unusable(indices: Arc<Vec<IndexMetadata>>) -> Arc<Vec<IndexMetadata>> {
    if indices.iter().all(index_is_usable) {
        return indices;
    }
    Arc::new(
        indices
            .iter()
            .filter(|idx| index_is_usable(idx))
            .cloned()
            .collect(),
    )
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    async fn prewarm_index(&self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(self, name, indices, None, available_segment_count, None)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options(&self, name: &str, options: &PrewarmOptions) -> Result<()> {
        self.prewarm_index_with_options_result(name, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options_result(
        &self,
        name: &str,
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            None,
        )
        .await
    }

    async fn prewarm_index_segments(&self, name: &str, segment_ids: &[Uuid]) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            None,
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
        .map(|_| ())
    }

    async fn prewarm_index_segments_with_options(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<()> {
        self.prewarm_index_segments_with_options_result(name, segment_ids, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_segments_with_options_result(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
    }

    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn IndexDescription>>> {
        let indices = self.load_indices().await?;
        let mut indices = if let Some(criteria) = criteria {
            indices.iter().filter(|idx| {
                if idx.index_details.is_none() {
                    log::warn!("The method describe_indices does not support indexes without index details.  Please retrain the index {}", idx.name);
                    return false;
                }
                // Only query inputs. `index_matches_criteria` compares this
                // slice against the single column named by `for_column`, so
                // passing the carried columns too would push its length past one
                // and silently drop every covered index from a filtered
                // describe.
                let fields = described_field_ids(idx)
                    .iter()
                    .filter_map(|id| self.schema().field_by_id(*id))
                    .collect::<Vec<_>>();
                match index_matches_criteria(idx, &criteria, &fields, false, self.schema()) {
                    Ok(matched) => matched,
                    Err(err) => {
                        log::warn!("Could not describe index {}: {}", idx.name, err);
                        false
                    }
                }
            }).collect::<Vec<_>>()
        } else {
            indices.iter().collect::<Vec<_>>()
        };
        indices.sort_by_key(|idx| &idx.name);

        let grouped: Vec<Vec<IndexMetadata>> = indices
            .into_iter()
            .chunk_by(|idx| idx.name.clone())
            .into_iter()
            .map(|(_, segments)| segments.cloned().collect::<Vec<_>>())
            .collect();

        let mut results = Vec::with_capacity(grouped.len());
        for segments in grouped {
            let desc = IndexDescriptionImpl::try_new(segments, self).await?;
            results.push(Arc::new(desc) as Arc<dyn IndexDescription>);
        }
        Ok(results)
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let derive = || async {
            let indices = load_all_indices(self).await?;
            // Best-effort inference and unreadable format markers must be
            // retried on later calls instead of freezing an incomplete listing.
            let retry_inference = indices
                .iter()
                .any(|idx| needs_vector_details_inference(idx, self.schema()));
            let listing = self.derive_index_listing(indices).await?;
            Ok(QueryIndexListing {
                indices: listing.indices,
                retry_derivation: retry_inference || !listing.unverifiable.is_empty(),
            })
        };
        // The key's e-tag is its generation token. Without one, a dataset
        // dropped and recreated at this URI restarts its version history, so a
        // version-only key could serve the previous incarnation's listing from
        // this cache, which is shared by every dataset opened at the URI.
        // `None` is not proof that two observations share an incarnation (see
        // `ManifestLocation::e_tag`), so derive privately and never publish.
        let Some(e_tag) = self.table_state.manifest_location.e_tag.as_deref() else {
            return Ok(derive().await?.indices);
        };
        let derived_key = QueryIndexListingKey(DerivedIndexListingKey {
            version: self.manifest.version,
            store_identity: &self.table_state.object_store.store_prefix,
            e_tag: Some(e_tag),
        });
        let (listing, was_cached) = self
            .index_cache
            .get_or_insert_with_key_hit(derived_key, derive)
            .await?;
        if was_cached && listing.retry_derivation {
            let refreshed = Arc::new(derive().await?);
            self.index_cache
                .insert_with_key(&derived_key, refreshed.clone())
                .await;
            return Ok(refreshed.indices.clone());
        }
        Ok(listing.indices.clone())
    }

    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices().await?;

        let mut indices = indices
            .iter()
            .filter(|idx| {
                // System indices are fieldless by design. Report malformed user indices,
                // but don't fail the operation (we might not be using that index).
                if idx.fields.is_empty() {
                    if !is_system_index(idx) {
                        log::error!("Index {} has no fields", idx.name);
                    }
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        // This sorting & chunking is only needed to calculate if there are multiple indexes on the same
        // field.  This fact is only needed for backwards compatibility behavior for indexes that don't have
        // index details.  At some point we should deprecate indexes without index details.
        //
        // TODO: At some point we should just fail if the index details are missing and ask the user to
        // retrain the index.
        indices.sort_by_key(|idx| idx.fields[0]);
        let indice_by_field = indices.into_iter().chunk_by(|idx| idx.fields[0]);
        for (field_id, indices) in &indice_by_field {
            let indices = indices.collect::<Vec<_>>();
            let has_multiple = indices.len() > 1;
            for idx in indices {
                let field = self.schema().field_by_id(field_id);
                if let Some(field) = field
                    && index_matches_criteria(
                        idx,
                        &criteria,
                        &[field],
                        has_multiple,
                        self.schema(),
                    )?
                {
                    let non_empty = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                        bitmap.intersection_len(self.table_state.fragment_bitmap.as_ref()) > 0
                    });
                    let is_fts_index = if let Some(details) = &idx.index_details {
                        IndexDetails(details.clone()).supports_fts()
                    } else {
                        false
                    };
                    // FTS indices must always be returned even if empty, because FTS queries
                    // require an index to exist. The query execution will handle the empty
                    // bitmap appropriately and fall back to scanning unindexed data.
                    // Other index types can be skipped if empty since they're optional optimizations.
                    if non_empty || is_fts_index {
                        return Ok(Some(idx.clone()));
                    }
                }
            }
        }
        return Ok(None);
    }

    async fn index_statistics(&self, index_name: &str) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }

        if index_name == FRAG_REUSE_INDEX_NAME {
            return index_statistics_frag_reuse(self).boxed().await;
        }

        if index_name == MEM_WAL_INDEX_NAME {
            return Err(Error::not_supported(
                "MemWAL indices are unavailable in this read-only build",
            ));
        }

        index_statistics_scalar(self, index_name, metadatas)
            .boxed()
            .await
    }

    async fn index_segment_statistics(
        &self,
        index_name: &str,
        index_uuids: &[Uuid],
    ) -> Result<Vec<IndexSegmentStatistics>> {
        if index_uuids.is_empty() {
            return Err(Error::invalid_input(
                "At least one index segment UUID is required".to_string(),
            ));
        }
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.iter().any(is_system_index) {
            return Err(Error::invalid_input(
                "Segment statistics are not supported for system indexes".to_string(),
            ));
        }
        let metadata_by_uuid: HashMap<_, _> = metadatas
            .iter()
            .map(|metadata| (metadata.uuid, metadata))
            .collect();
        let mut seen_uuids = HashSet::with_capacity(index_uuids.len());
        let mut statistics = Vec::with_capacity(index_uuids.len());
        for index_uuid in index_uuids {
            if !seen_uuids.insert(*index_uuid) {
                return Err(Error::invalid_input(format!(
                    "Duplicate index segment UUID: {index_uuid}"
                )));
            }
            let metadata = metadata_by_uuid.get(index_uuid).ok_or_else(|| {
                Error::index_not_found(format!("name={index_name}, uuid={index_uuid}"))
            })?;
            let field_id = metadata.fields.first().ok_or_else(|| {
                Error::index(format!("Index segment {index_uuid} has no indexed field"))
            })?;
            let field_path = self.schema().field_path(*field_id)?;
            statistics.push(collect_index_segment_statistics(self, metadata, &field_path).await?);
        }
        Ok(statistics)
    }

    async fn index_statistics_from_segments(
        &self,
        index_name: &str,
        segment_statistics: Vec<IndexSegmentStatistics>,
    ) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.iter().any(is_system_index) {
            return Err(Error::invalid_input(
                "Segment statistics are not supported for system indexes".to_string(),
            ));
        }
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={index_name}")));
        }
        serialize_regular_index_statistics(self, index_name, &metadatas, segment_statistics)
            .await?
            .ok_or_else(|| {
                Error::index(format!(
                    "Outdated fragment metadata for index {index_name}: collect index statistics \
                     in non-distributed mode to migrate before retrying"
                ))
            })
    }

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let indices = self.load_indices_by_name(index_name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }
        let field_id = indices[0].fields[0];
        let field_path = self.schema().field_path(field_id)?;
        let logical_index = self
            .open_logical_vector_index(&field_path, index_name)
            .await?;
        logical_index
            .as_ivf()?
            .read_partition(partition_id, with_vector)
            .await
    }
}
