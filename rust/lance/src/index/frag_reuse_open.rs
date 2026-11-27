// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::index::frag_reuse_reader::SegmentPlanParts;
use crate::index::frag_reuse_v0_cache::{
    V0HistoryDigests, V0HistoryDigestsKey, v0_segment_cache_id,
};
use crate::index::loading::DatasetIndexInternalExt;
use lance_core::Error;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, CompactFragReuseIndexHandle, FRAG_REUSE_INDEX_NAME,
    FragReuseIndexDetails,
};
use lance_index::scalar::{BatchRowIdRemapper, MetricsCollector, RowIdRemapper};
use lance_table::format::IndexMetadata;
use lance_table_runtime::frag_reuse as table_fri;

use roaring::RoaringBitmap;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;
use std::sync::Arc;
use uuid::Uuid;

/// The remapper resolved for one index open.
///
/// The FRI version picks the interface and the segment's need picks the
/// behavior: `V0` feeds the pre-existing synchronous consumers exactly as
/// before tagged histories existed; `V1Identity` carries no remapper at all
/// (the segment's rows are untouched, so the plugin's original load path
/// applies); `V1Translate` feeds the additive `*_with_remapping` entry points
/// that may await row-map reads.
#[derive(Clone)]
pub enum ResolvedRemapping {
    /// A v0 FRI mapping served by the compact in-memory handle.
    V0 {
        remapper: Arc<dyn RowIdRemapper>,
        /// See [`v0_segment_cache_id`].
        cache_id: Option<Uuid>,
    },
    /// A tagged history under which this segment's rows are unchanged.
    V1Identity,
    /// A tagged-history mapping whose payload may need asynchronous reads.
    V1Translate {
        remapper: Arc<dyn BatchRowIdRemapper>,
        /// The segment's translation identity (see
        /// [`super::frag_reuse_reader::FragmentReuseIndex::translation_fingerprint`]).
        fingerprint: [u8; 32],
    },
}

impl std::fmt::Debug for ResolvedRemapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V0 { cache_id, .. } => f
                .debug_struct("V0")
                .field("cache_id", cache_id)
                .finish_non_exhaustive(),
            Self::V1Identity => f.debug_tuple("V1Identity").finish(),
            Self::V1Translate { remapper, .. } => {
                f.debug_tuple("V1Translate").field(remapper).finish()
            }
        }
    }
}

pub(super) fn v0_cache_identity(dataset: &Dataset, entry: &IndexMetadata) -> Uuid {
    lance_table_runtime::frag_reuse::v0_cache_identity(&dataset.manifest, entry)
}

/// The FRI identity that belongs in an index's cache namespace, if any.
///
/// Only a v0 history is applied while the index is decoded (its remapper is
/// handed to the plugin's load path), so only then does the history, by
/// [`v0_segment_cache_id`], identify cached content. Under a tagged history the
/// FRI is deliberately left out: every rewrite and trim mints a new UUID, even
/// for transitions a segment never touches, and the translated namespace from
/// [`scoped_index_cache`] already carries the segment's own translation
/// identity.
pub(super) fn fri_cache_id(resolved: &Option<(Uuid, ResolvedRemapping)>) -> Option<&Uuid> {
    match resolved {
        Some((_, ResolvedRemapping::V0 { cache_id, .. })) => cache_id.as_ref(),
        _ => None,
    }
}

/// Scope the dataset-level index cache for one resolved remapping.
///
/// This owns the single cache-scoping rule, which classifies cached objects
/// by what they depend on rather than by snapshot:
///
/// * Content decoded without translation (an identity segment under a
///   tagged history, a v0 history, no history) depends only on the index
///   file, so it lives in the plain per-index namespace and an append or an
///   unrelated rewrite never cold-starts it.
/// * Content that embeds translated addresses (pages, postings, partitions
///   loaded through a `V1Translate` remapper) depends on the segment's
///   translation state, so it lives under a namespace named by the segment's
///   translation fingerprint. The entry goes cold exactly when that state
///   changes: a transition on the segment's path trimmed or replaced, a
///   fragment on its path dropped from the manifest, a sibling taking direct
///   ownership of one of its destinations. An append, a row-level delete, or
///   a rewrite of fragments the segment never covered leaves the fingerprint,
///   and the entry, in place.
///
/// Soundness rests on the fingerprint covering every input of
/// `remap_row_ids_excluding` (see `translation_fingerprint`); cached
/// translated objects hold the `FragmentReuseIndex` they were filled with, so
/// a stale one must never be served under a changed state.
pub(super) fn scoped_index_cache(
    dataset: &Dataset,
    resolved: &Option<(Uuid, ResolvedRemapping)>,
) -> crate::session::index_caches::DSIndexCache {
    crate::session::index_caches::DSIndexCache(match resolved {
        Some((_, ResolvedRemapping::V1Translate { fingerprint, .. })) => dataset
            .index_cache
            .with_key_prefix(&translated_namespace(fingerprint)),
        _ => dataset.index_cache.0.clone(),
    })
}

/// The cache namespace of translated content for one translation identity.
fn translated_namespace(fingerprint: &[u8; 32]) -> String {
    let mut prefix = String::with_capacity("fri-xlat/".len() + 2 * fingerprint.len());
    prefix.push_str("fri-xlat/");
    for byte in fingerprint {
        write!(prefix, "{byte:02x}").expect("writing to a String cannot fail");
    }
    prefix
}

/// The translation inputs one segment needs under a tagged history.
#[derive(Clone, Debug)]
enum SegmentRemappingPlan {
    /// The segment's stored coverage cannot intersect any rewritten path.
    Identity,
    /// The rewritten query coverage plus the fragments owned by other
    /// selected sibling segments of the same logical index, and the
    /// translation identity derived from them.
    Translate {
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
        fingerprint: [u8; 32],
    },
    /// Committed metadata exists but the reader derives no query coverage
    /// for this segment.
    MissingCoverage(MissingCoverageReason),
}

/// Why the reader derives no coverage for a registered segment. A direct
/// query open must refuse such segments because a scan was not planned for
/// their rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingCoverageReason {
    /// The stored bitmap is empty: every fragment the segment covered was
    /// withdrawn (an in-place rewrite), or it is a deferred definition.
    Withdrawn,
    /// The backtrack derives nothing from a non-empty bitmap: newer siblings
    /// own its direct coverage (superseded), or a destination lacks
    /// contributing sources.
    NoDerivedCoverage,
    /// This build cannot translate the segment: its type has no batch
    /// remapper, its coverage is unknown (no stored bitmap), or the history
    /// carries transitions this build cannot interpret.
    Unsupported,
    /// The stored metadata cannot be interpreted (no or undecodable index
    /// details).
    Corrupt,
    /// The segment's format marker could not be read, so whether this build
    /// can translate it is unknown. Retry the format probe on the next read.
    Unverifiable,
}

impl std::fmt::Display for MissingCoverageReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Withdrawn => "its coverage was withdrawn",
            Self::NoDerivedCoverage => {
                "the reader derives no coverage for it (superseded, or missing contributors)"
            }
            Self::Unsupported => "this build cannot translate it",
            Self::Corrupt => "its metadata cannot be interpreted",
            Self::Unverifiable => "its format marker could not be read",
        })
    }
}

/// The plan of a registered segment the snapshot plan does not know (a
/// stale cached plan): nothing is derived for it.
const PLAN_MISS: SegmentRemappingPlan =
    SegmentRemappingPlan::MissingCoverage(MissingCoverageReason::NoDerivedCoverage);

/// Snapshot-level plan of every committed segment's translation inputs.
///
/// Which rows a segment owns is decided once per manifest snapshot, from one
/// pass over the same `load_indices` output every per-open resolution used to
/// re-scan. Openers only look their segment up by UUID.
#[derive(Clone, Debug)]
struct FriQueryPlan {
    segments: HashMap<Uuid, SegmentRemappingPlan>,
}

impl DeepSizeOf for FriQueryPlan {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.segments
            .values()
            .map(|segment| match segment {
                SegmentRemappingPlan::Translate {
                    coverage,
                    excluded_fragments,
                    fingerprint,
                } => {
                    coverage.serialized_size()
                        + excluded_fragments.serialized_size()
                        + fingerprint.len()
                }
                _ => 0,
            })
            .sum::<usize>()
            + self.segments.len() * std::mem::size_of::<(Uuid, SegmentRemappingPlan)>()
    }
}

#[derive(Clone)]
struct FriQueryPlanKey<'a> {
    fri_uuid: &'a Uuid,
}

impl CacheKey for FriQueryPlanKey<'_> {
    type ValueType = FriQueryPlan;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.fri_uuid.to_string().into()
    }

    fn type_name() -> &'static str {
        "FriQueryPlan"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-query-plan", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.fri_uuid.as_bytes());
    }
}

/// Build or fetch the snapshot's FRI query plan.
///
/// Cached in the tagged (manifest-path scoped) namespace, so warm opens never
/// recompute coverage. A plan built from an unverifiable listing, or holding an
/// `Unverifiable` segment, is returned uncached, so concurrent cold builds of
/// such a snapshot each run the derivation -- the cost of re-probing instead of
/// freezing a transient fault for the snapshot. A healthy snapshot's cold
/// builds run the derivation concurrently too; the raw ledger, mapping and
/// format-marker readers they go through stay cached.
async fn fri_query_plan(
    dataset: &Dataset,
    fri: &IndexMetadata,
    stored: &[IndexMetadata],
    mapping: &Arc<super::frag_reuse_reader::FragmentReuseIndex>,
) -> lance_core::Result<Arc<FriQueryPlan>> {
    let cache = dataset
        .index_cache
        .with_key_prefix(dataset.table_state.manifest_location.path.as_ref());
    let key = FriQueryPlanKey {
        fri_uuid: &fri.uuid,
    };
    if let Some(cached) = cache.get_with_key(&key).await {
        return Ok(cached);
    }
    // The listing's own `unverifiable` set, not a guess from the plan's
    // contents: a segment the reader dropped for an unreadable marker is
    // simply absent here, and `missing_coverage_reason` below may then probe
    // it successfully and return a DEFINITIVE reason. Caching a plan built
    // from those two inconsistent views would freeze "no coverage" for the
    // whole snapshot.
    let mut unverifiable = BTreeSet::new();
    let plan = {
        let build = async {
            // The filtered listing carries the rewritten query coverage;
            // `stored` keeps provenance from before that rewrite. Callers
            // hold metadata returned by load_indices and cannot supply
            // this distinction.
            let listing = dataset.derived_index_listing().await?;
            unverifiable = listing.unverifiable;
            let indices = listing.indices;
            let stored_by_uuid: HashMap<Uuid, &IndexMetadata> =
                stored.iter().map(|entry| (entry.uuid, entry)).collect();
            // Group the filtered listing by logical index name; the
            // backtrack derives each member's sibling exclusions from the
            // stored provenance of the whole group in one pass, so
            // "direct coverage wins" is owned by one algorithm.
            let mut filtered_by_uuid: HashMap<Uuid, &IndexMetadata> =
                HashMap::with_capacity(indices.len());
            let mut groups: HashMap<&str, Vec<Uuid>> = HashMap::new();
            for entry in indices.iter() {
                filtered_by_uuid.insert(entry.uuid, entry);
                if entry.name != FRAG_REUSE_INDEX_NAME {
                    groups
                        .entry(entry.name.as_str())
                        .or_default()
                        .push(entry.uuid);
                }
            }
            let mut parts_by_uuid: HashMap<Uuid, SegmentPlanParts> = HashMap::new();
            for members in groups.into_values() {
                let provenance: Vec<RoaringBitmap> = members
                    .iter()
                    .map(|uuid| {
                        stored_by_uuid
                            .get(uuid)
                            .and_then(|source| source.fragment_bitmap.clone())
                            .unwrap_or_default()
                    })
                    .collect();
                for (uuid, parts) in members.iter().zip(mapping.segment_plans(&provenance)) {
                    parts_by_uuid.insert(*uuid, parts);
                }
            }
            let mut segments = HashMap::with_capacity(stored.len());
            for source in stored.iter() {
                let plan = if source
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.is_empty())
                {
                    // An empty bitmap needs no translation, but its pages
                    // may still hold the withdrawn rows: never identity.
                    SegmentRemappingPlan::MissingCoverage(MissingCoverageReason::Withdrawn)
                } else if !mapping.may_need_translation(source.fragment_bitmap.as_ref()) {
                    SegmentRemappingPlan::Identity
                } else if let Some(entry) = filtered_by_uuid.get(&source.uuid)
                    && let Some(bitmap) = &entry.fragment_bitmap
                {
                    let coverage = bitmap & dataset.table_state.fragment_bitmap.as_ref();
                    // Other selected segments own their direct coverage.
                    // Drop paths entering those fragments before later
                    // mappings can merge them with this segment's
                    // contribution.
                    let parts = parts_by_uuid.remove(&source.uuid).unwrap_or_default();
                    let fingerprint =
                        mapping.translation_fingerprint(&coverage, &parts.excluded, &parts.path);
                    SegmentRemappingPlan::Translate {
                        coverage,
                        excluded_fragments: parts.excluded,
                        fingerprint,
                    }
                } else if unverifiable.contains(&source.name) {
                    // This segment's own logical group lost a segment to an
                    // unreadable marker, so its absence here says nothing
                    // about its coverage. Re-probing could answer a definitive
                    // reason built on a group listing that is already known to
                    // be incomplete; report the uncertainty instead (the plan
                    // is uncached below). Another group's uncertainty leaves
                    // this segment's reason to the probe.
                    SegmentRemappingPlan::MissingCoverage(MissingCoverageReason::Unverifiable)
                } else {
                    SegmentRemappingPlan::MissingCoverage(
                        missing_coverage_reason(dataset, mapping, source).await?,
                    )
                };
                segments.insert(source.uuid, plan);
            }
            Ok::<_, Error>(FriQueryPlan { segments })
        };
        Arc::new(build.await?)
    };
    // A reason of `Unverifiable`, or a listing that lost a segment to an
    // unreadable marker, holds for this call alone: caching the plan would
    // keep the segment unverifiable for the whole snapshot. Return it
    // uncached so the next call re-probes.
    if unverifiable.is_empty()
        && !plan.segments.values().any(|plan| {
            matches!(
                plan,
                SegmentRemappingPlan::MissingCoverage(MissingCoverageReason::Unverifiable)
            )
        })
    {
        cache.insert_with_key(&key, plan.clone()).await;
    }
    Ok(plan)
}

/// Resolve the current fragment coverage for committed query segments.
///
/// Tagged histories derive coverage from [`FriQueryPlan`]; all other histories use the segment
/// metadata resolved for the query listing. Query callers must not substitute provenance for
/// derived coverage, and an unknown or unavailable plan is refused instead of treated as empty
/// coverage.
pub async fn resolve_query_segment_coverages(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> lance_core::Result<Vec<RoaringBitmap>> {
    let direct = |segment: &IndexMetadata| {
        segment
            .effective_fragment_bitmap(&dataset.table_state.fragment_bitmap)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Index '{}' segment {} is missing fragment coverage",
                    segment.name, segment.uuid
                ))
            })
    };

    let stored = super::loading::load_all_indices(dataset).await?;
    let Some(fri) = stored
        .iter()
        .find(|entry| entry.name == FRAG_REUSE_INDEX_NAME)
    else {
        return segments.iter().map(|segment| direct(segment)).collect();
    };
    if fri.index_version == 0 {
        return segments.iter().map(|segment| direct(segment)).collect();
    }
    if fri.index_version != 1 {
        return Err(Error::not_supported(format!(
            "FRI index_version {} is unsupported. Please upgrade to a newer version",
            fri.index_version
        )));
    }

    let mapping = super::frag_reuse_reader::FragmentReuseIndex::open(dataset, fri).await?;
    let plan = fri_query_plan(dataset, fri, &stored, &mapping).await?;
    let stored_by_uuid: HashMap<Uuid, &IndexMetadata> =
        stored.iter().map(|entry| (entry.uuid, entry)).collect();
    segments
        .iter()
        .map(|segment| match plan.segments.get(&segment.uuid) {
            Some(SegmentRemappingPlan::Identity) => {
                let stored = stored_by_uuid.get(&segment.uuid).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Index segment {} is not registered in the manifest",
                        segment.uuid
                    ))
                })?;
                direct(stored)
            }
            Some(SegmentRemappingPlan::Translate { coverage, .. }) => Ok(coverage.clone()),
            Some(SegmentRemappingPlan::MissingCoverage(reason)) => {
                Err(Error::not_supported(format!(
                    "FRI query coverage is unavailable for segment {} ({reason})",
                    segment.uuid
                )))
            }
            None if stored_by_uuid.contains_key(&segment.uuid) => {
                Err(Error::not_supported(format!(
                    "FRI query coverage is unavailable for segment {} (the reader derives no \
                     coverage for it)",
                    segment.uuid
                )))
            }
            None => Err(Error::invalid_input(format!(
                "Index segment {} is not registered in the manifest",
                segment.uuid
            ))),
        })
        .collect()
}

/// Why the reader derives no coverage for `source`, from what it can see of
/// the segment without opening its files.
async fn missing_coverage_reason(
    dataset: &Dataset,
    mapping: &super::frag_reuse_reader::FragmentReuseIndex,
    source: &IndexMetadata,
) -> lance_core::Result<MissingCoverageReason> {
    if mapping.has_unsupported_transitions() {
        return Ok(MissingCoverageReason::Unsupported);
    }
    match &source.fragment_bitmap {
        Some(bitmap) if bitmap.is_empty() => return Ok(MissingCoverageReason::Withdrawn),
        None => return Ok(MissingCoverageReason::Unsupported),
        Some(_) => {}
    }
    if source.index_details.is_none() {
        return Ok(MissingCoverageReason::Corrupt);
    }
    // A probe that cannot reach the format marker says nothing about the
    // segment's format, and the snapshot plan this reason goes into is built
    // for every segment: failing here would fail every query on the table.
    // Report it as unverifiable for THIS plan, so a later snapshot
    // reclassifies once the store answers again.
    Ok(
        match super::frag_reuse_reader::segment_supports_batch_remapping(dataset, source).await {
            Ok(Some(true)) => MissingCoverageReason::NoDerivedCoverage,
            Ok(Some(false)) => MissingCoverageReason::Unsupported,
            Ok(None) => MissingCoverageReason::Corrupt,
            Err(error) => {
                log::warn!(
                    "index {} (segment {}): could not read its format marker ({error}); \
                     reporting its coverage as unverifiable for this snapshot",
                    source.name,
                    source.uuid,
                );
                MissingCoverageReason::Unverifiable
            }
        },
    )
}

/// Resolve a committed segment against its snapshot's fragment reuse plan.
pub(super) async fn open_row_id_remapping_with_plan(
    dataset: &Dataset,
    index: &IndexMetadata,
    metrics: &dyn MetricsCollector,
) -> lance_core::Result<Option<(Uuid, ResolvedRemapping)>> {
    // The cheap cached stored listing decides the generation; the filtered
    // listing (whose tagged post-processing recomputes coverage) is only
    // consulted inside the once-per-snapshot plan build.
    let stored = super::loading::load_all_indices(dataset).await?;
    let Some(fri) = stored
        .iter()
        .find(|entry| entry.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(None);
    };
    super::frag_reuse_with_stable_row_ids::ensure_frag_reuse_applies(
        &dataset.manifest,
        &stored,
        index,
    )?;
    if fri.index_version == 0 {
        let Some(legacy) = dataset.open_frag_reuse_index(metrics).await? else {
            return Ok(None);
        };
        let history = v0_cache_identity(dataset, fri);
        let digests = dataset
            .index_cache
            .get_or_insert_with_key(V0HistoryDigestsKey { history: &history }, || async {
                Ok(V0HistoryDigests::new(&legacy.details))
            })
            .await?;
        let cache_id = v0_segment_cache_id(history, &legacy.details, &digests, index);
        return Ok(Some((
            history,
            ResolvedRemapping::V0 {
                remapper: Arc::new(CompactFragReuseIndexHandle(legacy)),
                cache_id,
            },
        )));
    }
    if fri.index_version != 1 {
        return Err(Error::not_supported(format!(
            "FRI index_version {} is unsupported. Please upgrade to a newer version",
            fri.index_version
        )));
    }
    // Everything below is v1-only code: legacy-only scopes must never get here.
    lance_index::scalar::check_batch_remapping_entry()?;
    let mapping = super::frag_reuse_reader::FragmentReuseIndex::open(dataset, fri).await?;
    let snapshot_plan = fri_query_plan(dataset, fri, &stored, &mapping).await?;
    let plan = match snapshot_plan.segments.get(&index.uuid) {
        Some(plan) => plan,
        None if stored.iter().any(|entry| entry.uuid == index.uuid) => &PLAN_MISS,
        None => {
            return Err(Error::not_supported(format!(
                "FRI remapping requires committed segment metadata for {}",
                index.uuid
            )));
        }
    };
    match plan {
        SegmentRemappingPlan::Identity => Ok(Some((fri.uuid, ResolvedRemapping::V1Identity))),
        SegmentRemappingPlan::MissingCoverage(reason) => Err(Error::not_supported(format!(
            "FRI query coverage is unavailable for segment {} ({reason}): the reader \
                 excludes it from the index listing on this snapshot; open it through the \
                 listing, not by uuid",
            index.uuid
        ))),
        SegmentRemappingPlan::Translate {
            coverage,
            excluded_fragments,
            fingerprint,
        } => Ok(Some((
            fri.uuid,
            ResolvedRemapping::V1Translate {
                remapper: Arc::new(super::frag_reuse_remapping::QueryRowIdRemapper::new(
                    mapping,
                    coverage.clone(),
                    excluded_fragments.clone(),
                )),
                fingerprint: *fingerprint,
            },
        ))),
    }
}

/// Load the version-zero fragment reuse history referenced by an index entry.
pub(super) async fn load_frag_reuse_index_details(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    table_fri::load_frag_reuse_index_details(&dataset.table_view().to_table(), index).await
}

/// open fragment reuse index based on its metadata details
pub(super) async fn open_frag_reuse_index(
    uuid: Uuid,
    details: &FragReuseIndexDetails,
) -> lance_core::Result<CompactFragReuseIndex> {
    CompactFragReuseIndex::try_new(uuid, details.clone())
}
