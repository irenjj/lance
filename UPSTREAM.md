# Lance read-only migration spike

- `upstream_repo`: `https://github.com/lancedb/lance.git`
- `upstream_tag`: `v15.1.0-beta.2`
- `upstream_sha`: `fc8f59e53233a46f0d73604d66254142802389c9`
- `target_datafusion_path`: `/home/renjj/.paseo/worktrees/0k1e574a/df-lake-plan`
- `target_datafusion_sha`: `831a58bc1`

## Scope

This local branch records a migration spike. Layer 4 is an incomplete library
build. The DataFusion checkout is a read-only path dependency. All build output
went to the Lance target directory.

The workspace excludes `rust/lance-namespace`, `rust/lance-namespace-datafusion`,
`rust/lance-namespace-impls`, `rust/lance-geo`, `rust/lance-tools`,
`rust/lance-datagen`, and `rust/examples`. The excluded source remains
available for review. The workspace uses Arrow and Parquet 60.2.0,
object_store 0.14.2, and local DataFusion 57.0.0-dev sub-crates. The package
MSRV is 1.98.0. The root lockfile records the Rust spike's dependency graph.

The module cut in `rust/lance` detaches dataset `cleanup`, `compaction_stats`,
`delta`, `expire`, `mem_wal`, `optimize`, `overlay`, `schema_evolution`,
`sql`, `update_join*`, `updater`, `version_retention`, `write`, and `tests`,
as well as `io::commit` and index `append`, `create`, `frag_reuse`,
`frag_reuse_remapping`, `frag_reuse_v0_cache`, `frag_reuse_with_stable_row_ids`,
`lifecycle`, `mem_wal`, `optimize`, and `remap`. It also detaches
fragment `aligned_update` and `write`, along with the dataset `hash_joiner`.
Remaining references between read and write code are deliberately visible as
compiler errors. Shared writer internals in lower-level crates still compile.

## Local patches

- Layer 1 adapts Arrow schema metadata and list offsets. It removes the
  namespace-backed storage-options implementation while keeping the generic
  storage-options provider. It uses slice-relative indices throughout
  `deep_copy_array_data_sliced` to match Arrow 60's `MutableArrayData`.
- Layer 2 removes namespace-managed table opening and keeps direct URI opening
  and custom store parameters. It adapts Blob V2 metadata/offset boundaries
  and inverted-index posting metadata parsing.
- Layer 3 replaces the umbrella-crate session with modular function
  registration, an execution `TaskContext`, and caller-supplied
  `datafusion_session::Session` for provider execution. It adapts SQL
  LIKE/ILIKE escape parsing, literal metadata, and scalar UDF registration.
  Geo and Substrait are disabled. Helpers whose public types require core
  `DataFrame` are detached.
- Layer 4 rewires read-side imports to modular crates, changes
  `TableProvider::scan` projections to slices, and applies the module cut
  above. The scanner and provider implementations are still present, with
  read behavior pending end-to-end verification.

## First compiler diagnostics

Raw Cargo logs are under `/tmp/lance-df57-spike`. Paths and line numbers in
this table refer to the first build's source snapshot. Counts exclude warnings
and Cargo's final compilation summary. The first pass used Rust 1.99.1, which
was already installed. Final verification uses the upstream pin, Rust 1.98.1.

| Layer | First rustc errors | Main diagnostics and examples | Elapsed |
| --- | ---: | --- | ---: |
| 1 | 6 | 3 E0599 and 3 E0308. Offset accessors in `lance-arrow/src/list.rs:50,94,99,108`. Metadata in `lance-arrow/src/lib.rs:792,797`. Cargo initially rejected an inherited core dependency before invoking rustc, in 0.05 s. | 64.61 s |
| 2 | 162 | 133 E0433, 1 E0432, 18 E0425, 10 E0408. `lance-index` pulls in `lance-datafusion`, so the first errors were in the Layer 3 dependency, such as `lance-datafusion/src/logical_expr.rs:10,12,21` and `numeric_coercion.rs:128`. After Layer 3 resolved, Layer 2 itself surfaced 19 errors, 14 E0308, 3 E0283 and 2 E0599. See `lance-table-runtime/src/blob/write.rs:211,1210` and `lance-index/src/scalar/inverted/index/posting_list.rs:19`. | 28.57 s, then 21.92 s |
| 3 | 162 | Same dependency diagnostics as Layer 2. After switching imports and the context, 16 E0308 remained. See `lance-datafusion/src/planner.rs:1090` for SQL AST changes, `signed_zero.rs:393` for metadata, and `udf.rs:23` for UDF registration. | 20.37 s, then 1.61 s |
| 4 | 855 | 844 E0433 and 11 E0432 before the module cut. Core imports and namespace or execution helpers, for example `lance/src/dataset/fragment/aligned_update.rs:14`, `dataset/scanner/fts.rs:19`, and `dataset/update_join.rs:41`. The first cut produced 115 errors in 34.80 s. The following pass produced 100 errors across 34 files in 30.60 s. | 44.83 s |

## Final verification

| Layer | Locked check on Rust 1.98.1 | Source and manifest changes | Unit tests |
| --- | --- | --- | --- |
| 1 | All five libraries passed, 89.79 s | 13 files, +36 / -137 | `lance-arrow` 361 and `lance-core` 639 passed |
| 2 | All four libraries passed, 89.55 s | 15 files, +48 / -258 | `lance-index-core` 9 passed |
| 3 | Both libraries passed, 23.01 s | 21 files, +342 / -437 | `lance-datafusion-exec` 79 passed. `lance-datafusion --lib` reported 18 compile errors in the old session, DataFrame, and replay spill test harness. Two separate planner integration tests passed. |
| 4 | Failed with 99 diagnostics across 33 files, 42.40 s | 83 files, +596 / -1023 | Dataset read and Arrow ordering fixtures are blocked by the library build |

Counts come from the final diff against the Lance tag. The root manifest
(+40 / -47) and Rust lockfile (+293 / -2049) are shared. Cargo timings are wall
times and may include waits on the build directory lock.

The first Arrow unit test run that compiled passed 359 cases and failed two
raw Struct offset tests. Arrow 60's variable `start..end` in
`MutableArrayData::try_extend` is relative to the input slice, and the Struct
extender applies the parent offset itself. Using slice-relative indices fixed
both tests. The first execution-utilities run passed 78 cases and failed one
sort-spill fixture with a 64 KiB memory pool. A 128 KiB pool and independent
input buffers exercise real spills while checking ordering, full row
retention and spill metrics. The planner checks cover built-in
function registration, logical custom escape parsing and execution with the
backslash escape. DataFusion 57's physical planner rejects custom escapes
other than backslash.

The final `cargo tree --locked` check covered the 11 passing crates and
`lance --no-default-features`. It found Arrow Array 60.2.0 and object_store
0.14.2 where present. The leaf crate's dev dependency graph contains Parquet
60.2.0. Across all 12 trees there were zero Arrow Array v59, Parquet v59
or `datafusion` core nodes. The lockfile contains only Parquet 60.2.0 and
DataFusion sub-crates at 57.0.0-dev.

## Read-side dependencies exposed by the cut

| Read path | Boundary that needs to move |
| --- | --- |
| `index/loading.rs` `load_all_indices` | Instantiates `io::commit::LanceCommitHook` for legacy vector details. Separate the inference hook from lifecycle hooks so index descriptors and coverage remain available on old snapshots. |
| `index/loading.rs` `open_vector_index_from_metadata_with_plan`, `index/scalar.rs` `open_scalar_index_with_plan` | Both call `index/frag_reuse.rs` `open_row_id_remapping_with_plan`. The read side needs row-ID resolution, coverage and cache identity from the same area as staged remapping and compaction. |
| `dataset/scanner/source.rs` `overlay_stale_index_rows` and `overlay_stale_vector_rows`, `io/exec/ordered_index_scan.rs` `live_rows`, `dataset/statistics.rs` | Overlay visibility is part of the read contract. Separate visibility and stale-row replay from `dataset/overlay.rs` and its writer. |
| `io/exec/fts/compound.rs` `index_query_local_residual` | The compound FTS path builds a query-local residual index using `QueryLocalFtsIndex` and `QueryLocalFtsStats` from `dataset::mem_wal::index`. Extract this query-time use into the read side to keep fallback and BM25 scoring. |
| `dataset/fragment.rs` `FileFragment::open`, `open_session`, and `read_physical_slice` | The same wrapper contains methods using `FragmentCreateBuilder`, `Updater`, and `OverlayWriter`. The writer methods and Blob rewrite/cleanup helpers need a separate boundary from fragment and Blob reader construction. |
| `index/loading.rs` `DatasetIndexInternalExt`, `index/api.rs` `DatasetIndexExt` | Read extension currently inherits the mixed index API. Split read lookup and opening from create, append, optimize, and remap implementation. |

The projection, predicate and limit paths are retained in
`datafusion/dataframe.rs` and `datafusion/logical_plan.rs`. The FTS provider is
retained in `dataset/udtf.rs`. Their behavior requires a runnable Layer 4 and
local snapshot fixtures. In particular, verify Blob descriptions and index
search on snapshots with overlays, deletions, compacted fragments, and legacy
index metadata.

## Remaining effort

The 33 files in the final check's diagnostics include repeated references and
compiler cascades. They are a lower bound on the files involved. A planning
estimate for one engineer is 8–14 working days before reevaluation.

| Work | Days |
| --- | ---: |
| Finish splitting Dataset and Fragment write APIs, preserve direct open and Blob description/reads | 1–2 |
| Separate read-side index hooks, row-ID remapping, coverage and cache identity | 2–4 |
| Separate overlay visibility and FTS residual indexing, verify ANN/FTS/scalar planning | 2–3 |
| Complete DataFusion trait and metadata changes once imports resolve | 1–2 |
| Local compatibility fixtures and regression runs, including Arrow ordering | 2–3 |

Layers 1–3 justify reusing the modular build. For a short Lake integration
spike, defer a full Lance fork until the product requires the complete scanner
and index semantics and can own the corresponding fixture suite.
