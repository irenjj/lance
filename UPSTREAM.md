# Lance read-only DataFusion 57 fork

- `upstream_repo` = `https://github.com/lancedb/lance.git`
- `upstream_tag` = `v15.1.0-beta.2`
- `upstream_sha` = `fc8f59e53233a46f0d73604d66254142802389c9`
- `spike_sha` = `c9e0dfd4335d2f00a5e874680e58bc67936a0af7`
- `readonly_baseline_sha` = `644d2e0334a0bc3c005aa68f90df31efc9af10a3`
- `target_datafusion_sha` = `9f21b10e76b13f35c379376d09d51fbc5e8581f3`

This branch opens Lance snapshots using the modular DataFusion 57 crates.
Cargo resolves Arrow and Parquet 60.2.0 and object_store 0.14.2. The lower
Lance layers retain file, manifest, and index construction primitives for
local fixtures. The workspace excludes namespace, geo, tools, datagen, and
the upstream examples crate.

## Write-side source retention

Dataset mutation, index building and maintenance, and their writer-backed
tests are restored at their upstream locations under
`#[cfg(feature = "write")]` and `#[cfg(all(test, feature = "write"))]`. The
`write` feature defaults to off in `lance`, `lance-datafusion`, `lance-table-runtime`, and
`lance-io`. The Lance manifest gates upstream writer-dependent targets
with `required-features`. The only integration tests selected by the
verified read-only build are `arrow_ordering` and `readonly_baseline`.

The feature preserves source for rebasing. Building the write configuration
still requires porting upstream DataFusion core APIs and namespace
integrations, and adapting writer tests that use the excluded datagen
crate. Supported verification uses the modular DataFusion read-only build.

| Diff against `upstream_sha` | Before source restoration | After source restoration |
| --- | --- | --- |
| Including `Cargo.lock` | 177 files, +4,690 / −20,384 | 177 files, +6,043 / −4,038 |
| Excluding `Cargo.lock` | 176 files, +4,395 / −18,265 | 176 files, +5,748 / −1,919 |
| Deleted files | 0 | 0 |

`Cargo.lock` already resolves with the 24 patches at the target DataFusion
revision. The Cargo manifests, lockfile, and Rust source contain zero
hard-coded local checkout paths from the migration.

## Read-side boundaries

| Boundary | Active read path |
| --- | --- |
| Legacy vector metadata | `index/legacy_details.rs::LegacyVectorDetails` supplies inference to the runtime's `load_indices_with_legacy_details`. It retains the `lance.vector-details` cache namespace and retries partially inferred listings. |
| Fragment reuse | `index/frag_reuse_open.rs::open_row_id_remapping_with_plan` resolves committed segments using `frag_reuse_reader` and `frag_reuse_v0_cache`. V0 identities, per-segment translation fingerprints, coverage, and stable-row-ID visibility are preserved. The reader has a single query purpose. |
| Data overlays | `dataset/overlay.rs` exposes the table-layer staleness helpers used by scans, ordered scalar-index reads, and dataset statistics. |
| Compound FTS | `io/exec/fts/query_local.rs` owns residual term postings and BM25 corpus counts per query shard. Exact match and phrase leaves share the committed tokenizer and scorer. |
| Dataset and fragment | `DatasetBuilder`, version and fragment metadata, `FileFragment::open`, `open_session`, physical slices, Blob descriptors and range reads, and `take` remain available. Blob location and payload reads use the existing runtime primitives. |
| Index API | `index/api.rs::DatasetIndexExt`, `index/read.rs`, and `index/loading.rs::DatasetIndexInternalExt` cover listing, prewarm, statistics, and scalar, vector, and inverted reads. MemWAL opens return `NotSupported`. Overlapping legacy index coverage that requires repair also returns an error. |

## Reproduce offline

Use Rust 1.98.1 and a local Cargo dependency cache. Set `DF_ROOT` to the
absolute path of a clean DataFusion checkout at `target_datafusion_sha`.
Run the following from a clean Lance checkout to bind the evidence to its
current commit. Choose an empty `EVIDENCE_DIR` for the run.

```bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=8 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
df_revision=9f21b10e76b13f35c379376d09d51fbc5e8581f3
test "$(git -C "$DF_ROOT" rev-parse HEAD)" = "$df_revision"
test -z "$(git -C "$DF_ROOT" status --porcelain)"
mapfile -t df_crates < <(sed -n 's/^name = "\(datafusion-[^"]*\)"$/\1/p' Cargo.lock)
test "${#df_crates[@]}" -eq 24
patch=()
for name in "${df_crates[@]}"; do
    dir="$DF_ROOT/datafusion/${name#datafusion-}"
    test -f "$dir/Cargo.toml"
    patch+=(--config "patch.crates-io.$name.path=\"$dir\"")
done

evidence_dir="${EVIDENCE_DIR:-/tmp/df-lake-evidence/b-p0b-optb}"
record() {
    local lane=$1
    shift
    bash "$DF_ROOT/scripts/evidence.sh" --name "$lane" --output-dir "$evidence_dir" \
        --notes "DataFusion $df_revision." -- "$@"
}

record lance-fmt cargo fmt --all -- --check
record lance-datafusion-test cargo test --locked --offline -p lance-datafusion --lib --tests "${patch[@]}"
record lance-clippy cargo clippy --locked --offline -p lance --no-default-features --lib --tests --no-deps "${patch[@]}" -- -D warnings
record lance-readonly-test cargo test --locked --offline -p lance --no-default-features --lib --tests "${patch[@]}"
record lance-tree-check bash rust/lance/ci/check_readonly_tree.sh "${patch[@]}"

revision=$(git rev-parse HEAD)
(
    cd "$evidence_dir"
    for manifest in *.json; do
        jq -e --arg revision "$revision" \
            '.schema == "df-evidence-v1" and .source_revision == $revision and .source_dirty == false and .exit_code == 0' \
            "$manifest" >/dev/null
        jq -r '.artifacts[] | "\(.sha256)  \(.path)"' "$manifest"
    done | sha256sum --check --strict
    sha256sum -- *.json > manifests.sha256
    sha256sum --check --strict manifests.sha256
)
```

The Clippy invocation applies strict linting to the leaf library and its
test targets. Dependency diagnostics can include existing deprecation and
dead-code warnings outside the leaf crate.

## Local fixtures and verification

- Programmatic fixtures generate two local fragments with a fixed vector seed,
  an IVF_FLAT index, a bitmap index, and a partially indexed text column.
  Assertions cover direct open, schema and fragment metadata, physical
  slicing, scanner and modular provider pushdown, indexed ANN with scalar
  prefilter and take, BM25, and compound FTS residuals. The Blob V2 fixture
  reads descriptors with its payload offline, then verifies a bounded range.
- Historical coverage combines compacted snapshots with v0 and v1 fragment
  reuse, an external deletion vector, and overlay replay. Checked-in
  snapshots exercise legacy deletion metadata, Blob V1 reads, and vector
  detail inference.
  The Arrow ordering test pins the read-layer ordering contract.

| Gate | Result |
| --- | --- |
| `lance-fmt` | Passed |
| `lance-datafusion-test` | 678 unit tests and 2 core-free planner tests passed |
| `lance-clippy` | Strict leaf and test linting passed |
| `lance-readonly-test` | 1,087 unit tests, 46 Arrow ordering tests, and 8 read-only fixture cases passed |
| `lance-tree-check` | Passed for normal, build, and dev dependencies |

The tree assertion includes normal, build, and development edges. It checks
for zero DataFusion core packages, Arrow and Parquet exclusively at 60.2.0,
object_store exclusively at 0.14.2, and zero AWS SDK or excluded Lance
packages in this build.
