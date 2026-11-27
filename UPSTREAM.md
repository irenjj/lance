# Lance read-only DataFusion 57 fork

- `upstream_repo` = `https://github.com/lancedb/lance.git`
- `upstream_tag` = `v15.1.0-beta.2`
- `upstream_sha` = `fc8f59e53233a46f0d73604d66254142802389c9`
- `spike_sha` = `c9e0dfd4335d2f00a5e874680e58bc67936a0af7`
- `target_datafusion_sha` = `0ddfccd0ee54565ad66484f1c9b86fdacbd790a8`

This branch opens Lance snapshots using the modular DataFusion 57 crates.
Cargo resolves Arrow and Parquet 60.2.0 and object_store 0.14.2. The lower
Lance layers retain file, manifest, and index construction primitives for
local fixtures. Dataset mutation and index maintenance APIs are outside the
compiled leaf crate. The workspace still excludes namespace, geo, tools,
datagen, and the upstream examples crate.

## Read-side boundaries

| Boundary | Active read path |
| --- | --- |
| Legacy vector metadata | `index/legacy_details.rs::LegacyVectorDetails` supplies inference to the runtime's `load_indices_with_legacy_details`. It retains the `lance.vector-details` cache namespace and retries partially inferred listings. |
| Fragment reuse | `index/frag_reuse_open.rs::open_row_id_remapping_with_plan` resolves committed segments using `frag_reuse_reader` and `frag_reuse_v0_cache`. V0 identities, per-segment translation fingerprints, coverage, and stable-row-ID visibility are preserved. The reader has a single query purpose. |
| Data overlays | `dataset/overlay.rs` exposes the table-layer staleness helpers used by scans, ordered scalar-index reads, and dataset statistics. |
| Compound FTS | `io/exec/fts/query_local.rs` owns residual term postings and BM25 corpus counts per query shard. Exact match and phrase leaves share the committed tokenizer and scorer. |
| Dataset and fragment | `DatasetBuilder`, version and fragment metadata, `FileFragment::open`, `open_session`, physical slices, Blob descriptors and range reads, and `take` remain available. Blob location and payload reads use the existing runtime primitives. |
| Index API | `index/api.rs::DatasetIndexExt`, `index/read.rs`, and `index/loading.rs::DatasetIndexInternalExt` cover listing, prewarm, statistics, and scalar, vector, and inverted reads. MemWAL opens return `NotSupported`. Overlapping legacy index coverage that requires repair also returns an error. |

The `lance` Cargo manifest explicitly selects `arrow_ordering` and
`readonly_baseline` as integration tests. Several upstream inline unit suites
use the detached high-level writers, index builders, or the excluded datagen
crate. Their source stays under unconditional false `cfg` guards where
practical for rebasing. The guards expose no Cargo feature or test target.

## Reproduce offline

Use Rust 1.98.1, a local Cargo dependency cache, and a DataFusion checkout at
`target_datafusion_sha`. Set `DF_ROOT` to that checkout's absolute path in
the invoking shell. The patch arguments belong on the command line.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
test "$(git -C "$DF_ROOT" rev-parse HEAD)" = 0ddfccd0ee54565ad66484f1c9b86fdacbd790a8 || exit 1
mapfile -t df_crates < <(sed -n 's/^name = "\(datafusion-[^"]*\)"$/\1/p' Cargo.lock)
patch=()
for name in "${df_crates[@]}"; do
    dir="$DF_ROOT/datafusion/${name#datafusion-}"
    test -f "$dir/Cargo.toml" || exit 1
    patch+=(--config "patch.crates-io.$name.path=\"$dir\"")
done

cargo fmt --all -- --check
cargo test --locked --offline -p lance-datafusion --lib --tests "${patch[@]}"
cargo clippy --locked --offline -p lance --no-default-features --lib --tests --no-deps "${patch[@]}" -- -D warnings
cargo test --locked --offline -p lance --no-default-features --lib --tests "${patch[@]}"
bash rust/lance/ci/check_readonly_tree.sh "${patch[@]}"
```

The Clippy invocation applies strict linting to the leaf library and its
test targets. Dependency linting in the same invocation trips existing
Arrow 60 deprecations in `lance-arrow/src/ipc.rs` and an explicit auto-deref
in `lance-arrow/src/schema.rs`, outside this change's file scope.

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
| Workspace formatting | Passed |
| `lance-datafusion --lib --tests` | 678 unit tests and 2 core-free planner tests passed |
| Strict `lance --lib --tests --no-deps` Clippy | Passed |
| `lance --no-default-features --lib --tests` | 1,087 unit tests, 46 Arrow ordering tests, and 8 read-only fixture cases passed |
| Dependency assertions | Passed |
| Doctests | 74 Lance and 5 lance-datafusion passed. 16 pre-existing ignored examples |

The tree assertion includes normal, build, and development edges. It checks
for zero DataFusion core packages, Arrow and Parquet exclusively at 60.2.0,
object_store exclusively at 0.14.2, and zero AWS SDK or excluded Lance
packages in this build.
