# Archived benchmark artifacts

This directory is the GPU-free entry point.

- `archived/json_logs/` contains self-contained JSON logs for all six datasets. These logs include environment records, labels, canonical embeddings, metric values, and repeated-run aggregate metrics. Each aggregate records its own `n`, so repeated scalar metrics and more expensive topology metrics can be audited separately.
- `archived/embedding_pngs/` contains all canonical embedding PNGs for the archived run: 6 datasets times 5 methods.
- `label_metadata.json` maps raw benchmark labels to legend text. MNIST labels are digit identities; Levine labels use the upstream population-name files from the Levine 13- and 32-dimensional benchmark-data repositories.

Run the local pipeline from these files with:

```bash
scripts/reproduce_archived.sh --check-only
scripts/reproduce_archived.sh
```

The check-only command validates exact file membership, real JSON/PNG content,
minimum payload sizes, the required committed Revision 3 publication files,
and Git tracking without creating an environment or using the network. No
Lambda.ai access, GPU, CUDA, RAPIDS, PyTorch, or Docker is required for this
archived-data path. A normal Python environment with NumPy/Matplotlib plus a
TeX installation is enough to rebuild figures and the manuscript PDF.

## Revision 3 large-data profile

The million-point profile deliberately keeps large public downloads and
high-dimensional prepared arrays outside Git:

- `revision3/raw/`: the pinned 10x archive and extracted source members;
- `revision3/prepared/`: common float32 inputs, released reference labels, fixed
  subset orders, and dataset manifests;
- `revision3/large_results/`: isolated same-H100 method/size records and full
  layouts for the declared \(d=2\) benchmark profile, including separately
  identified production-auto and forced-IVF-Flat DiRe control runs. DiRe
  remains the general dimensionality-reduction framework described in the
  pre-Revision-3 manuscript and accepts a user-specified target dimension,
  including non-visual dimensions used in subsequent analysis;
- `revision3/backend_policy_results/`: a fresh, isolated production-auto versus
  forced-IVF-Flat profile with synchronized stage timings, effective cuVS index
  records, fallback diagnostics, and fixed-query kNN graph samples; this stage
  does not overwrite the historical scaling records;
- `revision3/topology_sensitivity_results/`: the separately labelled
  spectral-only DiRe control at 100,000 and full scale;
- `revision3/evaluation/`: common-row local, centroid, context, and topology
  measurements, including fixed reference coordinates/diagrams, the historical
  diameter-normalized FastDTW approximation to Betti-curve distance, exact
  normalized bottleneck distances,
  a 99th-percentile pairwise-scale sensitivity analysis, a nested
  1,000/2,000/4,000-row subset-size check, and distinct row-subset versus
  layout-seed variability records. The primary audit uses ten independently
  seeded 4,000-row index subsets, each applied identically to the input and
  every embedding; three independently seeded full layouts are evaluated
  separately on one fixed subset;
- `revision3/small_json_logs/`: six reference datasets with 20 GPU and 10 CPU
  fits per method, direct GPU rank-based local-kNN atlas topology evaluated on
  every fit over paired, independently seeded fixed 1,000-row index subsets,
  with records requiring `prefer_ripser=false` and default DiRe as the fixed
  comparator;
- `revision3/bundles/`: compact deterministic archives plus SHA-256 sidecars;
- `revision3/downloads/` and `revision3/fetched/`: locally downloaded and
  safely verified bundles.

These directories are ignored because the public inputs and full layouts are
too large for ordinary Git history. `scripts/fetch_revision3_results.sh`
retrieves the compact archive from the GPU host, verifies both the outer hash
and every declared file, requires the matching successful CPU audit, renders
the publication artifacts, and only then records the active extraction in
`revision3/fetched/CURRENT`. It removes superseded Revision 3
downloads/extractions and requires exactly one current bundle, two matching
SHA-256 sidecars, one matching audit download, and a byte-identical installed
audit. The current semantic verifier also refuses a retained reducer
`result.json` with any status other than `success`.
The same contract verifier rejects an incomplete small suite, a non-atlas
small topology record, an invalid subset hash, or a method whose paired
row-index subset differs from the corresponding DiRe subset.
`scripts/build_revision3_local.sh` then regenerates
the tracked publication artifacts under `generated/revision3/` and replaces
the canonical `pics/` tree from the fetched twenty-run GPU and ten-run CPU
small-suite logs and embeddings. The replacement is clean rather than
incremental, and
`pics/render_manifest.json` records the source-bundle identifier plus exact
input/output membership, byte counts, and SHA-256 hashes. The local cleanliness
gate verifies that manifest against disk. The corresponding
`generated/revision3/render_manifest.json` does the same for every generated
figure, table, macro, summary, and installed CPU-audit record; it does not
need the raw 10x matrix, the arXiv input matrix, CUDA, or a GPU.
`make revision3-topology-audit` goes further: it recomputes every topology
score—including the nested subset-size and full-layout-seed sensitivities—from
the bundled fixed reference point clouds and full layouts, and checks the
values against the checksummed evaluation records. Independent pairs may run
concurrently; the emitted audit records its worker count, script hash,
platform, and dependency versions. The exact lock targets the
validated Python 3.10/Linux x86-64 environment; macOS/ARM does not have the
pinned GUDHI 3.13 package and should use the supplied Linux container rather
than silently changing the audit dependency.

The remote upload is not a clone of this data directory. The remote runner
first creates a deterministic, hash-verified archive containing exactly the
Revision 3 source allowlist and rejects any `data/`, `generated/`, `pics/`,
documentation-build, PDF, or common binary dataset member. The Docker context
is separately reduced to the Dockerfile and `requirements/bench.txt`. A
from-zero Revision 3 run uses
`scripts/run_remote_revision3.sh --fresh-remote-dir`; a normal clone uses the
committed payload above through `scripts/reproduce_archived.sh`.
When an existing remote data/cache tree is reused, the runner replaces the
entire executable source surface before extraction, so a deleted runner or
obsolete code path cannot survive from an earlier upload. After a successful
bundle and CPU audit, only the current bundle, extraction, audit, and pipeline
stage markers are retained.

The CPU-only audit derives its scalar-comparison count from the fetched bundle,
checks every value, and records the mismatch count and maximum absolute delta.
The resulting machine-readable comparison record is retained under
`generated/revision3/topology-audit.json`; no fixed comparison total is
hard-coded in this documentation.

The benchmark archive is evidence, not an opaque cache: it includes the exact
source revisions, transformations, package freeze, source commit, H100
identity, fixed row-index subsets, reducer parameters, success/failure records,
timings, memory samples, full-scale layouts, and evaluation outputs needed to
audit every manuscript number. It also embeds a source snapshot whose hashes
are recorded before reducer execution and rechecked during packaging. The
topology reference point clouds are deliberately
included even though the complete prepared matrices are excluded, allowing all
reported homology scores to be independently recomputed after fetching.
DiRe and every newly executed baseline are fitted to the full 1,306,127-cell
and 723,457-document inputs. The separately labelled Cell Ranger t-SNE
reference supplies upstream coordinates for all cells and is not represented
as a same-input or runtime rerun. Fixed row-index subsetting is confined to
diagnostics whose
exact Vietoris--Rips construction is intrinsically pairwise/combinatorial.
