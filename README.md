# DiRe paper reproducibility package

This repository contains the benchmark artifacts and build scripts for regenerating the figures, manuscript PDF, and Sphinx HTML manuscript for the DiRe paper. The archived data path does not require access to the original Lambda Labs instance.

DiRe is a force-directed dimensionality-reduction framework designed to
preserve global structure and homological features while remaining practical
on modern hardware. It is designed less as a purely local visualization
heuristic and more as a framework for embeddings whose large-scale geometry
can be quantified through Betti curves and persistence diagrams. DiRe accepts
a user-specified target dimension \(d\), including non-visual dimensions used
in subsequent analysis. The Revision 3 large-data profile in this repository
evaluates \(d=2\) outputs; that recorded parameter is not the scope of the
algorithm.

The deployed Sphinx manuscript is available at:

https://sashakolpakov.github.io/homological-stability-repro/

## What is included

- `dire_short.tex` and `dire_short.bib`: manuscript source.
- `data/archived/json_logs`: self-contained JSON logs from the Lambda Labs benchmark run, including environment records, dataset metadata, labels, embeddings, and metrics.
- `scripts/render_figures.py`: rebuilds the `pics/` tree expected by the LaTeX source.
- `data/archived/embedding_pngs`: canonical embedding PNGs from the archived benchmark run.
- `scripts/compile_manuscript.sh`: renders figures and compiles the manuscript.
- `scripts/latex_to_sphinx.py`: converts `dire_short.tex` into `docs/paper.rst` for the Sphinx manuscript clone.
- `scripts/run_benchmarks.sh`: reruns the benchmark suite and writes fresh logs.
- `scripts/reproduce_revision3.sh`: one-command, version-pinned million-point Revision 3 rerun on an 80 GB NVIDIA GPU followed by an independently isolated CPU-only topology audit.
- `scripts/run_remote_revision3.sh`: creates a size-gated, content-verified source-only archive and launches that rerun under `nohup`, optionally preserving remote run state and surviving SSH disconnects.
- `scripts/package_revision3_remote_source.py`: deterministically packages and verifies the executable remote source surface; its allowlist cannot contain `data/`, `generated/`, `pics/`, documentation builds, PDFs, or common binary dataset formats.
- `scripts/verify_archived_clone_payload.py`: verifies the distinct ordinary-clone mode, including all committed six-dataset logs, 30 embeddings, publication artifacts, real file contents, byte-size floors, and Git tracking.
- `scripts/fetch_revision3_results.sh`: resumably fetches the compact result bundle and matching CPU-only audit with `rsync`, verifies both SHA-256 sidecars plus every bundled file, and regenerates figures/tables locally.
- `docker/Dockerfile`: GPU-capable benchmark and manuscript build environment matching the NumPy 2/RAPIDS setup used on Lambda Labs.
- `docs/`: Sphinx source for the GitHub Pages site.

## Revision 3: push-button million-point rerun

This profile answers the scalability, initialization, baseline, dataset-size,
and presentation concerns raised in the second review round. It is separate
from the historical benchmark archive so that the original records remain
immutable.

There are two deliberately separate operating modes:

1. **Full remote rerun:** the upload script builds a deterministic archive
   containing exactly the Revision 3 source-file allowlist. Before any SSH
   transfer, the archive is rejected if its membership differs from that
   allowlist, contains a benchmark/derived-artifact path or data-like binary
   suffix, exceeds 16 MiB uncompressed or 8 MiB compressed, or fails its
   per-file hashes. The Docker context is independently allowlisted to the
   Dockerfile and pinned benchmark requirements. Thus committed benchmark data
   are neither uploaded nor sent to the Docker daemon; public inputs are
   fetched and prepared on the GPU host.
2. **Ordinary clone with committed evidence:** `git clone` retains
   `data/archived/` and `generated/revision3/`. Run
   `scripts/reproduce_archived.sh --check-only` for a network-free payload
   check, or `scripts/reproduce_archived.sh` for the complete GPU-free local
   rebuild. This mode consumes the committed archive and never pretends to be a
   from-zero GPU benchmark.

The two public large-data tests are:

- **10x embryonic mouse brain:** all 1,306,127 cells in the official Cell
  Ranger 20-component PCA projection, with the released k-means-20 and graph
  clusters, differential-expression output, and historical Cell Ranger t-SNE.
  The official analysis archive is pinned by SHA-256.
- **arXiv BGE-small corpus:** all 723,457 paper-level, 384-dimensional
  embeddings plus primary-category metadata from
  `igriv/dire-arxiv-bge-small-embeddings`, pinned to Hugging Face revision
  `9b4ceb24ccd20b5d82956ee26a1d80e678d45a7b`.

Preparation applies one documented common transformation to each input array.
Every newly executed reducer receives the same float32 rows in the same order.
The released full-cell Cell Ranger t-SNE coordinates are retained as an
external 10x reference, not as a same-input or runtime baseline. The GPU
runtime panel compares production-policy DiRe, a separately named forced
IVF-Flat DiRe control, cuML UMAP, cuML t-SNE, and cuML PCA, each configured
with target dimension \(d=2\), on the same GPU. CPU openTSNE and `umap-learn`
are run and labelled in a separate CPU-reference panel; CPU and GPU timings are
never mixed into a hardware-normalized ranking.

The production DiRe row uses the documented `cuvs_index_type="auto"` policy
and default `init="pca"` pathway. In the pinned cuVS implementation that
pathway uses cuML PCA for the 20-feature 10x input and cuML TruncatedSVD for the
384-feature arXiv input. The forced `cuvs_index_type="ivf_flat"` row is retained
as an explicit control, never overwritten or relabelled. For both policies, a
separate paired `backend_policy_results` stage records fresh repeats without
modifying the forced IVF-Flat control files. Each paired repeat records the
requested and effective cuVS index, synchronized
kNN-graph/initialization/layout timings, compile status, and chunked-force
fallback count. A fixed 20,000-query graph sample measures neighbor-set overlap
between the two policies. The suite does not promote a speed result until the
production-auto and forced-control embeddings have also passed through the
complete predeclared local/global/context/topology evaluation; it imposes no
post-hoc equivalence threshold. Library-level diagnostics are tracked
independently in
[`dire-rapids` issue #13](https://github.com/sashakolpakov/dire-rapids/issues/13).

One single-factor sensitivity row changes only that initialization to spectral
while holding all force parameters fixed. It is labelled as a sensitivity
analysis and is never substituted for the default result.

The six--dataset reference suite uses one consistent small--data topology
protocol. Every GPU configuration has 20 seeded fits and every CPU reference
has 10. The runner calls the GPU rank--based local--kNN atlas evaluator
directly on every fit using a fixed 1,000-row index subset; every record must
identify the atlas backend, the direct evaluator, and
`prefer_ripser=false`. The subset seed is independent of the layout seed, and
each corresponding method run records the identical indices and SHA--256
digest. Default DiRe is the fixed comparator. Generated outputs report mean,
sample standard deviation, minimum, every paired gap, and continuous
direction--adjusted log ratios rather than ordinal ranks.

The topology protocol uses ten independently seeded, fixed 4,000-row index
subsets. Every subset is applied identically to the high-dimensional input and
every embedding; these are diagnostic subsets, not reducer fits or biological
specimens. Three independently seeded full layouts are evaluated separately on
the first fixed subset. The suite
retains the historical unit-diameter, FastDTW-approximated Betti-curve
distances, adds exact
diameter-normalized bottleneck distances, and reports a separately labelled
99th-percentile pairwise-scale sensitivity analysis because a single extreme
point can dominate a diameter. The compact result bundle includes each sampled
high-dimensional reference point cloud and its persistence diagrams, so these
scores can be audited and recomputed locally rather than trusted as manuscript
transcriptions. A second, explicitly nested 1,000/2,000/4,000-row calculation
holds the canonical full-data layout fixed and tests whether the former
2,000-row effects reverse with diagnostic subset size. The generated audit
reports every subset-specific paired gap, its mean and relative effect, and
the observed paired standard deviation rather than rank or sign counts.

For every stochastic nonlinear GPU method, the targeted stage also retains
three independently seeded full layouts. Topology variation across the ten
fixed row-index subsets on the canonical layout is reported separately from
variation across three layout seeds on one fixed row-index subset. This
prevents subset instability and optimizer stochasticity from being merged into one
uninterpretable error bar.

DiRe and every newly executed baseline are fitted to every row; only the
topology diagnostic is computed on fixed row-index subsets. The external Cell
Ranger reference contains
coordinates for every cell but was generated upstream. This distinction is
essential: the 1,306,127-cell dataset has approximately 8.53e11 unordered
pairs, before any higher-dimensional Vietoris--Rips simplices are considered.
A full exact persistence computation is therefore not a meaningful
requirement for testing DiRe on a million-point dataset. Fixed, shared
row-index subsets make the diagnostic computationally defined, method-neutral,
repeatable, and open to subset-size sensitivity checks.

The full profile expects an NVIDIA GPU with at least 70,000 MiB of VRAM. On the
GPU host, from a clean clone:

```bash
scripts/reproduce_revision3.sh
```

That one command builds the pinned container, prepares both datasets, runs
nested fixed-seed scaling sweeps through 1,306,127 cells and 723,457 papers,
records cold/steady runtime and peak incremental device-memory use sampled
through NVML, runs shared-row-subset
local/global/context/topology evaluation, and repeats the nonlinear methods at
100,000 observations and full scale for a separately labelled sensitivity
audit that includes a one-factor spectral-initialization control. It then audits
production auto cuVS against the forced IVF-Flat control with fixed-query graph
overlap and synchronized stage timings, runs the
CPU-reference small suite, runs PCA-versus-spectral initialization ablations,
creates a compact checksummed result archive, and then starts a second
container with no GPU exposed to recompute every bundled topology score.
The H100 profile defaults to 24 concurrent topology workers, twenty GPU
small-suite fits and ten CPU-reference fits per configuration (with the atlas
diagnostic evaluated on every fit), and 24 workers for the independent
CPU-only topology audit; each value remains explicitly overrideable by its
corresponding environment variable and is recorded in the output.
The remote pipeline is marked successful only after that audit succeeds and
its own SHA-256 sidecar is written. Public raw downloads, caches,
and the two full high-dimensional prepared matrices stay on the GPU host and are intentionally
excluded from the compact archive; the much smaller fixed topology reference
subsets are included for audit. Before any reducer runs, the suite hashes its
complete Revision 3 source surface. Packaging rechecks those hashes and embeds
the exact runner/lockfile snapshot, so an old Git HEAD plus an unexplained
dirty worktree cannot masquerade as source provenance.

For a remote host, launch the same pipeline without tying it to an SSH
connection:

```bash
scripts/run_remote_revision3.sh \
  --host ubuntu@GPU_HOST \
  --ssh-key /path/to/id_ed25519 \
  --fresh-remote-dir
```

The command prints the remote log/status paths. It refuses to launch a second
job when the recorded PID is still alive unless `--force-launch` is explicitly
given. `--fresh-remote-dir` deletes an idle previous worktree, then performs a
source-only upload and prepares all public inputs from zero; it never leaves a
renamed stale worktree behind. Omit that option only when deliberately
resuming the existing remote `data/revision3` downloads, stages, and results.
On reuse, the entire executable source surface is replaced before extraction.
In either case, local committed data and generated publication artifacts are
never in the transfer archive or Docker build context.

After the remote status reports `state=success`, fetch, verify, and render on a
normal CPU workstation. The fetch host and remote host must both provide
`rsync`; interrupted transfers retain a deterministic partial file and resume
on the next invocation:

```bash
python3 -m pip install -r requirements/render.txt
scripts/fetch_revision3_results.sh \
  --host ubuntu@GPU_HOST \
  --ssh-key /path/to/id_ed25519
```

The fetcher:

1. reads the immutable archive named by the remote `LATEST` pointer;
2. verifies the archive SHA-256 sidecar;
3. rejects absolute paths, traversal, links, devices, duplicate entries, and
   unexpectedly large archives before extraction;
4. checks the recorded byte count and SHA-256 of every extracted file;
5. requires the current semantic contract: ten complete 4,000-row topology
   subsets, nested 1,000/2,000/4,000 records, bundled input centroids, and no
   retired replication/sign-count fields; it separately requires all six
   small--suite logs, the exact declared method set, 20 GPU and
   10 CPU repeats, direct atlas topology on every fit, fixed 1,000-row subset
   metadata, valid index hashes, and identical paired subsets across methods.
   It also rejects every retained reducer `result.json` whose status is not
   `success`;
6. downloads the audit named by `data/revision3/cpu_audit/LATEST`, verifies its
   SHA-256, requires `status=success`, and requires its `source_bundle` to be
   the exact fetched bundle;
7. installs the verified audit as
   `generated/revision3/topology-audit.json`;
8. replaces `generated/revision3` and regenerates the layouts, scaling plots,
   quality summaries, CSV tables, TeX fragments, backend-policy timing/overlap
   and full-quality-gate artifacts, normalization/seed/subset-size
   topology-sensitivity artifacts, value macros, marker table, and
   initialization-ablation figure. A CPU-audit JSON is carried across that
   replacement only when it records the same fetched bundle as its source. It
   also replaces the canonical `pics/` tree from the same bundle's fresh
   twenty-run GPU and ten-run CPU small-suite logs and embeddings, so the
   manuscript runtime and quality summaries cannot fall back to the older
   archived run. The renderer
   first clears that tree and writes `pics/render_manifest.json`, recording the
   exact source-bundle identifier, source-file hashes, output membership, byte
   counts, and output hashes;
9. updates `data/revision3/fetched/CURRENT` only after rendering succeeds,
   removes every superseded Revision 3 download and extraction, and runs the
   local clean-artifact invariant requiring exactly the current bundle, its
   two SHA-256 sidecars, its matching successful CPU audit, and byte-for-byte
   agreement with both generated-artifact manifests. Each manifest declares
   exact output membership, byte counts, and SHA-256 hashes, so an old or
   untracked figure cannot silently coexist with the final render. The final
   submission audit also requires the Sphinx source-static and built-static
   artifact trees to be byte-identical to those canonical renders.

The verified bundle also contains `source/source-manifest.json` and the exact
checksummed runner files used on the H100.

Rendering can be repeated without SSH or a GPU:

```bash
scripts/build_revision3_local.sh
```

After rendering, one target compiles the revised manuscript, point-by-point
response, editor letter, and Sphinx site from the verified records:

```bash
make revision3-submission
```

To repeat the already required independent CPU recomputation on another host,
install the small exact audit environment and run:

```bash
python3 -m pip install -r requirements/topology-audit.txt
make revision3-topology-audit
```

Independent topology pairs run concurrently (four workers by default) while
the result order and pairwise comparisons remain deterministic. Set
`TOPOLOGY_AUDIT_WORKERS=8` on a larger CPU host. The audit JSON records the
worker count, logical CPU count, platform, Python/dependency versions, and
audit-script SHA-256.

The exact audit lock is the Python 3.10/Linux x86-64 environment validated on
the benchmark host. In particular, PyPI does not publish GUDHI 3.13 for
macOS/ARM; do not silently substitute an older GUDHI release and call the
result an exact environment match. On such a host, run the audit in the
provided Linux container without passing a GPU:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile \
  -t homological-stability-repro:revision3 .
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/work -w /work \
  homological-stability-repro:revision3 \
  make revision3-topology-audit PYTHON=python3
```

The local fetch, checksum verification, rendering, manuscript, and Sphinx
steps remain platform-independent.

This re-runs Ripser, the exact sorted zero-birth-bar specialization for H0
bottleneck distance, compiled exact GUDHI matching for H1, and the FastDTW
approximation to Betti-curve DTW from the bundled reference subsets and all
retained full layouts, compares independently seeded row-index subsets, nested
1,000/2,000/4,000 subset sizes, and
full-layout-seed topology outputs against the checksummed JSON, and writes
`generated/revision3/topology-audit.json`. It does not download either complete
dataset.

For every fetched bundle, the audit records the exact scalar-comparison count,
mismatch count, and maximum absolute difference rather than relying on a
hard-coded total. The committed JSON also records the
Linux/Python/dependency environment and audit-script SHA-256 rather than
treating the bundled topology scores as trusted cache entries.

Equivalent Make targets are `revision3-gpu`, `revision3-remote`,
`revision3-fetch`, `revision3-render`, and `revision3-topology-audit`. For the
remote targets, set `REVISION3_HOST` and `REVISION3_SSH_KEY`.
`revision3-submission` is the GPU-free downstream build once a verified bundle
has been fetched. Its renderer re-verifies the fetched archive/audit hashes,
source-bundle identity, successful audit status, and single-bundle cleanliness,
replaces every downstream render, and then applies the strict byte-for-byte
cleanliness gate. Thus stale copies are repaired rather than blocking their own
regeneration, while macOS still cannot silently substitute a different GUDHI
environment.
`revision3-topology-audit` remains the explicit command for repeating the
already required remote CPU audit in the pinned Linux environment.

## Zero-to-hero from archived data

From a fresh clone on a normal CPU machine:

```bash
scripts/reproduce_archived.sh --check-only
scripts/reproduce_archived.sh
```

The first command proves that the committed payload is present, is not a set of
Git-LFS pointer stubs, has the exact six-dataset/30-image contract, includes
the required Revision 3 publication artifacts, and is tracked by Git. The
second uses that payload and regenerates:

- `pics/`: manuscript figure tree
- `dire_short.pdf`: compiled LaTeX manuscript
- `docs/_build/html/paper.html`: Sphinx clone of `dire_short.tex`
- `docs/_build/html/figures.html`: figure gallery
- `docs/_build/html/revision3.html`: million-point Revision 3 evidence page

The publication-ready Revision 3 tables, raster figures, and vector-PDF figures
under `generated/revision3/` are committed so this archived path is complete in
a fresh clone. They can be independently regenerated from the checksummed H100
bundle with `make revision3-render`.

The manuscript build requires `latexmk`, `pdflatex`, and `biber` on the host. If TeX is not installed locally, use Docker.

The equivalent manual commands are:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/render.txt -r requirements/docs.txt
scripts/run_pipeline.sh all
```

## Use archived data selectively

Generate only the figures:

```bash
python3 -m pip install -r requirements/render.txt
make figures
```

If your shell resolves `python3` to an environment without NumPy/Matplotlib, set `PYTHON=/path/to/python` on the `make` command.

Compile the manuscript from LaTeX and archived data:

```bash
make manuscript
```

The manuscript build requires `latexmk`, `pdflatex`, and `biber` on the host. If they are not installed, use the Docker workflow below.

Build the Sphinx site with the compiled PDF, regenerated figures, and Sphinx manuscript clone:

```bash
python3 -m pip install -r requirements/render.txt -r requirements/docs.txt
make docs
```

## Run in Docker

Docker is the easiest way to reproduce the full software environment. Install Docker with NVIDIA GPU support on GPU hosts if you plan to rerun benchmarks.

Build the image:

```bash
make docker-build
```

Generate figures from archived data:

```bash
make docker-figures
```

Compile the manuscript:

```bash
make docker-manuscript
```

Rerun the benchmarks from scratch on an NVIDIA GPU:

```bash
make docker-benchmarks
```

Fresh benchmark outputs are written to `data/fresh_benchmark_results` and packed into self-contained JSON logs in `data/fresh_json_logs` by default. The benchmark script uses one canonical seed for embedding pictures and repeated seeded runs for metric means and standard-deviation error bars. The default is `REPEATS=10`; override it on the `make` command if needed. Runtime reporting does not mix startup with steady execution: the first successful fit for each method–dataset configuration is retained as the configuration cold start, and subsequent successful fits form the steady-state mean and sample standard deviation. Atlas-topology metrics are more expensive and default to `TOPOLOGY_REPEATS=1`; increase this only if you explicitly want repeated topology computations. Local and context metrics use `METRIC_SUBSAMPLE=0.1`. The topology pass calls the direct GPU rank-based local-kNN atlas evaluator with `prefer_ripser=false` on a fixed-size, uniformly sampled, paired row-index subset. `TOPOLOGY_SAMPLE_SIZE=1000` by default; each JSON record includes the subset seed, indices, and SHA-256 digest.

MNIST loading first attempts OpenML (`mnist_784`). If OpenML times out or has another network failure, the benchmark loader automatically falls back to Hugging Face (`ylecun/mnist`, then `mnist`) and records the effective source in the JSON log. The Hugging Face fallback concatenates the `train` and `test` splits before any benchmark row cap is applied, matching the combined OpenML source. Set `FULL_DATASETS=1` to disable the row cap for fixed external datasets such as MNIST and Levine. The original CPU `umap-learn` baseline has a separate `UMAP_MAX_POINTS=10000` cap by default so full-dataset GPU runs do not force vanilla UMAP onto the full input; set `UMAP_MAX_POINTS=0` only if you explicitly want to disable that cap.

To render figures from those fresh JSON logs:

```bash
make docker-figures DATA_ROOT=data/fresh_json_logs
```

To compile the manuscript from those fresh JSON logs:

```bash
make docker-manuscript DATA_ROOT=data/fresh_json_logs
```

Or run the full fresh pipeline in one command:

```bash
make docker-reproduce
```

## Notes

The Docker image installs the current DiRe-RAPIDS `main` revision resolved on
2026-07-28, pinned immutably as
`293b622cc79fa8ea6fd5b54009e0930e3385b22f`. The runners record the requested
branch and resolution date plus the installed package's PEP 610
`direct_url.json` metadata, and fail if the installed commit does not match.
Dataset revisions, input transformations, hardware, software versions, subset
indices, method parameters, failures, and timing/memory protocols are retained
in machine-readable manifests.

## Funding

The archived benchmark logs were produced on Lambda.ai/Lambda Labs GPU infrastructure. The broader project has also been supported by the Google Cloud Research Award number GCP19980904. We thank both Lambda.ai and Google Cloud for supporting the computational work.
