# DiRe paper reproducibility package

This repository contains the benchmark artifacts and build scripts for regenerating the figures, manuscript PDF, and Sphinx HTML manuscript for the DiRe paper. The archived data path does not require access to the original Lambda Labs instance.

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
- `scripts/run_remote_gpu.sh`: uploads the repo to a GPU host, builds the Docker image there, runs the benchmark suite, and downloads JSON logs plus canonical embedding PNGs.
- `scripts/run_lambda_gpu.sh`: one-command Lambda.ai/Lambda Labs wrapper around the generic remote GPU runner.
- `docker/Dockerfile`: GPU-capable benchmark and manuscript build environment matching the NumPy 2/RAPIDS setup used on Lambda Labs.
- `docs/`: Sphinx source for the GitHub Pages site.

## Zero-to-hero from archived data

From a fresh clone on a normal CPU machine:

```bash
scripts/reproduce_archived.sh
```

This uses the archived JSON logs and embedding PNGs in `data/`. It regenerates:

- `pics/`: manuscript figure tree
- `dire_short.pdf`: compiled LaTeX manuscript
- `docs/_build/html/paper.html`: Sphinx clone of `dire_short.tex`
- `docs/_build/html/figures.html`: figure gallery

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

Fresh benchmark outputs are written to `data/fresh_benchmark_results` and packed into self-contained JSON logs in `data/fresh_json_logs` by default. The benchmark script uses one canonical seed for embedding pictures and repeated seeded runs for metric means and standard-deviation error bars. The default is `REPEATS=10`; override it on the `make` command if needed. Persistent-homology metrics are more expensive and default to `TOPOLOGY_REPEATS=1`; increase this only if you explicitly want repeated topology computations. Local and context metrics use `METRIC_SUBSAMPLE=0.1`. The topology pass is separate, uses the ripser backend, and uses `TOPOLOGY_SAMPLE_FRACTION=0.05` by default, recorded in each JSON log. The runner rejects topology sample fractions below `0.05`.

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

## Run on a remote GPU host and pull results back

The remote GPU workflow is split into two scripts:

- `scripts/run_remote_gpu.sh` is the generic SSH/Docker harness. It works with any reachable Linux GPU host that has SSH, Docker, NVIDIA drivers, and Docker GPU support.
- `scripts/run_lambda_gpu.sh` is a small [Lambda.ai / Lambda Labs](https://lambda.ai/service/gpu-cloud) convenience wrapper. It accepts a Lambda instance IP address, assumes the usual `ubuntu` SSH user unless told otherwise, prints the matching public key if available, and then delegates to `run_remote_gpu.sh`.

Before starting a remote run, make sure:

- The GPU instance has been booted and its SSH login is known, for example `ubuntu@132.145.143.182`.
- The public key matching your private key has been registered with the instance. For example, if the private key is `/Users/sasha/.ssh/id_ed25519`, register `/Users/sasha/.ssh/id_ed25519.pub`.
- Docker is installed on the remote host and can run either as the SSH user or via passwordless `sudo docker`.
- `nvidia-smi` works on the remote host.
- The local working tree contains the code and data you intend to upload. Generated local outputs are excluded during upload.

For a Lambda Labs instance, run:

```bash
scripts/run_lambda_gpu.sh --ip 132.145.143.182 --ssh-key /Users/sasha/.ssh/id_ed25519
```

The equivalent generic SSH/Docker command is:

```bash
HOST=ubuntu@132.145.143.182 SSH_KEY=/Users/sasha/.ssh/id_ed25519 scripts/run_remote_gpu.sh
```

Both commands run the same pipeline:

1. Check SSH, Docker, and `nvidia-smi` on the remote host.
2. Detect whether Docker should be called as `docker` or `sudo docker`, unless `REMOTE_DOCKER` is set explicitly.
3. Upload the current repository to `REMOTE_DIR` on the remote host, excluding generated artifacts.
4. Build the benchmark Docker image on the remote host.
5. Run the selected GPU task in that Docker image.
6. Download the task's declared outputs to local ignored directories.
7. Verify that the expected output files exist.

The default task is `benchmarks`. It runs the full benchmark suite and downloads:

- `data/remote_json_logs`
- `data/remote_embedding_pngs`

Build every local artifact from those downloaded files with:

```bash
DATA_ROOT=data/remote_json_logs EMBEDDING_DATA_ROOT=data/remote_embedding_pngs scripts/run_pipeline.sh all
```

This regenerates metric diagrams, combines the embedding PNGs with legends inferred from JSON labels, compiles LaTeX, and builds the Sphinx site.

The second task is `topology-sweep`. It runs a short blobs-only sweep over topology sample fractions and downloads:

- `data/topology_sampling_sweep`

Lambda wrapper:

```bash
scripts/run_lambda_gpu.sh --ip 132.145.143.182 --ssh-key /Users/sasha/.ssh/id_ed25519 --task topology-sweep
```

Generic runner:

```bash
HOST=ubuntu@132.145.143.182 SSH_KEY=/Users/sasha/.ssh/id_ed25519 GPU_TASK=topology-sweep scripts/run_remote_gpu.sh
```

Useful sweep overrides are `SWEEP_FRACTIONS="0.05 0.1 0.2"`, `SWEEP_REPEATS=1`, and `WARN_TOPOLOGY_HOURS=1.0`. The summary file is `data/topology_sampling_sweep/summary.json`.

Useful generic overrides:

- `REMOTE_DIR=~/homological-stability-repro-run`: remote working directory.
- `REMOTE_DOCKER=docker` or `REMOTE_DOCKER="sudo docker"`: force Docker command instead of autodetection.
- `DOCKER_BUILD_FLAGS=--no-cache`: force a fresh Docker dependency install.
- `REPEATS=10`: metric repeats for the full benchmark suite.
- `TOPOLOGY_REPEATS=1`: topology repeats for full benchmarks and topology calls per sweep repeat.
- `TOPOLOGY_SAMPLE_FRACTION=0.1`: topology sample fraction for full benchmarks.
- `MAX_POINTS=10000`, `METRIC_SUBSAMPLE=0.1`, `TOPOLOGY_STEPS=100`: benchmark sizing knobs.

Lambda Cloud also provides an optional [Guest Agent](https://docs.lambda.ai/public-cloud/guest-agent/) for console-side observability. It is not required by this repository or by Docker, but it is useful on Lambda instances if you want GPU, VRAM, and system metrics to appear in the Lambda Cloud console:

```bash
ssh ubuntu@<IP-ADDRESS>
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash
sudo systemctl --no-pager status lambda-guest-agent*
```

For the archived Lambda run, the same local pipeline is:

```bash
scripts/run_pipeline.sh all
```

## Notes

The Docker image installs DiRe-RAPIDS from the upstream `main` branch. For archival runs, record the resolved DiRe-RAPIDS commit hash from the benchmark JSON manifest.

## Funding

The archived benchmark logs were produced on Lambda.ai/Lambda Labs GPU infrastructure. The broader project has also been supported by the Google Cloud Research Award number GCP19980904. We thank both Lambda.ai and Google Cloud for supporting the computational work.
