# Archived benchmark artifacts

This directory is the GPU-free entry point.

- `archived/json_logs/` contains self-contained JSON logs for all six datasets. These logs include environment records, labels, canonical embeddings, metric values, and repeated-run aggregate metrics. Each aggregate records its own `n`, so repeated scalar metrics and more expensive topology metrics can be audited separately.
- `archived/embedding_pngs/` contains all canonical embedding PNGs for the archived run: 6 datasets times 4 methods.
- `label_metadata.json` maps raw benchmark labels to legend text. MNIST labels are digit identities; Levine labels use the upstream population-name files from the Levine 13- and 32-dimensional benchmark-data repositories.

Run the local pipeline from these files with:

```bash
scripts/reproduce_archived.sh
```

No Lambda.ai access, GPU, CUDA, RAPIDS, PyTorch, or Docker is required for this archived-data path. A normal Python environment with NumPy/Matplotlib plus a TeX installation is enough to rebuild figures and the manuscript PDF.

Remote benchmark downloads are also GPU-free render inputs once present locally. The standard remote runner downloads packaged JSON logs and canonical embedding PNGs to ignored directories such as `remote_json_logs/` and `remote_embedding_pngs/`; custom runs may use suffixed roots such as `remote_json_logs_full/` and `remote_embedding_pngs_full/`. Pass those roots as `DATA_ROOT` and `EMBEDDING_DATA_ROOT` to `scripts/run_pipeline.sh all` to regenerate `pics/`, `dire_short.pdf`, and the Sphinx documentation without rerunning benchmarks.
