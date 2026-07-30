DATA_ROOT ?= data/archived/json_logs
EMBEDDING_DATA_ROOT ?= data/archived/embedding_pngs
OUTPUT_ROOT ?= data/fresh_benchmark_results
JSON_LOG_ROOT ?= data/fresh_json_logs
EMBEDDING_PNG_ROOT ?= data/fresh_embedding_pngs
IMAGE ?= dire-paper-repro:latest
PYTHON ?= python3
MPLCONFIGDIR ?= .matplotlib-cache
REPEATS ?= 10
TOPOLOGY_REPEATS ?= 1
MAX_POINTS ?= 10000
FULL_DATASETS ?= 0
UMAP_MAX_POINTS ?= 10000
CPU_MAX_POINTS ?= $(UMAP_MAX_POINTS)
CPU_REPEATS ?= 3
CPU_JOBS ?= 1
METHODS ?= dire cuml_tsne cuml_umap opentsne umap
DIRE_ITERATIONS ?= 128
METRIC_SUBSAMPLE ?= 0.1
TOPOLOGY_SAMPLE_SIZE ?= 1000
TOPOLOGY_STEPS ?= 100
TOPOLOGY_AUDIT_WORKERS ?= 4
REVISION3_HOST ?=
REVISION3_SSH_KEY ?=
REVISION3_REMOTE_DIR ?= /home/ubuntu/homological-stability-repro-revision3

.PHONY: figures manuscript docs authorial-invariants revision-documents revision3-submission benchmarks reproduce all docker-build docker-figures docker-manuscript docker-benchmarks docker-reproduce revision3-gpu cuvs-knn-ab-gpu revision3-remote revision3-fetch revision3-render revision3-topology-audit clean

figures:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/render_figures.py --data-root $(DATA_ROOT) --embedding-png-root $(EMBEDDING_DATA_ROOT) --output .

manuscript:
	PYTHON=$(PYTHON) MPLCONFIGDIR=$(MPLCONFIGDIR) scripts/compile_manuscript.sh --data-root $(DATA_ROOT) --embedding-png-root $(EMBEDDING_DATA_ROOT)

docs: manuscript
	$(PYTHON) scripts/prepare_docs.py
	rm -rf docs/_build/html
	$(PYTHON) -m sphinx -b html docs docs/_build/html
	$(PYTHON) scripts/check_authorial_invariants.py --require-generated

authorial-invariants:
	$(PYTHON) scripts/check_authorial_invariants.py

revision-documents: manuscript
	latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_referees_round2.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error editor_cover_letter_round2.tex

revision3-submission:
	$(PYTHON) scripts/check_authorial_invariants.py
	$(MAKE) revision3-render PYTHON="$(PYTHON)"
	@bundle=$$(sed -n '1p' data/revision3/fetched/CURRENT); \
	$(MAKE) revision-documents \
	  PYTHON="$(PYTHON)" MPLCONFIGDIR="$(MPLCONFIGDIR)" \
	  DATA_ROOT="data/revision3/fetched/$$bundle/small_suite/json_logs" \
	  EMBEDDING_DATA_ROOT="data/revision3/fetched/$$bundle/small_suite/embedding_pngs"
	$(PYTHON) scripts/prepare_docs.py
	rm -rf docs/_build/html
	$(PYTHON) -m sphinx -b html docs docs/_build/html
	$(PYTHON) scripts/check_authorial_invariants.py --require-generated --require-clean-local-artifacts

benchmarks:
	PYTHON=$(PYTHON) MPLCONFIGDIR=$(MPLCONFIGDIR) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) CPU_REPEATS=$(CPU_REPEATS) CPU_JOBS=$(CPU_JOBS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) FULL_DATASETS=$(FULL_DATASETS) CPU_MAX_POINTS=$(CPU_MAX_POINTS) METHODS="$(METHODS)" DIRE_ITERATIONS=$(DIRE_ITERATIONS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_SIZE=$(TOPOLOGY_SAMPLE_SIZE) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS) scripts/run_benchmarks.sh $(OUTPUT_ROOT)

reproduce: benchmarks
	PYTHON=$(PYTHON) MPLCONFIGDIR=$(MPLCONFIGDIR) scripts/compile_manuscript.sh --data-root $(JSON_LOG_ROOT) --embedding-png-root $(EMBEDDING_PNG_ROOT)

all: figures manuscript

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-figures:
	docker run --rm -v $(CURDIR):/work -w /work $(IMAGE) make figures DATA_ROOT=$(DATA_ROOT) EMBEDDING_DATA_ROOT=$(EMBEDDING_DATA_ROOT)

docker-manuscript:
	docker run --rm -v $(CURDIR):/work -w /work $(IMAGE) make manuscript DATA_ROOT=$(DATA_ROOT) EMBEDDING_DATA_ROOT=$(EMBEDDING_DATA_ROOT)

docker-benchmarks:
	docker run --rm --gpus all -v $(CURDIR):/work -w /work $(IMAGE) make benchmarks OUTPUT_ROOT=$(OUTPUT_ROOT) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) CPU_REPEATS=$(CPU_REPEATS) CPU_JOBS=$(CPU_JOBS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) FULL_DATASETS=$(FULL_DATASETS) CPU_MAX_POINTS=$(CPU_MAX_POINTS) METHODS="$(METHODS)" DIRE_ITERATIONS=$(DIRE_ITERATIONS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_SIZE=$(TOPOLOGY_SAMPLE_SIZE) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS)

docker-reproduce:
	docker run --rm --gpus all -v $(CURDIR):/work -w /work $(IMAGE) make reproduce OUTPUT_ROOT=$(OUTPUT_ROOT) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) FULL_DATASETS=$(FULL_DATASETS) UMAP_MAX_POINTS=$(UMAP_MAX_POINTS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_SIZE=$(TOPOLOGY_SAMPLE_SIZE) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS)

revision3-gpu:
	scripts/reproduce_revision3.sh

cuvs-knn-ab-gpu:
	scripts/reproduce_cuvs_knn_ab.sh

revision3-remote:
	scripts/run_remote_revision3.sh --host $(REVISION3_HOST) --ssh-key $(REVISION3_SSH_KEY) --remote-dir $(REVISION3_REMOTE_DIR)

revision3-fetch:
	scripts/fetch_revision3_results.sh --host $(REVISION3_HOST) --ssh-key $(REVISION3_SSH_KEY) --remote-dir $(REVISION3_REMOTE_DIR)

revision3-render:
	PYTHON=$(PYTHON) scripts/build_revision3_local.sh

revision3-topology-audit:
	@bundle=$$(sed -n '1p' data/revision3/fetched/CURRENT); \
	$(PYTHON) scripts/audit_revision3_topology.py \
	  --bundle-root "data/revision3/fetched/$$bundle" \
	  --workers "$(TOPOLOGY_AUDIT_WORKERS)"

clean:
	latexmk -C dire_short.tex || true
	latexmk -C response_to_referees_round2.tex || true
	latexmk -C editor_cover_letter_round2.tex || true
