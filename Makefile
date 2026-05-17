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
METRIC_SUBSAMPLE ?= 0.1
TOPOLOGY_SAMPLE_FRACTION ?= 0.05
TOPOLOGY_STEPS ?= 100

.PHONY: figures manuscript docs benchmarks reproduce all docker-build docker-figures docker-manuscript docker-benchmarks docker-reproduce remote-benchmarks lambda-benchmarks clean

figures:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/render_figures.py --data-root $(DATA_ROOT) --embedding-png-root $(EMBEDDING_DATA_ROOT) --output .

manuscript:
	PYTHON=$(PYTHON) MPLCONFIGDIR=$(MPLCONFIGDIR) scripts/compile_manuscript.sh --data-root $(DATA_ROOT) --embedding-png-root $(EMBEDDING_DATA_ROOT)

docs: manuscript
	$(PYTHON) scripts/prepare_docs.py
	rm -rf docs/_build/html
	$(PYTHON) -m sphinx -b html docs docs/_build/html

benchmarks:
	PYTHON=$(PYTHON) MPLCONFIGDIR=$(MPLCONFIGDIR) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_FRACTION=$(TOPOLOGY_SAMPLE_FRACTION) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS) scripts/run_benchmarks.sh $(OUTPUT_ROOT)

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
	docker run --rm --gpus all -v $(CURDIR):/work -w /work $(IMAGE) make benchmarks OUTPUT_ROOT=$(OUTPUT_ROOT) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_FRACTION=$(TOPOLOGY_SAMPLE_FRACTION) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS)

docker-reproduce:
	docker run --rm --gpus all -v $(CURDIR):/work -w /work $(IMAGE) make reproduce OUTPUT_ROOT=$(OUTPUT_ROOT) JSON_LOG_ROOT=$(JSON_LOG_ROOT) EMBEDDING_PNG_ROOT=$(EMBEDDING_PNG_ROOT) REPEATS=$(REPEATS) TOPOLOGY_REPEATS=$(TOPOLOGY_REPEATS) MAX_POINTS=$(MAX_POINTS) METRIC_SUBSAMPLE=$(METRIC_SUBSAMPLE) TOPOLOGY_SAMPLE_FRACTION=$(TOPOLOGY_SAMPLE_FRACTION) TOPOLOGY_STEPS=$(TOPOLOGY_STEPS)

remote-benchmarks:
	scripts/run_remote_gpu.sh

lambda-benchmarks:
	scripts/run_lambda_gpu.sh

clean:
	latexmk -C dire_short.tex || true
