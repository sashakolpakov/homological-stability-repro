FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        biber \
        ca-certificates \
        git \
        latexmk \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements/bench.txt /tmp/requirements-bench.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-bench.txt \
    && python3 -m pip install --no-cache-dir sphinx alabaster

CMD ["bash"]
