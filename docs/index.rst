Homological Stability Reproducibility
=====================================

This site accompanies the DiRe manuscript. It provides the compiled paper, LaTeX source, benchmark logs, and regenerated figures.

Artifacts
---------

* `Sphinx manuscript clone <paper.html>`_
* `Compiled manuscript PDF <_static/paper/dire_short.pdf>`_
* `LaTeX source <_static/paper/dire_short.tex>`_
* `Bibliography <_static/paper/dire_short.bib>`_

Reproducibility
---------------

The repository stores self-contained JSON benchmark logs under ``data/archived/json_logs``. The command ``make figures`` regenerates manuscript figures from those logs. The Docker workflow can rerun the full benchmark suite on an NVIDIA GPU and produce fresh JSON logs. Downloaded remote benchmark logs and canonical embedding PNGs, for example ``data/remote_json_logs_full`` and ``data/remote_embedding_pngs_full``, can also be used as GPU-free inputs to ``scripts/run_pipeline.sh all``.

Funding
-------

The archived benchmark logs were produced on Lambda.ai/Lambda Labs GPU infrastructure. The broader project has also been supported by the Google Cloud Research Award number GCP19980904.

.. toctree::
   :maxdepth: 1

   paper
   figures
