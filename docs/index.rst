Homological Stability Reproducibility
=====================================

This site accompanies the DiRe manuscript. It provides the compiled paper, LaTeX source, benchmark logs, and regenerated figures.

DiRe is a force-directed dimensionality-reduction framework designed to
preserve global structure and homological features while remaining practical
on modern hardware. It is designed less as a purely local visualization
heuristic and more as a framework for embeddings whose large-scale geometry
can be quantified through Betti curves and persistence diagrams. DiRe accepts
a user-specified target dimension :math:`d`, including non-visual dimensions
used in subsequent analysis. The large-data experiments documented here
instantiate :math:`d=2`; that recorded parameter is not the scope of the
algorithm.

Artifacts
---------

* `Sphinx manuscript clone <paper.html>`_
* `Compiled manuscript PDF <_static/paper/dire_short.pdf>`_
* `LaTeX source <_static/paper/dire_short.tex>`_
* `Bibliography <_static/paper/dire_short.bib>`_

Reproducibility
---------------

The repository stores self-contained JSON benchmark logs under
``data/archived/json_logs``. The command ``make figures`` regenerates
manuscript figures from those logs. The Revision 3 remote workflow reruns the
full benchmark suite on an NVIDIA GPU, fetches one verified compact bundle, and
rebuilds the local publication artifacts from that bundle.

The Revision 3 profile adds the public 1,306,127-cell 10x mouse-brain dataset
and 723,457-paper arXiv corpus. ``scripts/reproduce_revision3.sh`` performs the
version-pinned H100 run followed by a no-GPU CPU topology audit;
``scripts/fetch_revision3_results.sh`` verifies and downloads both the compact
archive and its matching successful audit; and ``scripts/build_revision3_local.sh``
regenerates the publication-ready figures, tables, and result summaries under
``generated/revision3`` without a GPU. The fetcher changes ``CURRENT`` only
after verification and rendering succeed, removes superseded Revision 3
downloads/extractions, and enforces a single-bundle hash and byte-identity
cleanliness audit. The local renderer also replaces the manuscript's canonical
``pics/`` tree from the bundle's fresh twenty-run GPU and ten-run CPU
small-suite records rather than reusing the older archived runtime figure. It
records the source bundle
and exact source/output membership, byte counts, and SHA-256 hashes in
``pics/render_manifest.json``; the cleanliness audit checks those records
against disk. ``generated/revision3/render_manifest.json`` likewise declares
the exact generated-file membership, byte counts, and hashes so stale tables
or figures cannot coexist silently with the current render. The fetched bundle
also contains the six--dataset reference suite with 20 GPU and 10 CPU fits per
configuration. Its atlas diagnostic is evaluated on every fit over paired,
fixed 1,000-row index subsets by calling the GPU rank-based local-kNN atlas
evaluator directly; the records require ``prefer_ripser=false``. Default DiRe
and the separately labelled source-distributed topology preset are both
retained. The renderer reports raw means, sample standard deviations, minima,
every paired gap, and continuous paired effect sizes rather than ordinal
ranks. The fetched bundle also contains the
fixed topology reference subsets and diagrams: the historical Betti-curve DTW,
diameter-normalized bottleneck distances, and a 99th-percentile scale
sensitivity analysis are therefore regenerated from one version-pinned
workflow rather than transcribed into the manuscript. Nested
1,000/2,000/4,000-row subsets test diagnostic subset-size stability, separately
from three full-layout seeds.

Funding
-------

The archived benchmark logs were produced on Lambda.ai/Lambda Labs GPU infrastructure. The broader project has also been supported by the Google Cloud Research Award number GCP19980904.

.. toctree::
   :maxdepth: 1

   paper
   figures
   revision3
