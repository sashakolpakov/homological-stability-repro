from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from latex_to_sphinx import convert, latex_to_text, render_table  # noqa: E402


def test_latex_accents_are_preserved_as_unicode() -> None:
    rendered = latex_to_text(
        r"Poli{\v c}ar, Stra{\v z}ar, B{\"o}hm, and Fran{\c c}ois",
        {},
        {},
    )
    assert rendered == "Poličar, Stražar, Böhm, and François"


def test_compact_inline_math_has_valid_rst_boundaries() -> None:
    rendered = latex_to_text(
        r"4.720$\pm$0.044; 20$\rightarrow$19; "
        r"\texttt{n_components}$=d$; 10{} subsets",
        {},
        {},
    )

    assert rendered == (
        r"4.720\ :math:`\pm`\ 0.044; "
        r"20\ :math:`\rightarrow`\ 19; "
        r"``n_components``\ :math:`=d`; 10 subsets"
    )
    assert "{}" not in rendered


def test_generated_macros_inputs_and_tables_are_converted(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "values.tex").write_text(
        "\\newcommand{\\LargeN}{1,306,127}\n",
        encoding="utf-8",
    )
    (generated / "table.tex").write_text(
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Method & Rows \\\\\n"
        "\\midrule\n"
        "DiRe & \\LargeN \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )
    latex = tmp_path / "paper.tex"
    latex.write_text(
        "\\documentclass{article}\n"
        "\\input{generated/values.tex}\n"
        "\\title{Fixture}\n"
        "\\author{Author}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "The run contains \\LargeN observations.\n"
        "\\begin{table}\n"
        "\\centering\n"
        "\\input{generated/table.tex}\n"
        "\\caption{Measured scale.}\\label{tab:scale}\n"
        "\\end{table}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    bib = tmp_path / "paper.bib"
    bib.write_text("", encoding="utf-8")
    output = tmp_path / "paper.rst"

    convert(latex, bib, output)

    rendered = output.read_text(encoding="utf-8")
    assert "The run contains 1,306,127 observations." in rendered
    assert ".. list-table:: Table 1. Measured scale." in rendered
    assert "   * - Method" in rendered
    assert "     - Rows" in rendered
    assert "   * - DiRe" in rendered
    assert "     - 1,306,127" in rendered


def test_table_shortstack_linebreak_is_not_parsed_as_a_new_row() -> None:
    block = (
        "\\begin{table}\n"
        "\\begin{tabular}{ll}\n"
        "Metric & Paired gaps \\\\\n"
        "$\\beta_0$ & "
        "\\shortstack[l]{+1.0, +2.0\\\\+3.0, +4.0} \\\\\n"
        "\\end{tabular}\n"
        "\\caption{Fixture.}\n"
        "\\end{table}\n"
    )

    rendered = "\n".join(render_table(block, "1", {}, {}))

    assert rendered.count("   * - ") == 2
    assert "     - +1.0, +2.0 +3.0, +4.0" in rendered
    assert "{+1.0" not in rendered


def test_standalone_longtable_is_rendered_and_referenced(tmp_path: Path) -> None:
    latex = tmp_path / "paper.tex"
    latex.write_text(
        "\\documentclass{article}\n"
        "\\title{Fixture}\n"
        "\\author{Author}\n"
        "\\begin{document}\n"
        "The complete list is given in Table~\\ref{tab:markers}.\n"
        "\\begin{longtable}{@{}rrp{0.62\\linewidth}@{}}\n"
        "\\caption{Released marker list.}\\label{tab:markers}\\\\\n"
        "\\toprule\n"
        "Cluster & Cells & Markers \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "1 & 100 & GeneA, GeneB \\\\\n"
        "2 & 50 & GeneC, GeneD \\\\\n"
        "\\bottomrule\n"
        "\\end{longtable}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    bib = tmp_path / "paper.bib"
    bib.write_text("", encoding="utf-8")
    output = tmp_path / "paper.rst"

    convert(latex, bib, output)

    rendered = output.read_text(encoding="utf-8")
    assert "The complete list is given in Table 1." in rendered
    assert ".. list-table:: Table 1. Released marker list." in rendered
    assert "   * - Cluster" in rendered
    assert "     - Cells" in rendered
    assert "     - Markers" in rendered
    assert "   * - 1" in rendered
    assert "     - GeneA, GeneB" in rendered
    assert "longtable" not in rendered
