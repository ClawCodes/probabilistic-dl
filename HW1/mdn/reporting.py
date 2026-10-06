"""Utilities for turning experiment CSV summaries into LaTeX table rows."""

import csv
from pathlib import Path


TABLE_COLUMNS = (
    "model",
    "k",
    "train_nll",
    "test_nll",
    "train_rmse",
    "test_rmse",
)


def _latex_escape(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    text = str(value)
    for original, escaped in replacements.items():
        text = text.replace(original, escaped)
    return text


def write_latex_table(csv_path, tex_path=None):
    """Read a summary CSV and emit the complete table shown in the report."""
    csv_path = Path(csv_path)
    tex_path = Path(tex_path) if tex_path else csv_path.with_name(
        f"{csv_path.stem}_table.tex")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    rendered_rows = [
        r"\begin{tabular}{lccccc}" + "\n",
        r"  \toprule" + "\n",
        r"  Model & $K$ & Train NLL & Test NLL & Train RMSE & Test RMSE \\" + "\n",
        r"  \midrule" + "\n",
    ]
    for row in rows:
        values = [_latex_escape(row[column]) for column in TABLE_COLUMNS]
        rendered_rows.append("  " + " & ".join(values) + r" \\" + "\n")
    rendered_rows.extend([
        r"  \bottomrule" + "\n",
        r"\end{tabular}" + "\n",
    ])

    tex_path.write_text("".join(rendered_rows), encoding="utf-8")
    return tex_path
