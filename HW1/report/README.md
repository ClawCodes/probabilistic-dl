# MDN assignment report

The report source is `main.tex`. Put every report image in `figs/`; the LaTeX
source is configured so figure commands use filenames only.

The experiment scripts write these figure files directly to `figs/`:

- `synthetic_raw_data.png`
- `density_grid_2d.png`
- `synthetic_training.png`
- `synthetic_holdout_vs_k.png`
- `wind_turbine_features.png`
- `density_grid_wind_turbine.png`
- `wind_turbine_training.png`

Missing images appear as labeled placeholders, so the draft can still compile.
Replace each `\answerbox{...}{...}` command with the corresponding prose and
run the experiments to populate the result tables automatically.

Run the experiments from the repository root:

```powershell
python -m mdn.compare_k
python -m mdn.train_wind_turbine
python -m mdn.wind_turbine_eda
```

The scripts export model summaries and full training histories to `csvs/`:

- `synthetic_summary.csv`
- `synthetic_training_history.csv`
- `wind_turbine_summary.csv`
- `wind_turbine_training_history.csv`
- `wind_turbine_data_summary.csv`

For each `*_summary.csv`, the same experiment also creates a compact
`*_summary_table.tex` table directly from that CSV. The report inputs those
generated tables, so the CSV remains the single source of truth and no values
need to be copied into `main.tex`. No additional CSV package is required.

From this directory, compile with:

```powershell
pdflatex main.tex
pdflatex main.tex
```

The second pass resolves references. Submit the resulting `main.pdf`.
