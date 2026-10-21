# HW2: Predictive uncertainty

This project compares uncertainty estimates from a standard deep ensemble and
MC dropout, on a 2-D toy problem (two moons) and Fashion-MNIST.

## Configuration

Defaults live in `src/config.py`: `ExperimentConfig` holds the experiment-level
options, and `default_datasets()` holds the per-dataset `DatasetConfig` values.
Override them per run with command-line flags, or pass a JSON file with
`--config` (same keys as a run's `config.json`). Omitted top-level keys keep
their defaults. A `datasets` block replaces the dataset list, and any field left
out of a dataset falls back to the generic `DatasetConfig` default, not that
dataset's entry in `default_datasets()`.

| Key | Meaning |
|---|---|
| `seed` | Base random seed (ensemble member `m` uses `seed + m`) |
| `methods` | Methods to train: `ensemble`, `mc_dropout` |
| `ensemble_sizes` | Sizes k compared in the sweep; M = max members / MC passes are trained once |
| `subset_draws` | Max k-member subsets drawn per size for error bars |
| `ece_bins` | Number of confidence bins for ECE and reliability diagrams |
| `datasets.<name>` | Per-dataset training settings: `epochs`, `batch_size`, `learning_rate`, `weight_decay` (L2), `hidden_dim`, `dropout` (used by `mc_dropout` only) |

Datasets: `two_moons`, `fashion_mnist` (downloaded to `data/` on first use), `mnist`.

## Running the pipeline

All commands run from `HW2/`. Every run covers every configured method on every
configured dataset.

The pipeline has two ordered steps, `train -> plot`. `--step` picks where to
start, and every later step runs too.

- **`--step train`** (the default) trains the models and writes the training
  artifacts: weights, `history.csv` and `metrics.json`. The plot step then runs
  on the models still in memory. Without `--dir` this creates a new run; with
  `--dir` it retrains that run in place.
- **`--step plot`** skips training, so it needs trained models to load. It must
  be given `--dir` pointing at an existing run. It rebuilds the models from that
  run's `config.json`, loads their saved weights, and rewrites only the plot
  artifacts (`size_sweep*.csv` and `figures/`). `--step plot` without `--dir`
  is an error.

```bash
# New run with the defaults from src/config.py
python -m src.pipeline

# New run starting from a JSON config, e.g. a previous run's config
python -m src.pipeline --config outputs/run_1/config.json

# New run with overrides (applied on top of the defaults / --config)
python -m src.pipeline --seed 3 --methods ensemble --ensemble-sizes 1 2 5
python -m src.pipeline --set fashion_mnist.epochs=50 --set two_moons.dropout=0.2

# Quick smoke test
python -m src.pipeline --set two_moons.epochs=5 --set fashion_mnist.epochs=1 --ensemble-sizes 1 2

# Existing run: retrain from its config.json, or replot from saved weights
python -m src.pipeline --dir run_1
python -m src.pipeline --dir run_1 --step plot
```

- Without `--dir`, a new `outputs/run_N/` is created and the final config is
  written to `run_N/config.json`.
- With `--dir`, the run's `config.json` is the only source of truth; config
  flags are rejected so artifacts never drift from their config.
- Retraining always re-runs the plot step, so figures never lag behind the weights.

## Artifacts

```
outputs/run_N/
  config.json                # the exact config of this run
  <dataset>/
    function_space.csv       # plot: PC1/PC2 of each member's flattened test probabilities (shared PCA)
    disagreement.csv         # plot: per member pair, fraction of test inputs with a different top-1 class
    figures/
      function_space.png     # members of all methods in one PCA + per-method disagreement heatmaps
  <dataset>/<method>/
    weights/                 # train: member_XX.pt (ensemble) or model.pt (mc_dropout)
    history.csv              # train: member, epoch, split, loss, accuracy
                             #   (member = index/pass or "ensemble"; train and test splits)
    metrics.json             # train: full-model accuracy/error/NLL/ECE/entropy/MI, train and test
    size_sweep.csv           # plot: metrics for each k-member subset (k, draw, split)
    size_sweep_summary.csv   # plot: mean/std over subsets per (k, split)
    figures/
      size_sweep.png         # error, NLL, ECE, entropy, MI vs. k with error bars
      training_curves.png    # train/test loss and accuracy per epoch (member band + ensemble)
      reliability.png        # test reliability diagram: single member vs. full ensemble
      surface.png            # predictive probability and MI over the input plane (2-D only)
```

Ensemble members are seeded `seed + k`, so member `k` is identical for any
ensemble size. A run trains `max(ensemble_sizes)` members once, and the size
sweep evaluates subsets of that ensemble.

Error bars in the size sweep come from resampling k-member subsets of the trained
members: every member at k=1, every combination when there are at most
`subset_draws` of them, otherwise `subset_draws` random subsets (seeded). Subsets
overlap more as k approaches M, so the bars shrink partly by construction. The
test set is used for monitoring only, never for model selection.

Figures that compare methods on one dataset belong at the `<dataset>/` level.

## Adding a dataset or method

- Dataset: return a `DatasetBundle` from `load_dataset` in `src/data.py` and add
  the name and default hyperparameters in `src/config.py`.
- Method: implement `PredictiveModel` (`fit(x, y, x_test, y_test)` returning
  history rows, `sample_probabilities(x, n_samples)`, `save`, `load`) and
  register it in `build_method` in `src/methods/__init__.py`.

## Methods

- `ensemble`: M independently seeded MLPs (`seed + m`), no dropout.
- `mc_dropout`: one MLP trained with dropout (`datasets.<name>.dropout`). At
  prediction time dropout stays on, and each of T = `max(ensemble_sizes)`
  stochastic passes is treated as a "member" in the history, sweep and figures.
  Prediction-time masks use a forked, fixed-seed RNG, so evaluation never
  changes training and replots are reproducible.

Both use the same `weight_decay` (L2 / Gaussian prior).

`src/old/` holds the previous `run.py`/`sweep.py` for reference only; they no
longer run against the current modules.
