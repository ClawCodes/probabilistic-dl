# HW2: Predictive uncertainty

This project compares uncertainty estimates from a standard deep ensemble and
(later) MC dropout, on a 2-D toy problem (two moons) and Fashion-MNIST.

## Quick start

From `HW2`:

```bash
python -m src.pipeline                                   # new run, default config
python -m src.pipeline --set two_moons.epochs=50 --ensemble-sizes 1 2 5
python -m src.pipeline --dir run_1                       # retrain run_1 from its config.json
python -m src.pipeline --dir run_1 --step plot           # replot run_1 from saved weights
```

Pipeline steps are `train -> plot`; `--step` picks where to start. Every run trains
every configured method on every configured dataset.

- Without `--dir`, a new `outputs/run_N/` is created and the config (defaults,
  then `--config file.json`, `--seed`, `--methods`, `--ensemble-sizes`, then
  `--set DATASET.FIELD=VALUE` overrides) is written to `run_N/config.json`.
- With `--dir`, the run's `config.json` is the only source of truth; config
  flags are rejected so artifacts never drift from their config.
- `train` rewrites a method's training and plot artifacts; `plot` rewrites only
  plot artifacts, loading weights from disk.

Ensemble members are seeded `seed + k`, so member `k` is identical for any
ensemble size. A run trains `max(ensemble_sizes)` members once, and the size
sweep evaluates prefixes of that ensemble.

## Run layout

```
outputs/run_N/
  config.json
  <dataset>/<method>/
    weights/member_XX.pt   # train
    history.csv            # train: loss/accuracy per member per epoch
    members.csv            # train: final per-member loss/accuracy
    metrics.json           # train: full-model train/test accuracy and uncertainty
    size_sweep.csv         # plot
    figures/               # plot: size_sweep.png, surface.png (2-D datasets only)
```

Figures that compare methods on one dataset belong at the `<dataset>/` level.

## Adding a dataset or method

- Dataset: return a `DatasetBundle` from `load_dataset` in `src/data.py` and add
  the name and default hyperparameters in `src/config.py`.
- Method: implement `PredictiveModel` (`fit`, `predict(x, n_samples)`, `save`,
  `load`) and register it in `build_method` in `src/methods/__init__.py`.
  `mc_dropout` is scaffolded in `src/methods/mc_dropout.py`; build its
  `Prediction` with `prediction_from_samples`.

`src/old/` holds the previous `run.py`/`sweep.py` for reference only; they no
longer run against the current modules.
