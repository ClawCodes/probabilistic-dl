# HW2: Predictive uncertainty

This project compares uncertainty estimates from a standard deep ensemble and
(later) MC dropout. It starts with a deliberately simple 2-D classification
problem and is structured so the same experiment code works for high-dimensional
real data.

## Quick start

From `HW2`, install the dependencies with your preferred environment manager,
then run:

```powershell
python -m src.run --dataset two_moons --method ensemble --epochs 200 --members 5
```

All generated plots are written directly to `latex/figs/`; scalar results and
per-member histories go to `outputs/`. Compile the write-up from `latex/`.

## Adding a real/high-dimensional dataset

Implement a loader returning `DatasetBundle` in `src/data.py`, register it in
`load_dataset`, and expose any needed command-line options in `src/run.py`.
The bundle contains train/test tensors, class labels, and a `feature_shape`;
models flatten all non-batch feature dimensions, so tabular data, images, or
embeddings work without changing the ensemble code.

## Methods

- `ensemble` is fully implemented: independently initialized MLPs trained with
  different random seeds on the same data, L2/Gaussian-prior regularization,
  averaged predictive probabilities, and standard entropy / mutual-information
  uncertainty summaries.
- `mc_dropout` is intentionally a clear extension point. Its CLI path explains
  what is missing rather than silently running the wrong experiment. Implement
  `MCDropoutPredictor.fit` and `predict_proba` in `src/methods/mc_dropout.py`;
  it already shares the `PredictiveModel` interface used by the runner.

Use a fixed seed when comparing methods, and tune both methods using validation
data only once a validation split is added for the selected real dataset.
