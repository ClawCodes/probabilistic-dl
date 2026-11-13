# HW3: Conditional VAEs for neural population activity

This project trains the same conditional VAE on a 2-D toy neural manifold and MC_RTT, a public macaque primary-motor-cortex reaching dataset.

It deliberately follows the lightweight HW2 pattern: source code runs directly from the homework directory and figures are written directly into `latex/figs/`. It does **not** create a project virtual environment. The existing `ode-rnn` Conda environment is used because it already contains PyTorch and plotting dependencies; `h5py` was added there to read the NWB data.

## Workflow

Run commands from `HW3/`:

```powershell
$python = 'C:\Users\nicla\anaconda3\envs\ode-rnn\python.exe'

# Optional only if data/mc_rtt is absent; downloads the public ~50 MB training file.
& $python -m src.download_mc_rtt

# Train both experiments, save metrics under outputs/, and save report figures under latex/figs/.
& $python -m src.run --experiment all

# Fast checks for either half.
& $python -m src.run --experiment toy
& $python -m src.run --experiment real
```

Compile `latex/main.tex` from `latex/` after the run. The report source is intentionally an outline of bullets so the prose and final claims remain yours.

## Real-data representation

- Each observation is a 200-ms activity window centered on that reach's peak hand speed.
- It contains log-transformed firing rates for the 48 most active held-in M1 units in ten 20-ms bins.
- The cVAE condition is the peak-velocity direction, discretized into eight angular bins.
- The source NWB file supplies 810 training trials and 270 validation trials; the code preserves that split rather than randomly splitting correlated samples.

The dataset source, VAE reference, and NLB context are listed in `latex/references.bib`. Items marked `VERIFY` need a human source check before submission.
