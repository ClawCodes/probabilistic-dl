# HW3: Conditional VAEs for neural population activity

This project trains the same conditional VAE (cVAE) on a 2-D toy neural manifold and on MC_RTT, a public macaque primary-motor-cortex reaching dataset, and compares latent-space sizes on both.

It follows the lightweight HW2 pattern: source code runs directly from the homework directory and figures are written directly into `latex/figs/`.

## Workflow

Run commands from `HW3/`. With the project environment (`pyproject.toml` / `uv.lock`):

```bash
uv run python -m src.download_mc_rtt           # only if data/mc_rtt is absent (~50 MB)
uv run python -m src.run --experiment all      # both experiments
uv run python -m src.run --experiment toy      # toy only
uv run python -m src.run --experiment real     # MC_RTT only (each well under a minute on CPU)
```

Or with the existing `ode-rnn` Conda environment on Windows (PyTorch, matplotlib, and `h5py`):

```powershell
$python = 'C:\Users\nicla\anaconda3\envs\ode-rnn\python.exe'
& $python -m src.run --experiment all
```

Each run writes:
- `outputs/config.json`: hyperparameters.
- `outputs/metrics.json`: held-out loss, reconstruction, KL (total and per latent dimension), and active-dimension count for each latent size. A partial run only updates its own experiment's section.
- `outputs/weights/*.pt`: one checkpoint per latent size.
- All figures listed below, in `latex/figs/`.

Compile `latex/main.tex` from `latex/` after the run. The report source is intentionally an outline of bullets so the prose and final claims remain yours.

## Code layout

| File | Contents |
|---|---|
| `src/data.py` | Toy data generator, and the MC_RTT reader and preprocessing (`read_mc_rtt`, `window_centers`, `smoothed_rates`, `window_rates`, `load_mc_rtt_windows`) |
| `src/model.py` | `ConditionalVAE` (MLP encoder/decoder, diagonal-Gaussian latent), `kl_per_dimension`, `vae_loss` |
| `src/run.py` | Training loop, latent-size sweeps, metrics, and all model-result figures |
| `src/preprocessing_figs.py` | Figures showing how raw MC_RTT data becomes network input |
| `src/download_mc_rtt.py` | Downloads the NWB file from DANDI (dandiset 000129) |

## Model and training

- **cVAE.** The encoder takes the input concatenated with a one-hot direction label (8 classes) and outputs a mean and log-variance per latent dimension. The decoder takes a latent sample concatenated with the same label. Both are two hidden layers wide with SiLU activations.
- **Loss.** Reconstruction MSE (mean over input features) + β × KL(q(z | x, c) ‖ N(0, I)). The **KL is summed over latent dimensions**, so β means the same thing for every latent size in a sweep.
- **Optimizer.** AdamW, learning rate 2e-3, batch size 256, seed 7. Training and validation loss, reconstruction and KL are recorded every epoch. The validation pass saves and restores the random-number state, so it doesn't change training.
- **Active dimensions.** A latent dimension counts as active if its KL, averaged over held-out inputs, is above 0.01 nats.

| | Toy | MC_RTT |
|---|---|---|
| Latent sizes | 1, 2, 4, 8 | 2, 8, 16 |
| Hidden width | 128 | 256 |
| Epochs | 260 | 50 |
| β | 0.015 | 0.001875 |
| Split | random 80/20, standardized with train statistics | NLB-provided 810/270, standardized with train statistics |

The 50 MC_RTT epochs are deliberate. Validation reconstruction reaches its minimum around epochs 20–35 and then rises, so the training curves show the early stage of overfitting while the final models stay near their best validation score.

## Data

### Toy data
A synthetic 2-D dataset (`make_toy_arcs`): eight noisy crescents, one per direction label, 600 points each. Each crescent is the same shape rotated by its direction's angle (0°, 45°, …) plus Gaussian noise. The axes are plain features x₁, x₂, not firing rates. The data is a small, visible test of whether the cVAE can learn a direction-dependent low-dimensional manifold: points vary mainly along each crescent (one intrinsic dimension) and a little across its thickness.

### MC_RTT
Monkey Indy, Sabes lab (O'Doherty et al.), as packaged by the Neural Latents Benchmark (NLB):
- Self-paced reaches to random targets on a grid, with no trial structure, recorded continuously for about 10.8 minutes.
- 130 sorted M1 units (98 held-in, 32 held-out) and finger kinematics at 1 kHz.
- NLB cuts the recording into 1,080 back-to-back **600 ms chunks** ("trials"), each labelled `train` (810) or `val` (270). The chunks are not aligned to reaches.

### MC_RTT preprocessing (`load_mc_rtt_windows`)
1. **Units.** Keep the 48 held-in units with the most spikes in the session.
2. **Window centre.** In each NLB chunk, find the time of peak finger speed within the chunk's middle 400 ms, so a 200 ms window centred there fits inside the chunk.
3. **Direction label.** The angle of finger velocity at that moment, discretized into 8 sectors of 45° (0 = rightward/up-right, counter-clockwise).
4. **Firing rates by Gaussian smoothing.** For each unit, place a unit-area Gaussian (σ = 30 ms) on every spike and sum them, giving a continuous firing-rate estimate in Hz. Spikes just outside the window contribute, which avoids edge effects. The curve is read at the 10 bin centres (−90, −70, …, +90 ms from peak speed).
5. **Transform.** `log1p(rate)`, then flatten to 48 units × 10 bins = 480 features.
6. **Split and scaling.** Keep the NLB `train`/`val` split as provided. Standardize each feature with training-set mean and standard deviation.

**Why smoothing:** in a 20 ms bin a neuron almost never fires more than once. Raw spike counts were 86% zeros and otherwise almost always exactly one spike, which made the input essentially unpredictable coin flips. The cVAE's best validation reconstruction on counts was about 0.98, against a ≈1.0 "predict the average" baseline. With smoothing it reaches 0.74–0.87 depending on latent size. `load_mc_rtt_windows(smooth_sigma_ms=None)` still produces the old spike-count inputs.

**Known limitations:**
- Windows come from NLB's arbitrary chunks rather than detected reaches. Some windows sit on slow movement or on a reach split across a chunk boundary (about 29% have peak speed below 50 mm/s).
- Validation chunks are interleaved with, and sometimes directly next to, training chunks. Combined with the smoothing kernel, this makes validation scores somewhat optimistic. The provided split is kept for comparability with NLB.

## Figures

All figures are written to `latex/figs/` by `python -m src.run`. "Held-out" means the toy test split or the NLB validation split.

### Toy experiment (`toy_experiment` in `src/run.py`)

| File | What it shows | How it is produced |
|---|---|---|
| `toy_data_and_samples.png` | All toy observations next to 250 generated samples per direction | z = 2 model; samples decode z ~ N(0, I) with each direction's one-hot label, then are un-standardized |
| `toy_samples_by_latent.png` | Held-out observations next to generated samples for z = 1, 2, 4, 8, on shared axes; titles give active dims and test reconstruction | One model per latent size, 250 samples per direction each |
| `toy_latent.png` | Encoder means of the held-out points in the 2-D latent space, coloured by direction | z = 2 model, posterior mean μ for each held-out point |
| `toy_training.png` | Train (solid) and validation (dashed) loss, reconstruction MSE and total KL per epoch; one row per latent size | Per-epoch history recorded in `train()` |
| `toy_latent_sweep.png` | Held-out loss, reconstruction MSE and total KL against latent size, with active-dimension counts | Final metrics of each model (`plot_latent_sweep`) |

### MC_RTT preprocessing (`plot_mc_rtt_preprocessing` in `src/preprocessing_figs.py`)

These use the same functions as the training data, so they show exactly what the network receives.

| File | What it shows | How it is produced |
|---|---|---|
| `mc_rtt_raw_overview.png` | 6 s of the recording: finger speed (target onsets in blue), finger x/y position, and the 48-unit spike raster. Dotted lines are NLB chunk edges, grey marks a validation chunk, and orange bands are the 200 ms windows cut from each chunk | Raw NWB data from chunk 40 onward; window centres from `window_centers` |
| `mc_rtt_preprocessing_unit.png` | One unit around one fast reach: (a) raw spike times; (b) the Gaussian-smoothed rate with the 10 bin-centre values fed to the network | First window after the overview stretch with peak speed > 250 mm/s and peak not at the chunk edge; the busiest unit there; `smoothed_rates` |
| `mc_rtt_preprocessing_window.png` | The same window for all 48 units: (a) raw spike raster with 20 ms bin edges; (b) the 48 × 10 network input, log1p of the smoothed rate | `window_rates` with σ = 30 ms |
| `mc_rtt_direction_labels.png` | Finger velocity at every window centre, coloured by its 45° direction sector; the dashed circle marks 50 mm/s, inside which labels are unreliable | Velocity at each `window_centers` time |

### MC_RTT model results (`real_experiment` in `src/run.py`)

| File | What it shows | How it is produced |
|---|---|---|
| `mc_rtt_training.png` | Train (solid) and validation (dashed) loss, reconstruction MSE and total KL per epoch; one row per latent size (2, 8, 16) | Per-epoch history recorded in `train()` |
| `mc_rtt_latent_sweep.png` | Held-out loss, reconstruction MSE and total KL against latent size, with active-dimension counts | Final metrics of each model (`plot_latent_sweep`) |
| `mc_rtt_pca_by_latent.png` | Held-out windows and generated samples on the top two principal components of the held-out data, coloured by direction; one panel per latent size | PCA fitted on the 270 held-out windows (log1p units); 160 samples per direction from each model, projected onto the same axes |
| `mc_rtt_samples_by_latent.png` | For one direction (the one with most held-out windows): average window and 3 individual windows, held-out vs each latent size, on one colour scale | Held-out windows of that direction vs 160 samples per model |
| `mc_rtt_observed_and_reconstructions.png` | Three held-out windows (top) and their reconstructions (bottom) | z = 8 model; encoder → sampled z → decoder, un-standardized |
| `mc_rtt_samples.png` | One generated window per direction (0–7) | z = 8 model; first of 160 samples per direction |
| `mc_rtt_direction_profiles.png` | Population-average activity over time (all units averaged) per direction: held-out (black) vs generated (orange, dashed) | z = 8 model; mean over held-out windows of that direction vs mean over its 160 samples. The x-axis shows bin start times |
| `mc_rtt_latent.png` | Encoder means of the held-out windows in the 8-D latent space, projected to 2-D with PCA and coloured by direction | z = 8 model, posterior mean μ |

## References

The dataset source, VAE reference, and NLB context are listed in `latex/references.bib`. Items marked `VERIFY` need a human source check before submission.
