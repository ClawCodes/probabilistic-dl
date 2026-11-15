from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import ROOT, load_mc_rtt_windows, make_toy_arcs
from .model import ConditionalVAE, kl_per_dimension, vae_loss
from .preprocessing_figs import plot_mc_rtt_preprocessing


FIGS = ROOT / "latex" / "figs"
OUTPUTS = ROOT / "outputs"
COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:-1]
# KL weights for the summed-over-latent-dims KL. Each matches the effective
# weight of the earlier per-dim-averaged loss at one reference latent size
# (toy: z=2, 0.03/2; MC_RTT: z=8, 0.015/8) and is now fixed across the sweep.
TOY_BETA = 0.015
MC_RTT_BETA = 0.015 / 8
TOY_LATENTS, TOY_EPOCHS, TOY_HIDDEN = (1, 2, 4, 8), 260, 128
MC_RTT_LATENTS, MC_RTT_EPOCHS, MC_RTT_HIDDEN = (2, 8, 16), 50, 256
MC_RTT_SMOOTH_SIGMA_MS = 30  # Gaussian spike smoothing; rates read at 20 ms bin centres
ACTIVE_KL_THRESHOLD = 0.01  # nats; a latent dim below this is treated as unused


def seed_everything(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def split_and_standardize(x, c, labels, seed=7):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x)); split = int(.8 * len(x))
    train_i, test_i = order[:split], order[split:]
    mean, scale = x[train_i].mean(0), x[train_i].std(0) + 1e-5
    return (x[train_i] - mean) / scale, c[train_i], labels[train_i], (x[test_i] - mean) / scale, c[test_i], labels[test_i], mean, scale


def standardize_source_split(x, c, labels, is_train):
    """Keep MC_RTT's supplied trial split intact and standardize using train only."""
    mean, scale = x[is_train].mean(0), x[is_train].std(0) + 1e-5
    return (
        (x[is_train] - mean) / scale, c[is_train], labels[is_train],
        (x[~is_train] - mean) / scale, c[~is_train], labels[~is_train], mean, scale,
    )


@torch.no_grad()
def evaluate(model, x, c, beta):
    """One forward pass on held-out data; returns (loss, recon, kl) plus encoder outputs."""
    model.eval()
    reconstructed, mu, logvar = model(x, c)
    loss, rec, kl = vae_loss(reconstructed, x, mu, logvar, beta)
    return (loss.item(), rec.item(), kl.item()), mu, logvar, reconstructed


def train(x_train, c_train, x_test, c_test, latent_dim, epochs, hidden=128, beta=TOY_BETA, seed=7):
    """Train one cVAE. History columns: train loss/recon/KL, then validation loss/recon/KL."""
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalVAE(x_train.shape[1], c_train.shape[1], latent_dim, hidden).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(c_train)), batch_size=min(256, len(x_train)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    x_val, c_val = torch.from_numpy(x_test).to(device), torch.from_numpy(c_test).to(device)
    history = []
    for epoch in range(epochs):
        model.train(); values = []
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            reconstruction, mu, logvar = model(x, c)
            loss, rec, kl = vae_loss(reconstruction, x, mu, logvar, beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            values.append((loss.item(), rec.item(), kl.item()))
        # The validation pass draws random z; save/restore the RNG so it does not
        # change the training randomness (shuffling, reparameterization noise).
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        val_values, *_ = evaluate(model, x_val, c_val, beta)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        history.append((*np.mean(values, axis=0), *val_values))
    (loss, rec, kl), mu, logvar, reconstructed = evaluate(model, x_val, c_val, beta)
    kl_dims = kl_per_dimension(mu, logvar).mean(0).cpu().numpy()
    metrics = {
        "test_loss": loss,
        "test_reconstruction": rec,
        "test_kl": kl,  # total KL, summed over latent dims
        "test_kl_per_dim": [round(float(v), 4) for v in kl_dims],
        "active_dims": int((kl_dims > ACTIVE_KL_THRESHOLD).sum()),
    }
    return model, np.array(history), metrics, mu.cpu().numpy(), reconstructed.cpu().numpy()


def save(fig, name, tight=True):
    FIGS.mkdir(parents=True, exist_ok=True)
    if tight: fig.tight_layout()
    fig.savefig(FIGS / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def save_with_colorbar(fig, image, name, label="log1p firing rate"):
    """Lay out the panels, then put one shared colorbar in its own strip on the right so it can't overlap them."""
    fig.tight_layout(rect=(0, 0, .9, 1))
    fig.colorbar(image, cax=fig.add_axes((.92, .15, .015, .7)), label=label)
    save(fig, name, tight=False)


def save_checkpoint(model, name, latent_dim, x_dim, cond_dim, metadata):
    """Persist each fitted model and the minimal shape metadata for reuse."""
    weights = OUTPUTS / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "x_dim": x_dim,
        "cond_dim": cond_dim,
        "metadata": metadata,
    }, weights / name)


def plot_training(histories, name, title, beta):
    """Rows = latent sizes; columns = total loss, reconstruction, KL. Train solid, validation dashed."""
    dims = list(histories)
    fig, axes = plt.subplots(len(dims), 3, figsize=(11, 2.3 * len(dims)), sharex=True, sharey="col", squeeze=False)
    columns = (f"loss (recon + {beta:g}·KL)", "reconstruction MSE", "KL (nats, summed over z)")
    for row, zdim in enumerate(dims):
        h = histories[zdim]
        for col in range(3):
            ax = axes[row, col]
            ax.plot(h[:, col], color="C0", label="train")
            ax.plot(h[:, col + 3], color="C1", linestyle="--", label="validation")
            if row == 0: ax.set_title(columns[col], fontsize=10)
            if row == len(dims) - 1: ax.set_xlabel("epoch")
        axes[row, 0].set_ylabel(f"z = {zdim}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    save(fig, name)


def plot_latent_sweep(results, name, title):
    """Held-out loss, reconstruction and total KL (labelled with active dims) against latent size."""
    dims = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, label in zip(axes, ("test_loss", "test_reconstruction", "test_kl"),
                              ("test loss", "test reconstruction MSE", "test KL (nats, summed)")):
        ax.plot(dims, [results[d][key] for d in dims], marker="o")
        ax.set(xlabel="latent dimensionality", ylabel=label, xticks=dims)
    for d in dims:
        axes[2].annotate(f"{results[d]['active_dims']}/{d} active", (d, results[d]["test_kl"]),
                         textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[2].margins(y=.2)
    fig.suptitle(title, fontsize=11)
    save(fig, name)


def fit_pca2(x):
    """Top-2 principal axes of x; returns (mean, components) so other data can share the projection."""
    mean = x.mean(0); _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:2]


def pca2(x):
    mean, components = fit_pca2(x)
    return (x - mean) @ components.T


def toy_experiment():
    x, c, labels = make_toy_arcs()
    xtr, ctr, ltr, xte, cte, lte, mean, scale = split_and_standardize(x, c, labels)
    x_test_raw = xte * scale + mean
    n_samples = 250
    sample_conditions = np.repeat(np.eye(8, dtype=np.float32), n_samples, axis=0)
    sample_labels = np.repeat(np.arange(8), n_samples)
    results, all_samples, histories = {}, {}, {}
    for zdim in TOY_LATENTS:
        model, history, metrics, latent, reconstructed = train(xtr, ctr, xte, cte, zdim, TOY_EPOCHS, hidden=TOY_HIDDEN, beta=TOY_BETA)
        device = next(model.parameters()).device
        with torch.no_grad(): samples = model.sample(torch.from_numpy(sample_conditions).to(device)).cpu().numpy()
        samples = samples * scale + mean
        all_samples[zdim], histories[zdim], results[zdim] = samples, history, metrics
        save_checkpoint(model, f"toy_z{zdim}.pt", zdim, xtr.shape[1], ctr.shape[1], {"seed": 7, "beta": TOY_BETA, "epochs": TOY_EPOCHS})
        if zdim == 2:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
            for k in range(8):
                axes[0].scatter(x[labels == k, 0], x[labels == k, 1], s=4, alpha=.35, color=COLORS[k])
                axes[1].scatter(samples[k*250:(k+1)*250, 0], samples[k*250:(k+1)*250, 1], s=4, alpha=.35, color=COLORS[k])
            axes[0].set_title("Toy observations"); axes[1].set_title("cVAE samples, requested direction")
            for ax in axes: ax.set(xlabel="simulated neuron 1", ylabel="simulated neuron 2", aspect="equal")
            save(fig, "toy_data_and_samples.png")
            fig, ax = plt.subplots(figsize=(4.5, 4));
            for k in range(8): ax.scatter(latent[lte == k, 0], latent[lte == k, 1], s=6, alpha=.45, color=COLORS[k], label=f"dir. {k}")
            ax.set(title="Toy encoded means (2-D latent)", xlabel="$z_1$", ylabel="$z_2$"); ax.legend(ncol=2, fontsize=8, frameon=False); save(fig, "toy_latent.png")

    plot_training(histories, "toy_training.png", "Toy cVAE training by latent size", TOY_BETA)

    # Generated samples for every latent size next to the observations, on shared axes.
    dims = list(results)
    fig, axes = plt.subplots(1, len(dims) + 1, figsize=(3.4 * (len(dims) + 1), 3.8), sharex=True, sharey=True)
    for k in range(8):
        axes[0].scatter(x_test_raw[lte == k, 0], x_test_raw[lte == k, 1], s=4, alpha=.45, color=COLORS[k])
    axes[0].set_title("held-out observations")
    for ax, zdim in zip(axes[1:], dims):
        for k in range(8):
            pts = all_samples[zdim][sample_labels == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=.35, color=COLORS[k])
        r = results[zdim]
        ax.set_title(f"samples, z = {zdim}\nactive dims {r['active_dims']}/{zdim}, test recon {r['test_reconstruction']:.4f}", fontsize=10)
    for ax in axes: ax.set(xlabel="simulated neuron 1", aspect="equal")
    axes[0].set_ylabel("simulated neuron 2")
    fig.suptitle(f"Toy cVAE samples by latent size (β = {TOY_BETA:g})", fontsize=11, y=1.06)
    save(fig, "toy_samples_by_latent.png")

    plot_latent_sweep(results, "toy_latent_sweep.png", "Toy latent-size comparison")
    return {"toy": {"beta": TOY_BETA, "epochs": TOY_EPOCHS, **results}}


def real_experiment():
    plot_mc_rtt_preprocessing(save, COLORS, smooth_sigma_ms=MC_RTT_SMOOTH_SIGMA_MS)
    x, c, labels, source_train, metadata = load_mc_rtt_windows(smooth_sigma_ms=MC_RTT_SMOOTH_SIGMA_MS)
    xtr, ctr, ltr, xte, cte, lte, mean, scale = standardize_source_split(x, c, labels, source_train)
    n_samples = 160
    condition = np.repeat(np.eye(8, dtype=np.float32), n_samples, axis=0)
    sample_labels = np.repeat(np.arange(8), n_samples)
    results, histories, samples_by_z = {}, {}, {}
    for zdim in MC_RTT_LATENTS:
        model, history, metrics, latent, reconstructed = train(xtr, ctr, xte, cte, zdim, MC_RTT_EPOCHS, hidden=MC_RTT_HIDDEN, beta=MC_RTT_BETA)
        results[zdim], histories[zdim] = metrics, history
        save_checkpoint(model, f"mc_rtt_z{zdim}.pt", zdim, xtr.shape[1], ctr.shape[1], {**metadata, "seed": 7, "beta": MC_RTT_BETA, "epochs": MC_RTT_EPOCHS, "hidden_dim": MC_RTT_HIDDEN})
        device = next(model.parameters()).device
        with torch.no_grad(): sampled = model.sample(torch.from_numpy(condition).to(device)).cpu().numpy()
        samples_by_z[zdim] = sampled * scale + mean
        if zdim == 8:
            saved = (latent, reconstructed * scale + mean)
    latent, reconstruction = saved
    observed = xte * scale + mean
    samples = samples_by_z[8]
    units, bins = metadata["n_units"], metadata["window_ms"] // metadata["bin_ms"]

    # --- Existing z = 8 figures (referenced by main.tex) ---
    lo, hi = np.quantile(np.concatenate((observed.ravel(), reconstruction.ravel(), samples.ravel())), (.01, .99))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        source = observed[i] if i < 3 else reconstruction[i - 3]
        image = ax.imshow(source.reshape(units, bins), aspect="auto", cmap="magma", origin="lower", vmin=lo, vmax=hi)
        kind = "observed" if i < 3 else "reconstructed"
        ax.set_title(f"{kind} trial {i % 3 + 1}"); ax.set(xlabel="time bin", ylabel="unit")
    save_with_colorbar(fig, image, "mc_rtt_observed_and_reconstructions.png")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        sample = samples[i * n_samples]
        image = ax.imshow(sample.reshape(units, bins), aspect="auto", cmap="magma", origin="lower", vmin=lo, vmax=hi)
        ax.set_title(f"sampled activity; dir. {i}"); ax.set(xlabel="time bin", ylabel="unit")
    save_with_colorbar(fig, image, "mc_rtt_samples.png")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    times = np.arange(bins) * metadata["bin_ms"] - metadata["window_ms"] / 2
    for direction, ax in enumerate(axes.ravel()):
        real_profile = observed[lte == direction].reshape(-1, units, bins).mean(axis=(0, 1))
        generated_profile = samples[sample_labels == direction].reshape(-1, units, bins).mean(axis=(0, 1))
        ax.plot(times, real_profile, color="black", label="held-out" if direction == 0 else None)
        ax.plot(times, generated_profile, color="#d95f02", linestyle="--", label="generated" if direction == 0 else None)
        ax.set(title=f"direction {direction}", xlabel="ms from peak speed", ylabel="mean log1p rate")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "mc_rtt_direction_profiles.png")
    proj = pca2(latent)
    fig, ax = plt.subplots(figsize=(4.5, 4));
    for k in range(8): ax.scatter(proj[lte == k, 0], proj[lte == k, 1], s=8, alpha=.45, color=COLORS[k], label=f"dir. {k}")
    ax.set(title="MC_RTT latent means (8-D, PCA view)", xlabel="PC 1", ylabel="PC 2"); ax.legend(ncol=2, fontsize=8, frameon=False); save(fig, "mc_rtt_latent.png")

    # --- Latent-size sweep figures ---
    plot_training(histories, "mc_rtt_training.png", "MC_RTT cVAE training by latent size", MC_RTT_BETA)
    plot_latent_sweep(results, "mc_rtt_latent_sweep.png", "MC_RTT latent-size comparison")

    # PCA of held-out windows; generated samples projected onto the same two axes.
    pca_mean, components = fit_pca2(observed)
    dims = list(samples_by_z)
    fig, axes = plt.subplots(1, len(dims) + 1, figsize=(3.6 * (len(dims) + 1), 3.8), sharex=True, sharey=True)
    proj = (observed - pca_mean) @ components.T
    for k in range(8):
        axes[0].scatter(proj[lte == k, 0], proj[lte == k, 1], s=8, alpha=.6, color=COLORS[k], label=f"dir. {k}")
    axes[0].set_title(f"held-out windows (n = {len(observed)})", fontsize=10)
    axes[0].legend(ncol=2, fontsize=7, frameon=False)
    for ax, zdim in zip(axes[1:], dims):
        proj = (samples_by_z[zdim] - pca_mean) @ components.T
        for k in range(8):
            ax.scatter(proj[sample_labels == k, 0], proj[sample_labels == k, 1], s=4, alpha=.35, color=COLORS[k])
        ax.set_title(f"samples, z = {zdim} ({n_samples}/direction)\nactive dims {results[zdim]['active_dims']}/{zdim}", fontsize=10)
    for ax in axes: ax.set(xlabel="PC 1")
    axes[0].set_ylabel("PC 2")
    fig.suptitle("MC_RTT: held-out and generated windows on the held-out data's top two principal components", fontsize=11, y=1.06)
    save(fig, "mc_rtt_pca_by_latent.png")

    # Heatmaps for one direction: average window + 3 individual windows, held-out vs each latent size.
    direction = int(np.argmax(np.bincount(lte, minlength=8)))
    real_dir = observed[lte == direction]
    rows = [("held-out", real_dir)] + [(f"z = {z}", samples_by_z[z][sample_labels == direction]) for z in dims]
    lo, hi = np.quantile(np.concatenate([observed.ravel()] + [s.ravel() for s in samples_by_z.values()]), (.01, .99))
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.3 * len(rows)), sharex=True, sharey=True)
    for r, (label, windows) in enumerate(rows):
        panels = [(f"{label}: mean of {len(windows)}", windows.mean(0))] + [(f"{label}: {'trial' if r == 0 else 'sample'} {i + 1}", windows[i]) for i in range(3)]
        for col, (panel_title, values) in enumerate(panels):
            ax = axes[r, col]
            image = ax.imshow(values.reshape(units, bins), aspect="auto", cmap="magma", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(panel_title, fontsize=9)
            if col == 0: ax.set_ylabel("unit")
            if r == len(rows) - 1: ax.set_xlabel("time bin")
    fig.suptitle(f"MC_RTT direction {direction}: held-out windows vs generated samples by latent size", fontsize=11)
    save_with_colorbar(fig, image, "mc_rtt_samples_by_latent.png")

    return {"mc_rtt": {"beta": MC_RTT_BETA, "epochs": MC_RTT_EPOCHS, "metadata": metadata, "metrics": results}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=("toy", "real", "all"), default="all"); args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    metrics_path = OUTPUTS / "metrics.json"
    summary = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config = {
        "seed": 7,
        "kl": "summed over latent dims, averaged over batch",
        "toy": {"latent_dims": list(TOY_LATENTS), "epochs": TOY_EPOCHS, "hidden_dim": TOY_HIDDEN, "beta": TOY_BETA},
        "mc_rtt": {"latent_dims": list(MC_RTT_LATENTS), "epochs": MC_RTT_EPOCHS, "hidden_dim": MC_RTT_HIDDEN, "beta": MC_RTT_BETA, "bin_ms": 20, "window_ms": 200, "units": 48, "smooth_sigma_ms": MC_RTT_SMOOTH_SIGMA_MS},
    }
    (OUTPUTS / "config.json").write_text(json.dumps(config, indent=2))
    if args.experiment in ("toy", "all"): summary.update(toy_experiment())
    if args.experiment in ("real", "all"): summary.update(real_experiment())
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
