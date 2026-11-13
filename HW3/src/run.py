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
from .model import ConditionalVAE, vae_loss


FIGS = ROOT / "latex" / "figs"
OUTPUTS = ROOT / "outputs"
COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:-1]


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


def train(x_train, c_train, x_test, c_test, latent_dim, epochs, hidden=128, beta=.03, seed=7):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalVAE(x_train.shape[1], c_train.shape[1], latent_dim, hidden).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(c_train)), batch_size=min(256, len(x_train)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    history = []
    for epoch in range(epochs):
        model.train(); values = []
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            reconstruction, mu, logvar = model(x, c)
            loss, rec, kl = vae_loss(reconstruction, x, mu, logvar, beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            values.append((loss.item(), rec.item(), kl.item()))
        history.append(np.mean(values, axis=0))
    model.eval()
    with torch.no_grad():
        x, c = torch.from_numpy(x_test).to(device), torch.from_numpy(c_test).to(device)
        reconstructed, mu, logvar = model(x, c)
        loss, rec, kl = vae_loss(reconstructed, x, mu, logvar, beta)
    return model, np.array(history), {"test_loss": loss.item(), "test_reconstruction": rec.item(), "test_kl": kl.item()}, mu.cpu().numpy(), reconstructed.cpu().numpy()


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIGS / name, dpi=180, bbox_inches="tight"); plt.close(fig)


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


def plot_loss(history, name, title, beta):
    fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.plot(history[:, 0], label="ELBO loss"); ax.plot(history[:, 1], label="reconstruction"); ax.plot(history[:, 2] * beta, label=rf"{beta:g} × KL")
    ax.set(xlabel="epoch", ylabel="loss", title=title); ax.legend(frameon=False); save(fig, name)


def pca2(x):
    centered = x - x.mean(0); _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def toy_experiment():
    x, c, labels = make_toy_arcs()
    xtr, ctr, ltr, xte, cte, lte, mean, scale = split_and_standardize(x, c, labels)
    results = {}
    for zdim in (1, 2, 4, 8):
        model, history, metrics, latent, reconstructed = train(xtr, ctr, xte, cte, zdim, 260)
        results[zdim] = metrics
        save_checkpoint(model, f"toy_z{zdim}.pt", zdim, xtr.shape[1], ctr.shape[1], {"seed": 7, "beta": .03, "epochs": 260})
        if zdim == 2:
            device = next(model.parameters()).device
            sample_conditions = np.repeat(np.eye(8, dtype=np.float32), 250, axis=0)
            with torch.no_grad(): samples = model.sample(torch.from_numpy(sample_conditions).to(device)).cpu().numpy()
            samples = samples * scale + mean
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
            plot_loss(history, "toy_training.png", "Toy cVAE training (2-D latent)", beta=.03)
    fig, ax = plt.subplots(figsize=(5, 3.4)); dims = list(results); ax.plot(dims, [results[d]["test_reconstruction"] for d in dims], marker="o")
    ax.set(xlabel="latent dimensionality", ylabel="test reconstruction MSE", title="Toy latent-size comparison", xticks=dims); save(fig, "toy_latent_sweep.png")
    return {"toy": results}


def real_experiment():
    x, c, labels, source_train, metadata = load_mc_rtt_windows()
    xtr, ctr, ltr, xte, cte, lte, mean, scale = standardize_source_split(x, c, labels, source_train)
    results = {}; saved = None
    for zdim in (2, 8, 16):
        model, history, metrics, latent, reconstructed = train(xtr, ctr, xte, cte, zdim, 180, hidden=256, beta=.015)
        results[zdim] = metrics
        save_checkpoint(model, f"mc_rtt_z{zdim}.pt", zdim, xtr.shape[1], ctr.shape[1], {**metadata, "seed": 7, "beta": .015, "epochs": 180, "hidden_dim": 256})
        if zdim == 8:
            device = next(model.parameters()).device
            condition = np.repeat(np.eye(8, dtype=np.float32), 160, axis=0)
            with torch.no_grad(): sampled = model.sample(torch.from_numpy(condition).to(device)).cpu().numpy()
            saved = (history, latent, xte * scale + mean, reconstructed * scale + mean, sampled * scale + mean)
    history, latent, observed, reconstruction, samples = saved
    units, bins = metadata["n_units"], metadata["window_ms"] // metadata["bin_ms"]
    lo, hi = np.quantile(np.concatenate((observed.ravel(), reconstruction.ravel(), samples.ravel())), (.01, .99))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        source = observed[i] if i < 3 else reconstruction[i - 3]
        image = ax.imshow(source.reshape(units, bins), aspect="auto", cmap="magma", origin="lower", vmin=lo, vmax=hi)
        kind = "observed" if i < 3 else "reconstructed"
        ax.set_title(f"{kind} trial {i % 3 + 1}"); ax.set(xlabel="time bin", ylabel="unit")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.82, label="log1p firing rate")
    save(fig, "mc_rtt_observed_and_reconstructions.png")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        sample = samples[i * 160]
        image = ax.imshow(sample.reshape(units, bins), aspect="auto", cmap="magma", origin="lower", vmin=lo, vmax=hi)
        ax.set_title(f"sampled activity; dir. {i}"); ax.set(xlabel="time bin", ylabel="unit")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.82, label="log1p firing rate")
    save(fig, "mc_rtt_samples.png")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    times = np.arange(bins) * metadata["bin_ms"] - metadata["window_ms"] / 2
    for direction, ax in enumerate(axes.ravel()):
        real_profile = observed[lte == direction].reshape(-1, units, bins).mean(axis=(0, 1))
        generated_profile = samples[direction * 160:(direction + 1) * 160].reshape(-1, units, bins).mean(axis=(0, 1))
        ax.plot(times, real_profile, color="black", label="held-out" if direction == 0 else None)
        ax.plot(times, generated_profile, color="#d95f02", linestyle="--", label="generated" if direction == 0 else None)
        ax.set(title=f"direction {direction}", xlabel="ms from peak speed", ylabel="mean log1p rate")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "mc_rtt_direction_profiles.png")
    proj = pca2(latent)
    fig, ax = plt.subplots(figsize=(4.5, 4));
    for k in range(8): ax.scatter(proj[lte == k, 0], proj[lte == k, 1], s=8, alpha=.45, color=COLORS[k], label=f"dir. {k}")
    ax.set(title="MC_RTT latent means (8-D, PCA view)", xlabel="PC 1", ylabel="PC 2"); ax.legend(ncol=2, fontsize=8, frameon=False); save(fig, "mc_rtt_latent.png")
    plot_loss(history, "mc_rtt_training.png", "MC_RTT cVAE training (8-D latent)", beta=.015)
    fig, ax = plt.subplots(figsize=(5, 3.4)); dims = list(results); ax.plot(dims, [results[d]["test_reconstruction"] for d in dims], marker="o")
    ax.set(xlabel="latent dimensionality", ylabel="test reconstruction MSE", title="MC_RTT latent-size comparison", xticks=dims); save(fig, "mc_rtt_latent_sweep.png")
    return {"mc_rtt": {"metadata": metadata, "metrics": results}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=("toy", "real", "all"), default="all"); args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    metrics_path = OUTPUTS / "metrics.json"
    summary = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config = {
        "seed": 7,
        "toy": {"latent_dims": [1, 2, 4, 8], "epochs": 260, "hidden_dim": 128, "beta": .03},
        "mc_rtt": {"latent_dims": [2, 8, 16], "epochs": 180, "hidden_dim": 256, "beta": .015, "bin_ms": 20, "window_ms": 200, "units": 48},
    }
    (OUTPUTS / "config.json").write_text(json.dumps(config, indent=2))
    if args.experiment in ("toy", "all"): summary.update(toy_experiment())
    if args.experiment in ("real", "all"): summary.update(real_experiment())
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
