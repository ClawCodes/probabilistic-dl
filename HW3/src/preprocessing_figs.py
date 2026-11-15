"""Figures that show how raw MC_RTT data becomes the network's input.

Built on the same helpers as `load_mc_rtt_windows`, so every panel shows exactly
what the training pipeline computes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import read_mc_rtt, smoothed_rates, window_centers, window_rates

INK, MUTED, WINDOW, CURVE, POINT = "#222222", "#9a9a9a", "#d95f02", "#1b6ca8", "#d95f02"


def plot_mc_rtt_preprocessing(save, colors, bin_ms=20, window_ms=200, smooth_sigma_ms=30,
                              overview_chunk=40, overview_seconds=6.0):
    """Write the four preprocessing figures. `save(fig, name)` writes a figure; `colors` are the 8 direction colors."""
    raw = read_mc_rtt()
    centers, labels = window_centers(raw, window_ms)
    speed, starts, split, spikes = raw["speed"], raw["starts"], raw["split"], raw["spikes"]
    half = window_ms / 2000.0

    # ---------- 1. Recording overview ----------
    t0 = starts[overview_chunk]; t1 = t0 + overview_seconds
    ms = np.arange(int(t0 * 1000), int(t1 * 1000))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2.2]})
    for ax in axes:
        for i, s in enumerate(starts):
            if t0 <= s < t1:
                ax.axvline(s, color=MUTED, lw=.6, ls=":")
                if split[i] == "val": ax.axvspan(s, s + .6, color="black", alpha=.06, lw=0)
        for c in centers[(centers >= t0) & (centers < t1)]:
            ax.axvspan(c - half, c + half, color=WINDOW, alpha=.18, lw=0)
    target_onsets = (np.flatnonzero(np.any(np.diff(raw["target"], axis=0) != 0, axis=1)) + 1) / 1000
    axes[0].plot(ms / 1000, speed[ms], color=INK, lw=1)
    for tc in target_onsets[(target_onsets >= t0) & (target_onsets < t1)]:
        axes[0].axvline(tc, color=CURVE, lw=1.2)
    axes[0].set_ylabel("finger speed (mm/s)")
    axes[0].set_title("Finger speed. Blue: new target appears. Dotted: NLB 600 ms chunk edges (gray = validation chunk). "
                      f"Orange: the {window_ms} ms window cut from each chunk, centred on its peak speed.", loc="left", fontsize=9)
    axes[1].plot(ms / 1000, raw["position"][ms, 0], color=INK, lw=1, label="x")
    axes[1].plot(ms / 1000, raw["position"][ms, 1], color=MUTED, lw=1, label="y")
    axes[1].set_ylabel("finger position (mm)"); axes[1].legend(frameon=False, loc="upper right")
    for row, st in enumerate(spikes):
        sel = st[(st >= t0) & (st < t1)]
        axes[2].vlines(sel, row + .1, row + .9, color=INK, lw=.6)
    axes[2].set(ylabel=f"unit ({len(spikes)} most active held-in; top = most active)", xlabel="time in recording (s)", ylim=(0, len(spikes)))
    axes[2].set_title("Spike raster: each tick is one spike", loc="left", fontsize=9)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    save(fig, "mc_rtt_raw_overview.png")

    # ---------- Pick one clean window: fast reach, peak not at the chunk edge ----------
    chunk = next(i for i, c in enumerate(centers)
                 if c > t1 and speed[int(round(c * 1000))] > 250 and half + .01 < c - starts[i] < .6 - half - .01)
    center = centers[chunk]
    edges = center + np.linspace(-half, half, window_ms // bin_ms + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    smoothed = window_rates(spikes, center, bin_ms, window_ms, smooth_sigma_ms)
    rel = lambda t: (np.asarray(t) - center) * 1000

    # ---------- 2. One unit: raw spikes -> smoothed rate -> samples at bin centres ----------
    unit = int(np.argmax([np.sum((st > center - .3) & (st < center + .3)) for st in spikes[:-8]]))
    st = spikes[unit]; local = st[(st > center - .3) & (st < center + .3)]
    t = np.linspace(center - .3, center + .3, 1200)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [.5, 1.6]})
    for ax in axes:
        ax.axvspan(-window_ms / 2, window_ms / 2, color=WINDOW, alpha=.08, lw=0)
        for e in edges: ax.axvline(rel(e), color=MUTED, lw=.5, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].vlines(rel(local), 0, 1, color=INK, lw=1.2)
    axes[0].set(yticks=[])
    axes[0].set_title(f"(a) Raw spike times of one unit around a reach (each tick = one spike; shaded = {window_ms} ms window, dotted = {bin_ms} ms bins)", loc="left", fontsize=9)
    axes[1].plot(rel(t), smoothed_rates(st, t, smooth_sigma_ms / 1000), color=CURVE, lw=1.6, label=f"Gaussian-smoothed rate (σ = {smooth_sigma_ms} ms)")
    axes[1].plot(rel(bin_centers), smoothed[unit], "o", color=POINT, ms=7, label="value at each bin centre → network input")
    axes[1].vlines(rel(local), 0, 3, color=INK, lw=1)
    axes[1].set(ylabel="rate (Hz)", xlabel="ms from peak finger speed")
    axes[1].set_title(f"(b) A Gaussian on every spike, summed, read at the {window_ms // bin_ms} bin centres", loc="left", fontsize=9)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "mc_rtt_preprocessing_unit.png")

    # ---------- 3. Whole window: raw raster -> smoothed network input ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for row, s in enumerate(spikes):
        sel = s[(s >= center - half) & (s < center + half)]
        axes[0].vlines(rel(sel), row + .1, row + .9, color=INK, lw=.8)
    for e in edges: axes[0].axvline(rel(e), color=MUTED, lw=.5, ls=":")
    axes[0].set(xlim=(-window_ms / 2, window_ms / 2), ylim=(0, len(spikes)), xlabel="ms from peak speed", ylabel="unit")
    axes[0].set_title(f"(a) raw spikes, {window_ms} ms window", fontsize=10)
    axes[0].spines[["top", "right"]].set_visible(False)
    image = axes[1].imshow(np.log1p(smoothed), aspect="auto", origin="lower", cmap="magma", vmin=0,
                           extent=(-window_ms / 2, window_ms / 2, 0, len(spikes)))
    axes[1].set(xlabel="ms from peak speed", yticks=[])
    axes[1].set_title(f"(b) network input\nlog1p(smoothed rate, σ = {smooth_sigma_ms} ms) at bin centres", fontsize=10)
    fig.colorbar(image, ax=axes[1], label="log1p(rate in Hz)", shrink=.85)
    fig.suptitle(f"One window (peak speed {speed[int(round(center * 1000))]:.0f} mm/s): "
                 f"{len(spikes)} units × {window_ms // bin_ms} bins = {len(spikes) * (window_ms // bin_ms)} inputs", fontsize=10, y=1.04)
    save(fig, "mc_rtt_preprocessing_window.png", tight=False)

    # ---------- 4. Direction labels from velocity at the window centre ----------
    idx = np.round(centers * 1000).astype(int)
    v = raw["velocity"][idx]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lim = np.abs(v).max() * 1.08
    for k in range(8):
        ang = k * np.pi / 4
        ax.plot([0, lim * 1.5 * np.cos(ang)], [0, lim * 1.5 * np.sin(ang)], color=MUTED, lw=.7, ls=":")
        mid = ang + np.pi / 8
        ax.text(.9 * lim * np.cos(mid), .9 * lim * np.sin(mid), f"{k}", ha="center", va="center", fontsize=11, color=INK, weight="bold")
        sel = labels == k
        ax.scatter(v[sel, 0], v[sel, 1], s=9, alpha=.6, color=colors[k], lw=0)
    slow = np.linalg.norm(v, axis=1) < 50
    ax.add_patch(plt.Circle((0, 0), 50, fill=False, color=INK, lw=1, ls="--"))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="finger velocity x (mm/s)", ylabel="finger velocity y (mm/s)")
    ax.set_title(f"Direction label = 45° sector of the velocity at each window centre\n"
                 f"{len(v)} windows; dashed circle = 50 mm/s ({100 * slow.mean():.0f}% of windows inside)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "mc_rtt_direction_labels.png")
