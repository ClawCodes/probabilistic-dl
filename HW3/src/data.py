from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float32)[labels.astype(int)]


def make_toy_arcs(seed: int = 7, n_per_direction: int = 600):
    """Eight noisy, rotated crescents representing direction-conditioned neural states."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), n_per_direction)
    angles = labels * (2 * np.pi / 8)
    phase = rng.uniform(-1.05, 1.05, size=len(labels))
    radius = 1.25 + 0.17 * rng.normal(size=len(labels))
    # A crescent with along-manifold variation, rotated into a movement direction.
    base = np.column_stack((radius * np.cos(phase), 0.60 * np.sin(phase) + 0.18 * phase**2))
    rotation = np.stack((np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)), axis=1).reshape(-1, 2, 2)
    x = np.einsum("nij,nj->ni", rotation, base) + 0.055 * rng.normal(size=base.shape)
    return x.astype(np.float32), one_hot(labels, 8), labels


def _direction_labels(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.linalg.norm(velocity, axis=1)
    angle = np.mod(np.arctan2(velocity[:, 1], velocity[:, 0]), 2 * np.pi)
    labels = np.floor(angle / (2 * np.pi / 8)).astype(int)
    return labels, speed


def load_mc_rtt_windows(
    data_dir: Path = DATA / "mc_rtt", bin_ms: int = 20, window_ms: int = 200,
    max_units: int = 48,
):
    """Build movement-peak population-rate windows from the official NWB file.

    Each observation is a 200-ms window centered on peak hand speed in one
    reach. The condition is the peak-velocity direction in eight angular bins.
    The source file already supplies an explicit train/validation trial split;
    it is retained rather than randomly splitting correlated windows.
    """
    path = data_dir / "sub-Indy_desc-train_behavior+ecephys.nwb"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m src.download_mc_rtt")
    if window_ms % bin_ms:
        raise ValueError("window_ms must be divisible by bin_ms")
    bins_per_window = window_ms // bin_ms
    half_window = window_ms / 2000.0
    with h5py.File(path, "r") as nwb:
        velocity = nwb["processing/behavior/finger_vel/data"][:, :2].astype(np.float32)
        starts = nwb["intervals/trials/start_time"][:]
        stops = nwb["intervals/trials/stop_time"][:]
        split = np.asarray(nwb["intervals/trials/split"][:], dtype="S").astype(str)
        heldout = nwb["units/heldout"][:].astype(bool)
        spike_times = nwb["units/spike_times"][:]
        spike_ends = nwb["units/spike_times_index"][:]

    # Use held-in units only, matching the standard NLB analysis convention.
    unit_starts = np.r_[0, spike_ends[:-1]]
    counts = spike_ends - unit_starts
    candidate_units = np.flatnonzero(~heldout)
    units = candidate_units[np.argsort(counts[candidate_units])[-max_units:]]
    unit_spikes = [spike_times[unit_starts[u]:spike_ends[u]] for u in units]

    centers, labels = [], []
    for start, stop in zip(starts, stops):
        lo, hi = int(np.ceil(start * 1000)), int(np.floor(stop * 1000))
        # A 200-ms centered window must fit entirely in its trial.
        lo += int(window_ms // 2); hi -= int(window_ms // 2)
        if hi <= lo:
            continue
        segment = velocity[lo:hi]
        peak = lo + int(np.argmax(np.linalg.norm(segment, axis=1)))
        vx, vy = velocity[peak]
        labels.append(int(np.floor(np.mod(np.arctan2(vy, vx), 2 * np.pi) / (2 * np.pi / 8))))
        centers.append(peak / 1000.0)
    centers = np.asarray(centers)
    labels = np.asarray(labels)
    # Every accepted trial corresponds to the same position in `split` here.
    # All 600-ms source trials leave room for the selected 200-ms window.
    if len(centers) != len(split):
        raise RuntimeError("Unexpected MC_RTT trial duration; update split alignment logic.")

    offsets = np.linspace(-half_window, half_window, bins_per_window + 1)
    features = np.empty((len(centers), len(units), bins_per_window), dtype=np.float32)
    for trial, center in enumerate(centers):
        edges = center + offsets
        for unit, times in enumerate(unit_spikes):
            # searchsorted is much faster and lower-memory than a 1-kHz spike matrix.
            features[trial, unit] = np.diff(np.searchsorted(times, edges)) * (1000.0 / bin_ms)
    features = np.log1p(features).reshape(len(centers), -1)
    return features, one_hot(labels, 8), labels, split == "train", {
        "bin_ms": bin_ms,
        "window_ms": window_ms,
        "n_units": len(units),
        "n_trials": len(features),
        "n_train_trials": int((split == "train").sum()),
        "n_validation_trials": int((split == "val").sum()),
        "rate_transform": "log1p(Hz)",
    }
