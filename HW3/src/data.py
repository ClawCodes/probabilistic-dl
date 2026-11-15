from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MC_RTT_FILE = DATA / "mc_rtt" / "sub-Indy_desc-train_behavior+ecephys.nwb"


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


# ---------------------------------------------------------------------------
# MC_RTT
# ---------------------------------------------------------------------------

def read_mc_rtt(path: Path = MC_RTT_FILE, max_units: int = 48) -> dict:
    """Read the NWB file once: behavior at 1 kHz, NLB trial chunks, and spike times
    of the `max_units` most active held-in units (ascending activity, last = most active)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m src.download_mc_rtt")
    with h5py.File(path, "r") as nwb:
        velocity = nwb["processing/behavior/finger_vel/data"][:, :2].astype(np.float32)
        position = nwb["processing/behavior/finger_pos/data"][:, :2].astype(np.float32)
        target = nwb["processing/behavior/target_pos/data"][:].astype(np.float32)
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
    return {
        "velocity": velocity,                       # (T, 2) mm/s, 1 kHz from t = 0
        "speed": np.linalg.norm(velocity, axis=1),
        "position": position,                       # (T, 2) mm
        "target": target,                           # (T, 2) mm
        "starts": starts, "stops": stops, "split": split,
        "units": units,
        "spikes": [spike_times[unit_starts[u]:spike_ends[u]] for u in units],
    }


def window_centers(raw: dict, window_ms: int = 200):
    """Per NLB chunk: the time of peak finger speed (seconds) such that a centered
    window fits inside the chunk, and the velocity direction there in 8 bins of 45 deg."""
    centers, labels = [], []
    for start, stop in zip(raw["starts"], raw["stops"]):
        lo, hi = int(np.ceil(start * 1000)), int(np.floor(stop * 1000))
        # A centered window must fit entirely in its trial.
        lo += int(window_ms // 2); hi -= int(window_ms // 2)
        if hi <= lo:
            continue
        peak = lo + int(np.argmax(raw["speed"][lo:hi]))
        vx, vy = raw["velocity"][peak]
        labels.append(int(np.floor(np.mod(np.arctan2(vy, vx), 2 * np.pi) / (2 * np.pi / 8))))
        centers.append(peak / 1000.0)
    return np.asarray(centers), np.asarray(labels)


def smoothed_rates(spike_times: np.ndarray, t: np.ndarray, sigma_s: float) -> np.ndarray:
    """Gaussian-smoothed firing rate (Hz) at times `t`: a unit-area Gaussian of width
    `sigma_s` on every spike, summed. Spikes outside the window still contribute."""
    t = np.asarray(t, dtype=np.float64)
    near = spike_times[(spike_times > t.min() - 4 * sigma_s) & (spike_times < t.max() + 4 * sigma_s)]
    if len(near) == 0:
        return np.zeros(len(t))
    z = (t[:, None] - near[None, :]) / sigma_s
    return np.exp(-0.5 * z**2).sum(axis=1) / (sigma_s * np.sqrt(2 * np.pi))


def binned_rates(spike_times: np.ndarray, edges: np.ndarray, bin_ms: int) -> np.ndarray:
    """Spike count in each bin between consecutive `edges`, converted to Hz."""
    return np.diff(np.searchsorted(spike_times, edges)) * (1000.0 / bin_ms)


def window_rates(spikes, center: float, bin_ms: int, window_ms: int, smooth_sigma_ms: float | None):
    """(units, bins) firing rates for one window centred on `center` (s).
    Smoothed rates are read at bin centres; without smoothing, spikes are counted per bin."""
    n_bins = window_ms // bin_ms
    edges = center + np.linspace(-window_ms / 2000.0, window_ms / 2000.0, n_bins + 1)
    if smooth_sigma_ms is None:
        return np.stack([binned_rates(st, edges, bin_ms) for st in spikes])
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return np.stack([smoothed_rates(st, bin_centers, smooth_sigma_ms / 1000.0) for st in spikes])


def load_mc_rtt_windows(
    path: Path = MC_RTT_FILE, bin_ms: int = 20, window_ms: int = 200,
    max_units: int = 48, smooth_sigma_ms: float | None = 30,
):
    """Build movement-peak population-rate windows from the official NWB file.

    Each observation is a 200-ms window centered on peak finger speed in one NLB
    chunk, with one log1p firing rate per unit and 20-ms bin. By default rates are
    Gaussian-smoothed (sigma 30 ms) and read at bin centres; `smooth_sigma_ms=None`
    gives raw binned spike counts instead. The condition is the peak-velocity
    direction in eight angular bins. The source file's train/validation split is kept.
    """
    if window_ms % bin_ms:
        raise ValueError("window_ms must be divisible by bin_ms")
    raw = read_mc_rtt(path, max_units)
    centers, labels = window_centers(raw, window_ms)
    split = raw["split"]
    # Every accepted trial corresponds to the same position in `split` here.
    # All 600-ms source trials leave room for the selected 200-ms window.
    if len(centers) != len(split):
        raise RuntimeError("Unexpected MC_RTT trial duration; update split alignment logic.")

    features = np.stack([
        window_rates(raw["spikes"], center, bin_ms, window_ms, smooth_sigma_ms) for center in centers
    ]).astype(np.float32)
    features = np.log1p(features).reshape(len(centers), -1)
    return features, one_hot(labels, 8), labels, split == "train", {
        "bin_ms": bin_ms,
        "window_ms": window_ms,
        "n_units": len(raw["units"]),
        "n_trials": len(features),
        "n_train_trials": int((split == "train").sum()),
        "n_validation_trials": int((split == "val").sum()),
        "smoothing": (f"gaussian, sigma {smooth_sigma_ms:g} ms, sampled at bin centers"
                      if smooth_sigma_ms is not None else "none (spike counts per bin)"),
        "rate_transform": "log1p(Hz)",
    }
