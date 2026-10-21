"""Accuracy, error, and calibration metrics computed from predicted probabilities."""
import torch


def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    return (probs.argmax(1) == y).float().mean().item()


def error_rate(probs: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - accuracy(probs, y)


def nll(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Mean negative log-likelihood of the true class (cross-entropy on probabilities)."""
    return -probs.gather(1, y[:, None]).clamp_min(1e-12).log().mean().item()


def reliability_bins(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict[str, torch.Tensor]:
    """Per-bin mean confidence, accuracy and count over equal-width confidence bins."""
    confidence, predicted = probs.max(1)
    correct = (predicted == y).float()
    # Bin b holds confidences in (b/bins, (b+1)/bins]; confidence >= 1/C > 0.
    index = (confidence * bins).ceil().long().clamp(1, bins) - 1
    count = torch.bincount(index, minlength=bins).float()
    conf_sum = torch.bincount(index, weights=confidence, minlength=bins)
    correct_sum = torch.bincount(index, weights=correct, minlength=bins)
    safe = count.clamp_min(1)
    return {"confidence": conf_sum / safe, "accuracy": correct_sum / safe, "count": count}


def ece(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
    table = reliability_bins(probs, y, bins)
    weights = table["count"] / table["count"].sum()
    return (weights * (table["accuracy"] - table["confidence"]).abs()).sum().item()


def pairwise_disagreement(samples: torch.Tensor) -> torch.Tensor:
    """(S, S) fraction of inputs where members i and j predict a different top-1 class, from (S, N, C)."""
    predicted = samples.argmax(2)
    return (predicted[:, None, :] != predicted[None, :, :]).float().mean(2)


def split_metrics(probs: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """The loss/accuracy pair recorded per epoch in history.csv."""
    return {"loss": nll(probs, y), "accuracy": accuracy(probs, y)}
