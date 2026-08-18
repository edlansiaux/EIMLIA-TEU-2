"""Quadratic-weighted kappa, bootstrap CIs, and threshold (not non-inferiority) tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def bootstrap_kappa_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = quadratic_kappa(y_true[idx[i]], y_pred[idx[i]])
    point = quadratic_kappa(y_true, y_pred)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def threshold_test(
    samples: np.ndarray,
    threshold: float,
) -> dict:
    """One-sided test H0: kappa <= threshold vs H1: kappa > threshold.

    This is a threshold/superiority formulation, not a non-inferiority test.
    Non-inferiority would compare two arms with a pre-specified margin on a
    difference; here a single arm is tested against a fixed deployment bar.
    """
    p_above = float(np.mean(samples > threshold))
    mean = float(np.mean(samples))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    crosses = lo < threshold < hi
    return {
        "mean": mean,
        "ci95": (float(lo), float(hi)),
        "threshold": threshold,
        "p_exceed": p_above,
        "ci_crosses_threshold": bool(crosses),
        "test_name": "one-sided threshold/superiority vs fixed bar",
        "not_noninferiority": True,
    }


def critical_undertriage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of true Level 1-2 cases assigned Level >= 3."""
    crit = y_true <= 2
    if not np.any(crit):
        return 0.0
    return float(np.mean(y_pred[crit] >= 3))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels=(1, 2, 3, 4, 5)) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
