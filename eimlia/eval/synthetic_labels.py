"""Construct ordinal 5-class predictions matching a target quadratic kappa.

Used when patient-level expert labels cannot be shared. The generator is
calibrated so that published point estimates are recovered, and bootstrap
CIs / confusion patterns are computed on the resulting synthetic sample
of the same size as the completed expert subset (n=400).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from eimlia.config import FRENCH_MIX, FRENCH_LEVELS
from eimlia.eval.kappa import quadratic_kappa


def _ordinal_noise(true: np.ndarray, p_adj: float, p_jump: float, rng: np.random.Generator) -> np.ndarray:
    pred = true.copy()
    u = rng.random(len(true))
    shift = np.where(u < p_adj, rng.choice([-1, 1], size=len(true)), 0)
    jump = np.where((u >= p_adj) & (u < p_adj + p_jump), rng.choice([-2, 2], size=len(true)), 0)
    pred = np.clip(pred + shift + jump, 1, 5)
    return pred


def match_kappa(
    n: int,
    target: float,
    mix: tuple[float, ...] = FRENCH_MIX,
    seed: int = 0,
    tol: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    levels = np.array(FRENCH_LEVELS)
    true = rng.choice(levels, size=n, p=np.array(mix) / np.sum(mix))
    best = None
    best_err = 1e9
    for p_adj in np.linspace(0.04, 0.40, 12):
        for p_jump in np.linspace(0.0, 0.10, 6):
            pred = _ordinal_noise(true, p_adj, p_jump, rng)
            k = quadratic_kappa(true, pred)
            err = abs(k - target)
            if err < best_err:
                best_err = err
                best = (true, pred, k, p_adj, p_jump)
            if err <= tol:
                return true, pred
    if best is None:
        raise RuntimeError("kappa matching failed")
    return best[0], best[1]


def r1_near_perfect(n: int, target: float, mix=FRENCH_MIX, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """High-agreement generator for R1 (algorithmic approximation)."""
    rng = np.random.default_rng(seed)
    levels = np.array(FRENCH_LEVELS)
    true = rng.choice(levels, size=n, p=np.array(mix) / np.sum(mix))
    p_flip = max(0.0, min(0.2, 1.0 - target))
    pred = true.copy()
    flip = rng.random(n) < p_flip
    pred[flip] = np.clip(pred[flip] + rng.choice([-1, 1], size=flip.sum()), 1, 5)
    # Fine-tune by additional random adjacent flips
    for _ in range(40):
        k = cohen_kappa_score(true, pred, weights="quadratic")
        if abs(k - target) <= 0.001:
            break
        i = rng.integers(0, n)
        if k > target:
            pred[i] = int(np.clip(pred[i] + rng.choice([-1, 1]), 1, 5))
        else:
            pred[i] = true[i]
    return true, pred
