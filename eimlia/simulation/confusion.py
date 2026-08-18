"""Class-specific residual-error injection for DES-MAS.

A single weighted kappa does not determine Level 1-2 undertriage. Residual
AI errors are therefore sampled from a 5x5 confusion matrix calibrated to
the R2 n=400 pattern, not from a global concordance scalar.
"""

from __future__ import annotations

import numpy as np

from eimlia.config import FRENCH_LEVELS, R2_N400_KAPPA
from eimlia.eval.kappa import confusion, critical_undertriage_rate, quadratic_kappa
from eimlia.eval.synthetic_labels import match_kappa


def confusion_sampler(name: str, seed: int = 0) -> dict:
    true, pred = match_kappa(2_000, R2_N400_KAPPA[name], seed=seed)
    cm = confusion(true, pred).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    return {
        "model": name,
        "P_pred_given_true": cm.tolist(),
        "kappa_w": quadratic_kappa(true, pred),
        "critical_undertriage": critical_undertriage_rate(true, pred),
        "source": "R2 n=400 class-conditional pattern, not global kappa",
    }


def inject(true_level: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(true_level)
    for i, t in enumerate(true_level):
        out[i] = rng.choice(FRENCH_LEVELS, p=P[t - 1])
    return out
