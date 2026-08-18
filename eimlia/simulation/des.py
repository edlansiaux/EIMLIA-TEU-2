"""Lightweight priority-queue DES with class-specific AI residual errors.

50 independent 3-day replications. Human-AI interaction is a Bernoulli
acceptance gate (p_acc=0.85). Median length of stay (DMS) is the flow
endpoint; Level 1-2 undertriage is the safety endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eimlia.config import (
    FRENCH_MIX,
    MEAN_ARRIVAL_PER_H,
    N_SIM_REPS,
    N_STATIONS,
    P_ACCEPT,
    R2_N400_KAPPA,
    SERVICE_MINUTES,
    SIM_HOURS,
)
from eimlia.eval.kappa import critical_undertriage_rate, quadratic_kappa
from eimlia.simulation.confusion import confusion_sampler, inject


@dataclass
class RunStats:
    dms_median: float
    concordance: float
    undertriage: float
    n_patients: int


def _arrivals(hours: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.poisson(rate * hours)
    return np.sort(rng.uniform(0.0, hours * 60.0, size=n))


def _one_run(P: np.ndarray | None, load: float, rng: np.random.Generator) -> RunStats:
    t_arr = _arrivals(SIM_HOURS, MEAN_ARRIVAL_PER_H * load, rng)
    n = len(t_arr)
    true = rng.choice(np.arange(1, 6), size=n, p=np.array(FRENCH_MIX) / np.sum(FRENCH_MIX))
    if P is None:
        assigned = true.copy()
    else:
        ai = inject(true, P, rng)
        accept = rng.random(n) < P_ACCEPT
        assigned = np.where(accept, ai, true)
    service = np.array([SERVICE_MINUTES[int(a)] for a in assigned], dtype=float)
    n_servers = N_STATIONS
    free_at = np.zeros(n_servers)
    start = np.zeros(n)
    waiting: list[tuple[int, float, int]] = []
    i = 0
    t = 0.0
    done = 0
    while done < n:
        while i < n and t_arr[i] <= t + 1e-9:
            waiting.append((int(assigned[i]), float(t_arr[i]), i))
            i += 1
        waiting.sort()
        assigned_now = False
        for s in range(n_servers):
            if waiting and free_at[s] <= t + 1e-9:
                _, _, idx = waiting.pop(0)
                start[idx] = t
                free_at[s] = t + service[idx]
                done += 1
                assigned_now = True
        if assigned_now:
            continue
        cand = []
        if i < n:
            cand.append(float(t_arr[i]))
        later = free_at[free_at > t + 1e-9]
        if waiting and later.size:
                cand.append(float(later.min()))
        if not cand:
            if i < n:
                t = float(t_arr[i])
                continue
            break
        t = min(cand)
    los = (start + service) - t_arr
    conc = 1.0 if P is None else quadratic_kappa(true, assigned)
    under = 0.0 if P is None else critical_undertriage_rate(true, assigned)
    return RunStats(float(np.median(los)), float(conc), float(under), n)


def run_scenarios(n_reps: int = N_SIM_REPS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    samplers = {m: confusion_sampler(m, seed=seed + i) for i, m in enumerate(R2_N400_KAPPA)}
    scenarios = {
        "S1_manual": (None, 1.0),
        "S2a_TRIAGEMASTER": (np.array(samplers["TRIAGEMASTER"]["P_pred_given_true"]), 1.0),
        "S2b_URGENTIAPARSE": (np.array(samplers["URGENTIAPARSE"]["P_pred_given_true"]), 1.0),
        "S2c_EMERGINET": (np.array(samplers["EMERGINET"]["P_pred_given_true"]), 1.0),
        "S3_crisis_ensemble": (np.array(samplers["URGENTIAPARSE"]["P_pred_given_true"]), 2.0),
    }
    results = {"replications": n_reps, "horizon_hours": SIM_HOURS, "error_model": "class-conditional confusion (R2 n=400)", "scenarios": {}}
    s1 = []
    for name, (P, load) in scenarios.items():
        dms, conc, under = [], [], []
        for r in range(n_reps):
            st = _one_run(P, load, np.random.default_rng(rng.integers(0, 1_000_000_000)))
            dms.append(st.dms_median)
            conc.append(st.concordance)
            under.append(st.undertriage)
        dms = np.array(dms)
        if name == "S1_manual":
            s1 = dms
            delta = np.zeros_like(dms)
        else:
            delta = (dms - np.median(s1)) / np.median(s1) * 100.0
        results["scenarios"][name] = {
            "DMS_median": float(np.median(dms)),
            "DMS_ci95": [float(np.quantile(dms, 0.025)), float(np.quantile(dms, 0.975))],
            "delta_DMS_pct": float(np.median(delta)),
            "delta_DMS_ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
            "concordance_kappa": float(np.mean(conc)),
            "critical_undertriage": float(np.mean(under)),
            "n_reps": n_reps,
        }
    results["confusion_sources"] = samplers
    return results
