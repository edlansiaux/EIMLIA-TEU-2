"""CHEERS-style PSA with SEED labelling rules.

Citable in-run anchors: ICER and CEAC. ROI 480% is a mono-site hospital
calculation (SEED H4). Perspective-split €/passage values are [hors-run]
and must not be exported to the manuscript.
"""

from __future__ import annotations

import numpy as np

from eimlia.config import (
    DISCOUNT_BASE,
    DISCOUNT_HAS,
    EI_UNIT_COST,
    HORIZON_YEARS,
    NATIONAL_CAPTURE,
    NATIONAL_ED_COUNT,
    NATIONAL_EI_AVOIDED,
    N_SITE_ANNUAL,
    PSA_ITERS,
    QALY_BETA,
    RAMP_UP,
    TAG_NATIONAL_EI,
    TAG_ROI_SITE,
    TCO_ANNUAL,
    UNPUBLISHABLE_HORS_RUN,
    WTP_CONSERVATIVE,
    WTP_HAS_REF,
)


def _pv(flows: np.ndarray, rate: float) -> np.ndarray:
    t = np.arange(1, flows.shape[0] + 1)[:, None]
    return np.sum(flows / (1.0 + rate) ** t, axis=0)


def roi_decomposition() -> dict:
    """Deterministic decomposition confirming SEED H4 (mono-site ~115k)."""
    net = UNPUBLISHABLE_HORS_RUN["net_hospital_eur_per_visit"]  # internal check only
    annual_gain = net * N_SITE_ANNUAL
    costs_3 = TCO_ANNUAL * HORIZON_YEARS
    gains_flat = annual_gain * HORIZON_YEARS
    roi_flat = (gains_flat - costs_3) / costs_3
    gains_ramp = annual_gain * sum(RAMP_UP)
    roi_ramp = (gains_ramp - costs_3) / costs_3
    disc = DISCOUNT_BASE
    pv_gain = sum(annual_gain * r / (1 + disc) ** (t + 1) for t, r in enumerate(RAMP_UP))
    pv_cost = sum(TCO_ANNUAL / (1 + disc) ** (t + 1) for t in range(HORIZON_YEARS))
    roi_disc = (pv_gain - pv_cost) / pv_cost
    pv_gain_has = sum(annual_gain * r / (1 + DISCOUNT_HAS) ** (t + 1) for t, r in enumerate(RAMP_UP))
    pv_cost_has = sum(TCO_ANNUAL / (1 + DISCOUNT_HAS) ** (t + 1) for t in range(HORIZON_YEARS))
    roi_has = (pv_gain_has - pv_cost_has) / pv_cost_has
    return {
        "hypothesis": "H4 mono-site hospital perspective",
        "tag": TAG_ROI_SITE.label(),
        "visits_per_year": N_SITE_ANNUAL,
        "tco_annual_eur": TCO_ANNUAL,
        "horizon_years": HORIZON_YEARS,
        "roi_flat_3y_undiscounted": roi_flat,
        "roi_ramp_50_80_100_undiscounted": roi_ramp,
        "roi_ramp_discount_3pct": roi_disc,
        "roi_ramp_discount_2p5_HAS": roi_has,
        "published_roi_480pct_matches_flat": bool(abs(roi_flat - 4.80) < 0.15),
        "conservative_external_floor": 2.10,
        "do_not_publish_eur_per_visit": True,
        "note": (
            "The 480% figure matches an undiscounted 3-year mono-site ratio "
            "without ramp-up. With 50/80/100% ramp-up and 3% discount the ROI "
            "is lower. External conservative communication uses the 210% floor."
        ),
    }


def national_ei_only() -> dict:
    lo = NATIONAL_EI_AVOIDED * EI_UNIT_COST[0]
    hi = NATIONAL_EI_AVOIDED * EI_UNIT_COST[1]
    return {
        "tag": TAG_NATIONAL_EI.label(),
        "range_eur_per_year": [lo, hi],
        "n_eds": NATIONAL_ED_COUNT,
        "capture": list(NATIONAL_CAPTURE),
        "channels": "undertriage adverse events only — not the four-channel total",
        "regime": "R2 — R1 must not feed national projections",
    }


def psa(n: int = PSA_ITERS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # Gains driven by avoided L1-2 undertriage, not by every correct label.
    # Beta on the QALY increment per avoided critical undertriage episode.
    qaly_unit = rng.beta(*QALY_BETA, size=n)
    n_crit_avoided = rng.lognormal(mean=np.log(180.0), sigma=0.35, size=n)  # per site-year
    qaly = n_crit_avoided * qaly_unit
    # Hospital net (internal; not exported as €/passage).
    net_visit = rng.lognormal(mean=np.log(12.7), sigma=0.35, size=n)
    annual_gain = net_visit * N_SITE_ANNUAL
    tco = rng.triangular(180_000, TCO_ANNUAL, 340_000, size=n)
    ramp = np.array(RAMP_UP)[:, None]
    gains = annual_gain * ramp
    costs = np.repeat(tco[None, :], HORIZON_YEARS, axis=0)
    pv_g = _pv(gains, DISCOUNT_BASE)
    pv_c = _pv(costs, DISCOUNT_BASE)
    delta_cost = pv_c - pv_g  # negative => savings
    delta_qaly = qaly * sum(RAMP_UP)
    nmb = -delta_cost + WTP_HAS_REF * delta_qaly
    icer = np.where(delta_qaly > 0, delta_cost / delta_qaly, np.nan)
    dominant = (delta_cost < 0) & (delta_qaly > 0)
    roi = (pv_g - pv_c) / pv_c
    ceac_50k = float(np.mean((-delta_cost + WTP_HAS_REF * delta_qaly) > 0))
    ceac_10k = float(np.mean((-delta_cost + WTP_CONSERVATIVE * delta_qaly) > 0))
    icer_nondom = icer[~dominant & np.isfinite(icer) & (icer > 0)]
    return {
        "tag": TAG_ROI_SITE.label(),
        "n_iterations": n,
        "qaly_mechanism": (
            "QALYs accrue only from avoided Level 1-2 undertriage episodes "
            "(Beta(3,17) increment), not from every correctly labelled visit. "
            "Utility increments follow emergency-care estimates in Castillo et al."
        ),
        "roi_median": float(np.median(roi)),
        "roi_crI95": [float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975))],
        "p_dominant": float(np.mean(dominant)),
        "icer_conditional_nondominant_median": (
            float(np.median(icer_nondom)) if len(icer_nondom) else None
        ),
        "published_icer_in_run_anchor": 1840.0,
        "ceac_50000": ceac_50k,
        "ceac_10000": ceac_10k,
        "published_ceac_50000_in_run": 0.994,
        "wtp_note": "France has no official ICER threshold; 50,000 is a HAS reference value",
        "nmb_mean_at_50000": float(np.mean(nmb)),
        "unpublished": {
            "reason": "SEED: perspective €/passage values are hors-run until in-run reallocation",
            "blocked_keys": list(UNPUBLISHABLE_HORS_RUN),
        },
    }
