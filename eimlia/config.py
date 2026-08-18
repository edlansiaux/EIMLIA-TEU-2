"""Canonical constants for the dual-regime evaluation and SEED-compliant economics.

Patient-level records are not shareable (CNIL MR-004 / GDPR). Published
cohort sizes, R1 observations, and the completed n=400 R2 point estimates
are the only empirical anchors used here.
"""

from dataclasses import dataclass

FRENCH_LEVELS = (1, 2, 3, 4, 5)

# Calibration cohort: 2018-2023 complete structured records (NOT an annual volume).
N_CALIBRATION = 340_536
N_EXCLUDED = 118_723
N_VISITS_RAW = 459_259
TEST_FRACTION = 0.20
N_TEST_R1 = 68_108

# Simulation annual perimeter (SEED Q1, lecture A).
N_SIM_ANNUAL = 600_000
N_SITE_ANNUAL = 115_000  # typical CHU-scale site (SEED H4)
N_SITES_SIMULATED = N_SIM_ANNUAL // N_SITE_ANNUAL  # 5 sites in the DES population

# Expert R2 sample.
N_R2_COMPLETED = 400
N_R2_PLANNED = 3_000
N_EXPERTS = 5
FLEISS_KAPPA = 0.72
FLEISS_KAPPA_CI = (0.68, 0.76)
ICC_2_1 = 0.74
ICC_CI = (0.70, 0.78)
DISAGREEMENT_RATE = 0.235
REMODERATION_KAPPA = 0.91
REMODERATION_CI = (0.84, 0.97)
N_REMODERATION = 30

# Nurse baseline and deployment threshold (not a non-inferiority margin).
NURSE_KAPPA_W = 0.65
DEPLOYMENT_THRESHOLD = 0.80
THRESHOLD_DELTA = DEPLOYMENT_THRESHOLD - NURSE_KAPPA_W  # 0.15; superiority vs 0.65

# Empirical FRENCH mix used for synthetic residual-error matrices (adult ED).
FRENCH_MIX = (0.03, 0.10, 0.32, 0.38, 0.17)

# R1 observed (full test set).
R1_KAPPA = {
    "URGENTIAPARSE": 0.996,
    "EMERGINET": 0.939,
    "TRIAGEMASTER": 0.895,
}
R1_KAPPA_CI = {
    "URGENTIAPARSE": (0.995, 0.996),
    "EMERGINET": (0.939, 0.940),
    "TRIAGEMASTER": (0.894, 0.895),
}

# Empirical R2 point estimates on the completed n=400 moderated subset.
# These are the only clinical-validity observations; n=3,000 figures are projections.
R2_N400_KAPPA = {
    "URGENTIAPARSE": 0.81,
    "EMERGINET": 0.74,
    "TRIAGEMASTER": 0.69,
}

# Simulation.
P_ACCEPT = 0.85
N_SIM_REPS = 40
SIM_HOURS = 72
MEAN_ARRIVAL_PER_H = 6.5  # within the 2.1-12.8 range
N_STATIONS = 17
SERVICE_MINUTES = {1: 38.0, 2: 34.0, 3: 28.0, 4: 22.0, 5: 16.0}

# Economics — in-run / previously published anchors (SEED Q4-Q5).
DISCOUNT_BASE = 0.03
DISCOUNT_HAS = 0.025
HORIZON_YEARS = 3
TCO_ANNUAL = 254_000.0
RAMP_UP = (0.50, 0.80, 1.00)
PSA_ITERS = 50_000
WTP_HAS_REF = 50_000.0  # reference, not an official French threshold
WTP_CONSERVATIVE = 10_000.0
QALY_BETA = (3.0, 17.0)  # mean 0.15; applied only to avoided L1-2 undertriage
EI_UNIT_COST = (3_300.0, 5_300.0)
NATIONAL_EI_AVOIDED = 15_000
NATIONAL_ED_COUNT = 650
NATIONAL_CAPTURE = (0.08, 0.13)

# Unpublished [hors-run] perspective splits must not appear in the manuscript.
UNPUBLISHABLE_HORS_RUN = {
    "net_hospital_eur_per_visit": 12.7,
    "net_am_eur_per_visit": 26.5,
    "net_societal_eur_per_visit": 19.2,
}


@dataclass(frozen=True)
class ResultTag:
    """SEED quadruplet: perimeter; channels; regime; penetration."""

    perimeter: str
    channels: str
    regime: str
    penetration: str

    def label(self) -> str:
        return f"{self.perimeter}; {self.channels}; {self.regime}; {self.penetration}"


TAG_R1_SITE = ResultTag("single-site calibration", "none (predictive)", "R1", "n/a")
TAG_R2_N400 = ResultTag("n=400 moderated subset", "none (predictive)", "R2 empirical", "n/a")
TAG_ROI_SITE = ResultTag(
    f"mono-site ~{N_SITE_ANNUAL:,} visits/year",
    "hospital net (all modelled channels)",
    "R2-anchored",
    "100% of one site",
)
TAG_NATIONAL_EI = ResultTag(
    "national 20M visits / 650 EDs",
    "undertriage adverse-event channel only",
    "R2",
    "implicit capture 8-13%",
)
