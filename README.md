# EIMLIA-TEU-2

Pre-clinical **evaluation framework** for AI-assisted emergency-department triage. The contribution is methodological: it separates **algorithmic approximation (R1)** from **clinical validity (R2)**, injects **class-conditional** residual errors into a hybrid DES–MAS, and reports health-economic results with explicit perimeters.

This is **not** a new triage algorithm. TRIAGEMASTER, URGENTIAPARSE and EMERGINET were published elsewhere and are used here as instruments.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Dual-regime protocol

| Regime | Training target | Evaluation reference | What a high score means |
|--------|-----------------|----------------------|-------------------------|
| **R1** | FRENCH reconstructed labels | Same FRENCH labels | The model recovered the deterministic scale |
| **R2** | FRENCH reconstructed labels | External 5-physician consensus | The model agrees with clinicians |

If FRENCH and experts disagree, an R1-perfect model **automatically** disagrees with experts. That is a label artefact, not evidence of a new medical algorithm.

Expert moderation is complete for **n = 400** of **3,000** planned cases. Only n = 400 is empirical R2. n = 3,000 figures are design-based projections.

### Empirical anchors (published R1 + completed R2 subset)

| Model | R1 κw (n = 68,108) | R2 κw n = 400 | 95% CI | Crosses 0.80? | L1–2 undertriage |
|-------|--------------------|---------------|--------|---------------|------------------|
| URGENTIAPARSE | 0.996 | 0.81 | [0.76, 0.85] | **yes** | 9.6% |
| EMERGINET | 0.939 | 0.74 | [0.70, 0.79] | no (below) | 11.3% |
| TRIAGEMASTER | 0.895 | 0.69 | [0.64, 0.75] | no (below) | 27.0% |

Deployment bar: κw ≥ 0.80. Nurse baseline: 0.65. Tests against these bars are **threshold/superiority tests**, not two-arm non-inferiority.

FRENCH vs expert on n = 400: κw ≈ 0.81 [0.77, 0.85] (ceiling for any R1-perfect model).

---

## Repository layout

```
EIMLIA-TEU-2/
├── eimlia/                     # Canonical evaluation package
│   ├── config.py               # Cohort sizes, tags, unpublished-value guards
│   ├── eval/                   # Dual-regime kappa, bootstrap, threshold tests
│   ├── simulation/             # Class-conditional confusion + priority DES
│   └── economics/              # CHEERS PSA, ROI decomposition, national EI scenario
├── scripts/run_eval_pipeline.py
├── results/                    # JSON written by the pipeline (no patient data)
├── manuscript/                 # IEEE source + compiled PDF
├── Rapport.ipynb               # Historical end-to-end notebook (patient data not included)
├── requirements.txt
└── LICENSE
```

Patient-level extracts are **not** in this repository (CNIL MR-004 / GDPR). The pipeline reproduces CIs, confusion patterns, DES residual-error injection and economic labels from published anchors.

---

## Quick start

```bash
git clone https://github.com/edlansiaux/EIMLIA-TEU-2.git
cd EIMLIA-TEU-2
pip install -r requirements.txt
python scripts/run_eval_pipeline.py
```

Outputs:

- `results/dual_regime.json` — R1/R2 tables, FRENCH-vs-expert ceiling, threshold tests
- `results/simulation.json` — class-conditional confusion and 40-replication DES
- `results/economics.json` — mono-site ROI decomposition, PSA, national EI-only scenario

Build the paper (XeLaTeX / tectonic):

```bash
cd manuscript
tectonic -X compile main.tex
```

---

## Simulation: why confusion matrices, not a scalar kappa

A single weighted kappa does not determine clinically relevant errors. Two models with similar κw can differ by a factor of three in Level 1–2 undertriage (9.6% vs 27.0% on n = 400). Residual AI errors in the DES–MAS are sampled from `P(ŷ | y)`, estimated on the completed expert subset, then passed through a Bernoulli acceptance gate (p_acc = 0.85).

Flow KPIs in the manuscript (DMS ≈ 31–37 min) come from the calibrated hybrid engine documented in `Rapport.ipynb`. The package `eimlia.simulation` is the **reproducible residual-error module** (class-conditional injection + independent replications).

---

## Health economics (perimeter-tagged reporting)

Every figure must be tagged **(perimeter; channels; regime; penetration)**.

| Output | Tag | Status |
|--------|-----|--------|
| ROI 480% [210, 1250] | mono-site ~115k visits/year; hospital; R2; 100% of one site | Undiscounted 3-year, no ramp-up |
| ROI ≈ 341% | same, with 50/80/100% ramp-up | Prefer this for planning |
| Conservative floor 210% | lower CrI | Use in external conservative communication |
| ICER 1,840 €/QALY | in-run | Citable; France has no official threshold |
| CEAC 99.4% at 50,000 €/QALY | in-run | HAS reference WTP, not a legal bar |
| 50–80 M€/year | national; **undertriage AE channel only**; R2; capture 8–13% | Conservative derived scenario |

**Not published here:** perspective-split €/visit reallocations that are not yet regenerated inside the PSA.

QALYs accrue from **avoided Level 1–2 undertriage**, not from every correct five-class label (Beta(3,17) increment; Castillo et al. emergency utilities). Discount 3% base, 2.5% HAS sensitivity. Indirect societal (human-capital) layer excluded from the base case (conservative).

Calibration cohort **340,536** = multi-year extract, **not** an annual volume. Simulation perimeter **600,000 visits/year**.

---

## What the historical notebook contains

`Rapport.ipynb` is the original CHU Lille pipeline (merge 2018–2023, FRENCH reconstruction, model training, PM4Py, SimPy/Mesa, deterministic TCO). It requires local `data/brutes` files that are not distributed. Treat `eimlia/` as the evaluation code that matches the revised manuscript. The notebook’s deterministic ROI 10,260% / ICER 93 €/QALY is **withdrawn** and superseded by the PSA.

---

## Ethics

CESREES clearance; Health Data Hub 27797006; GDPR; CNIL MR-004. No patient-level data in git.

---

## Citation

```bibtex
@inproceedings{lansiaux2026eimlia,
  title  = {Disentangling Algorithmic Approximation from Clinical Validity
            in AI Emergency Triage},
  author = {Lansiaux, \'Edouard and Zgaya-Biau, Hayfa and Ammi, Mehdi},
  year   = {2026}
}
```

Related model papers: BDCAT 2025 and JMIR Medical Informatics 2026 (TIAEU architectures).

## License

MIT. Clinical use requires prospective validation and regulatory review.
