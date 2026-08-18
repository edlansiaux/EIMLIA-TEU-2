"""Entry point: dual-regime metrics, class-specific DES, SEED-compliant PSA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eimlia.economics.psa import national_ei_only, psa, roi_decomposition
from eimlia.eval.dual_regime import run_dual_regime, save_report
from eimlia.simulation.des import run_scenarios


def dump(obj, path: Path) -> None:
    save_report(obj, path)


def main() -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    dual = run_dual_regime()
    save_report(dual, out / "dual_regime.json")
    sim = run_scenarios()
    dump(sim, out / "simulation.json")
    econ = {
        "roi_decomposition": roi_decomposition(),
        "psa": psa(n=8_000),
        "national_conservative": national_ei_only(),
    }
    dump(econ, out / "economics.json")
    print("Wrote", out / "dual_regime.json")
    print("Wrote", out / "simulation.json")
    print("Wrote", out / "economics.json")
    print("Mean R1-R2 gap:", round(dual["protocol"]["mean_R1_R2_gap"], 3))
    print("FRENCH vs expert kappa:", round(dual["french_vs_expert"]["kappa_w"], 3))
    for name, row in dual["models"].items():
        r2 = row["R2_n400"]
        print(
            f"{name}: R2 n=400 {r2['kappa_w_point']:.2f} "
            f"CI [{r2['ci95'][0]:.2f},{r2['ci95'][1]:.2f}] "
            f"crosses 0.80={r2['deployment_threshold_test']['ci_crosses_threshold']} "
            f"undertriage={r2['critical_undertriage']:.3f}"
        )


if __name__ == "__main__":
    main()
