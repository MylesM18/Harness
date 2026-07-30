#!/usr/bin/env python3
"""
Analyse a real judgments file and emit every figure plus the statistics table.

    python scripts/analyze.py --input data/judgments.jsonl

Identical code path to demo_synthetic.py. The only difference is where the
judgments came from, which is the point: the analysis was fixed and reviewable
before any of it was real.
"""
import argparse, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sps

from harness.scenarios import load_all, REAL_SCENARIOS_DIR
from harness import metrics, plots, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/judgments.jsonl")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--tost-bound", type=float, default=0.15,
                    help="region of practical equivalence for content invariance, "
                         "on the -1..+1 stance scale. Fix this BEFORE looking at results.")
    args = ap.parse_args()

    scenarios = load_all(REAL_SCENARIOS_DIR)
    j = pd.read_json(ROOT / args.input, lines=True)
    figdir = ROOT / args.figdir
    figdir.mkdir(exist_ok=True)

    print(f"{len(j):,} judgments | {j.conversation_id.nunique()} conversations "
          f"| {j.scenario_id.nunique()} scenarios | {j.model.nunique()} models\n")

    if j.scenario_id.nunique() < 8:
        print("!! fewer than 8 scenarios. Results describe this scenario set, not "
              "these models. State that as a scope condition.\n")

    out = metrics.compute_all(j, scenarios)

    plots.fig_channel_separation(out["divergence"], figdir / "fig1_channel_separation.png")
    plots.fig_friction_survival(out["judgments_enriched"], figdir / "fig2_friction_survival.png")
    plots.fig_asymmetric_attrition(out["coverage_attrition"], figdir / "fig3_asymmetric_attrition.png")
    plots.fig_speaker_free_floor(out["uat"], figdir / "fig4_speaker_free_floor.png")
    plots.fig_horizon(out["horizon"], figdir / "fig6_horizon.png")
    plots.fig_flip_blindspot(out["judgments_enriched"], figdir / "fig7_flip_blindspot.png")

    report, rows, pvals = {}, [], {}
    for m in sorted(j.model.unique()):
        d_div = out["divergence"][out["divergence"].model == m]
        d_cov = out["coverage_attrition"]
        d_cov = d_cov[(d_cov.model == m) & (d_cov.arm.isin(["pro", "con"]))]
        d_j = j[(j.model == m) & (j.arm.isin(["pro", "con"]))]
        d_uat = out["uat"][out["uat"].model == m]
        late = d_j.turn_index >= d_j.turn_index.quantile(0.75)

        aai = stats.simple_slope(d_cov, "aai")
        fhl = metrics.friction_half_life(out["friction"], model=m)

        early_cut, late_cut = d_div.turn_index.quantile(0.25), d_div.turn_index.quantile(0.75)
        deltas = (d_div[d_div.turn_index >= late_cut].groupby(["scenario_id", "replicate"]).content_div.mean()
                  - d_div[d_div.turn_index <= early_cut].groupby(["scenario_id", "replicate"]).content_div.mean())
        eq = stats.tost_equivalence(deltas.to_numpy(), bound=args.tost_bound)

        pvals[f"AAI_slope::{m}"] = float(2 * sps.norm.sf(abs(aai["slope_per_turn"] / (aai["boot_se"] + 1e-12))))

        report[m] = {
            "PRIMARY_aai_slope_per_turn": aai,
            "PRIMARY_terminal_friction": float(d_j[late].contains_challenge.mean()),
            "friction": fhl,
            "harness_ratio_mean": float(d_div.harness_ratio.mean(skipna=True)),
            "harness_ratio_defined_share": float(d_div.hr_defined.mean()),
            "noise_share_mean": float(d_div.noise_share.mean()),
            "content_invariance_TOST": eq,
            "uat_mean": float(d_uat.uat.mean()),
            "raw_gap_mean": float(d_uat.raw_gap.mean()),
            "floor_share_of_raw": float(abs(d_uat.floor_gap).mean() / (abs(d_uat.raw_gap).mean() + 1e-9)),
        }
        rows += [
            dict(model=m, metric="content_invariance",
                 value_normalised=float(np.clip(1 - d_div.content_div.mean() / 0.5, 0, 1))),
            dict(model=m, metric="friction_retained",
                 value_normalised=float(d_j[late].contains_challenge.mean())),
            dict(model=m, metric="symmetric_coverage",
                 value_normalised=float(np.clip(1 - abs(d_cov[d_cov.turn_index >= d_cov.turn_index.quantile(.75)].aai.mean()) / 0.4, 0, 1))),
            dict(model=m, metric="floor_corrected_resistance",
                 value_normalised=float(np.clip(1 - abs(d_uat.uat.mean()) / 0.5, 0, 1))),
            dict(model=m, metric="horizon_alignment",
                 value_normalised=float(out["horizon"].query("model == @m and arm in ['pro','con']").serves_objective.mean())),
            dict(model=m, metric="harness_ratio",
                 value_normalised=float(d_div.harness_ratio.mean(skipna=True))),
        ]

    plots.fig_profile_scorecard(pd.DataFrame(rows), figdir / "fig5_profile_scorecard.png")

    report["_multiplicity_secondary"] = stats.benjamini_hochberg(pvals).to_dict("records")
    report["_scope"] = {
        "n_scenarios": int(j.scenario_id.nunique()),
        "generalises_to": "this scenario set" if j.scenario_id.nunique() < 8 else "the sampled domains",
        "judge_model": sorted(j.judge_model.unique().tolist()),
    }

    p = ROOT / "data/report.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(f"\nreport -> {p}\nfigures -> {figdir}")


if __name__ == "__main__":
    main()
