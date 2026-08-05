#!/usr/bin/env python3
"""
Re-render the figures from judgments already on disk. No API calls, no re-run.

    python scripts/render_figures.py \
        --input data/judgments.stageD_full.jsonl \
        --report data/report.json \
        --figdir figures_stageD_full

Why this exists separately from scripts/analyze.py: analyze.py is the analysis of
record and it rewrites data/report.json as a side effect. When the only thing that
changed is how a figure looks, re-running the analysis to get new PNGs risks
perturbing the numbers of record for no reason. This script recomputes the metric
frames the figures need, reads the already-published statistics out of report.json
for the on-plot annotations, and writes nothing but images.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from harness.scenarios import load_all, REAL_SCENARIOS_DIR
from harness import metrics, plots


def _aai_slopes(report: dict, models) -> dict:
    """Published AAI slope + CI + BH p-value per model, straight from report.json."""
    pvals = {r["metric"]: r["p"] for r in report.get("_multiplicity_secondary", [])}
    out = {}
    for m in models:
        s = (report.get(m) or {}).get("PRIMARY_aai_slope_per_turn") or {}
        if not s:
            continue
        out[m] = {
            "slope_per_turn": s.get("slope_per_turn"),
            "ci_lo": s.get("ci_lo"),
            "ci_hi": s.get("ci_hi"),
            "p": pvals.get(f"AAI_slope::{m}"),
        }
    return out


def _scorecard(out: dict, j: pd.DataFrame) -> pd.DataFrame:
    """The fig-5 normalisation, mirroring scripts/analyze.py (the rows block in
    main()). Kept identical on purpose - if analyze.py's normalisation changes,
    change it here too, or the scorecard PNG stops matching the report."""
    rows = []
    for m in sorted(j.model.unique()):
        d_div = out["divergence"][out["divergence"].model == m]
        d_cov = out["coverage_attrition"]
        d_cov = d_cov[(d_cov.model == m) & (d_cov.arm.isin(["pro", "con"]))]
        d_j = j[(j.model == m) & (j.arm.isin(["pro", "con"]))]
        d_uat = out["uat"][out["uat"].model == m]
        late = d_j.turn_index >= d_j.turn_index.quantile(0.75)
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
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/judgments.stageD_full.jsonl")
    ap.add_argument("--report", default="data/report.json")
    ap.add_argument("--figdir", default="figures_stageD_full")
    args = ap.parse_args()

    report = json.loads((ROOT / args.report).read_text())
    scenarios = load_all(REAL_SCENARIOS_DIR)
    j_all = pd.read_json(ROOT / args.input, lines=True)

    # Same rule as analyze.py: figures describe the PRIMARY judge alone. Handing a
    # two-judge frame to compute_all would union the coders, not average them.
    primary = report["_scope"].get("primary_judge",
                                   j_all.judge_model.value_counts().index[0])
    j = j_all[j_all.judge_model == primary].copy()
    models = sorted(j.model.unique())
    print(f"{len(j):,} judgments | {j.conversation_id.nunique()} conversations "
          f"| {j.scenario_id.nunique()} scenarios | judge {primary}")

    out = metrics.compute_all(j, scenarios)
    figdir = ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    written = [
        plots.fig_channel_separation(out["divergence"], figdir / "fig1_channel_separation.png"),
        plots.fig_friction_survival(out["judgments_enriched"], figdir / "fig2_friction_survival.png"),
        plots.fig_asymmetric_attrition(out["coverage_attrition"], figdir / "fig3_asymmetric_attrition.png",
                                       slopes=_aai_slopes(report, models)),
        plots.fig_speaker_free_floor(out["uat"], figdir / "fig4_speaker_free_floor.png"),
        plots.fig_profile_scorecard(_scorecard(out, j), figdir / "fig5_profile_scorecard.png"),
        plots.fig_horizon(out["horizon"], figdir / "fig6_horizon.png"),
        plots.fig_flip_blindspot(out["judgments_enriched"], figdir / "fig7_flip_blindspot.png"),
    ]
    for p in written:
        print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
