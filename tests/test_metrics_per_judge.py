"""
gap #2 as real code: `metrics.compute_all` groups by `model`, never
`judge_model`. Feeding a multi-judge frame into one `compute_all` call silently
POOLS independent coders - it averages judges that START_HERE.md says must be
reported separately. `metrics.compute_all_per_judge` runs the full battery once
per `judge_model` group and keeps them apart.

These tests assert exactly that separation on a planted-truth frame built by
simulate.py (the same generator the metric-recovery suite trusts): two judges
coding the SAME subject conversations must come back as two independent
batteries, never one pooled frame.

Run: .venv/bin/python -m pytest tests/test_metrics_per_judge.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from harness.scenarios import load_all, TEST_SCENARIOS_DIR
from harness.simulate import simulate_judgments
from harness import metrics


TWO_JUDGES = ["gemini-3.1-pro", "gpt-5.6-terra"]


@pytest.fixture(scope="module")
def scenarios():
    return load_all(TEST_SCENARIOS_DIR)


@pytest.fixture(scope="module")
def two_judge_frame(scenarios):
    """One set of subject conversations, coded by two distinctly-tagged judges.

    simulate_judgments stamps a single judge_model ("synthetic"); we re-tag two
    copies so the frame looks like a real cross-family panel run - identical
    subject rows, two independent coders.
    """
    base = simulate_judgments(scenarios,
                              models={"holds": "holds", "drifts": "drifts"},
                              n_turns=6, n_replicates=2, seed=7)
    tagged = []
    for jm in TWO_JUDGES:
        g = base.copy()
        g["judge_model"] = jm
        tagged.append(g)
    return pd.concat(tagged, ignore_index=True)


def test_returns_one_battery_per_distinct_judge(two_judge_frame, scenarios):
    """Result keys equal the distinct judge ids - one full battery each."""
    out = metrics.compute_all_per_judge(two_judge_frame, scenarios)
    assert set(out.keys()) == set(TWO_JUDGES)
    # Each value is a full compute_all battery (same metric frames as compute_all).
    single = metrics.compute_all(
        two_judge_frame[two_judge_frame.judge_model == TWO_JUDGES[0]], scenarios)
    for battery in out.values():
        assert set(battery.keys()) == set(single.keys())


def test_judges_are_not_pooled(two_judge_frame, scenarios):
    """The crisp no-pooling assertion: each per-judge battery is computed on ONLY
    that judge's rows. A single pooled compute_all call would carry both judge
    ids in its enriched frame; the per-judge batteries must each carry exactly
    one."""
    out = metrics.compute_all_per_judge(two_judge_frame, scenarios)
    for jm, battery in out.items():
        enriched = battery["judgments_enriched"]
        assert list(enriched.judge_model.unique()) == [jm], (
            f"battery for {jm} pooled other judges: "
            f"{enriched.judge_model.unique()}")
    # And the pooled control genuinely mixes them, proving the frame could pool.
    pooled = metrics.compute_all(two_judge_frame, scenarios)
    assert set(pooled["judgments_enriched"].judge_model.unique()) == set(TWO_JUDGES)


def test_missing_judge_model_column_falls_back_to_single_battery(two_judge_frame,
                                                                 scenarios):
    """A frame with no judge_model column (single-judge legacy path) returns one
    battery under the None key - never an empty dict, never a crash."""
    frame = two_judge_frame.drop(columns=["judge_model"])
    out = metrics.compute_all_per_judge(frame, scenarios)
    assert set(out.keys()) == {None}
    assert set(out[None].keys()) == set(metrics.compute_all(frame, scenarios).keys())
