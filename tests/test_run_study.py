"""
scripts/run_study.py is the real study driver. gap #1 and gap #2 are closed here:
the driver must build the cross-family judge PANEL (not a single Anthropic judge)
and keep the judges SEPARATED end to end - a combined frame plus one file per
judge, and a per-judge summary that never pools.

These tests drive the driver's seams OFFLINE - a real MockModelClient subject, a
real synthetic judge panel, real Runner/Judge code - with no key and no network.
They assert the wiring: every judge tags its rows, one judge failing does not
drop the others, and the per-judge files/summary keep coders apart.

Run: .venv/bin/python -m pytest tests/test_run_study.py -v
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from harness.scenarios import load_scenario, load_all, REAL_SCENARIOS_DIR, TEST_SCENARIOS_DIR
from harness.schema import RunSpec
from harness.runner import Runner
from harness.judge import Judge
from harness.simulate import simulate_judgments
from harness import providers


# run_study lives under scripts/ (not an importable package); load it by path.
_RS_PATH = ROOT / "scripts" / "run_study.py"
_spec = importlib.util.spec_from_file_location("run_study", _RS_PATH)
run_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_study)


SCENARIO_ID = "S04_tanking_strategy"
PANEL = ["gemini-3.1-pro", "gpt-5.6-terra"]


@pytest.fixture(scope="module")
def sc():
    return load_scenario(REAL_SCENARIOS_DIR / f"{SCENARIO_ID}.yaml")


class _BoomJudgeClient:
    """A judge client whose create always raises, to prove one judge failing does
    not drop the others. Judge retries then surfaces a RuntimeError, which the
    driver's loop must catch and log against THIS judge's model id."""

    def __init__(self):
        self.messages = providers._messages_namespace(self._create)

    def _create(self, **_kw):
        raise RuntimeError("judge SDK exploded")


# ---------------------------------------------------------------------------
# safe-id and judge-model resolution: small pure helpers, easy to get wrong.
# ---------------------------------------------------------------------------

def test_safe_judge_id_sanitizes_non_alphanumerics():
    assert run_study._safe_judge_id("gemini-3.1-pro") == "gemini_3_1_pro"
    assert run_study._safe_judge_id("gpt-5.6-terra") == "gpt_5_6_terra"


def test_resolve_judge_models_defaults_to_panel_and_singular_overrides():
    panel_args = types.SimpleNamespace(judge_model=None, judge_models=list(PANEL))
    assert run_study.resolve_judge_models(panel_args) == PANEL
    # The singular --judge-model alias, when passed, overrides the list.
    single_args = types.SimpleNamespace(judge_model="claude-sonnet-4-6",
                                        judge_models=list(PANEL))
    assert run_study.resolve_judge_models(single_args) == ["claude-sonnet-4-6"]


# ---------------------------------------------------------------------------
# The core offline smoke: a panel of two coders, run through the real Runner and
# real Judge, must tag EVERY row with its judge - both judges on every turn.
# ---------------------------------------------------------------------------

def test_run_conversations_offline_tags_every_judge(sc, tmp_path):
    spec = RunSpec(scenario_id=SCENARIO_ID, arm="pro", pressure="gradual",
                   model="mock-subject", n_turns=2)
    panel = providers.panel_from_models(PANEL, subject_models=["mock-subject"],
                                        offline=True)
    runner = Runner(client=providers.MockModelClient(), cache_dir=tmp_path)

    rows, failures = run_study.run_conversations([spec], {SCENARIO_ID: sc},
                                                 panel, runner)

    assert failures == []
    # 2 turns x 2 judges = 4 rows, each carrying its subject and judge id.
    assert len(rows) == 4
    assert {r["judge_model"] for r in rows} == set(PANEL)
    assert all(r["model"] == "mock-subject" and r["scenario_id"] == SCENARIO_ID
               for r in rows)
    # Both judges must appear on BOTH turns - no judge silently skipped.
    for t in (1, 2):
        judges_on_turn = {r["judge_model"] for r in rows if r["turn_index"] == t}
        assert judges_on_turn == set(PANEL)


def test_run_conversations_isolates_a_failing_judge(sc, tmp_path):
    """One judge blowing up must not cost the others their rows; the failure is
    logged against the offending judge's model id."""
    good = Judge(model="good-judge", client=providers.SyntheticJudgeClient())
    boom = Judge(model="boom-judge", client=_BoomJudgeClient())
    panel = providers.JudgePanel([good, boom])
    spec = RunSpec(scenario_id=SCENARIO_ID, arm="pro", pressure="gradual",
                   model="mock-subject", n_turns=2)
    runner = Runner(client=providers.MockModelClient(), cache_dir=tmp_path)

    rows, failures = run_study.run_conversations([spec], {SCENARIO_ID: sc},
                                                 panel, runner)

    # The good judge still produced a full set; the bad judge produced none.
    assert {r["judge_model"] for r in rows} == {"good-judge"}
    assert len(rows) == 2
    # Every failure is a judge-stage failure attributed to the boom judge.
    assert len(failures) == 2
    assert all(f["stage"] == "judge" and f["judge"] == "boom-judge"
               for f in failures)


# ---------------------------------------------------------------------------
# Output: one combined file with a judge_model column, PLUS one file per judge,
# each holding only that judge's rows.
# ---------------------------------------------------------------------------

def test_write_judgments_writes_combined_and_per_judge_files(tmp_path):
    rows = [
        {"judge_model": "gemini-3.1-pro", "turn_index": 1, "stance": 0.1},
        {"judge_model": "gpt-5.6-terra", "turn_index": 1, "stance": -0.2},
        {"judge_model": "gemini-3.1-pro", "turn_index": 2, "stance": 0.0},
    ]
    out = tmp_path / "judgments.jsonl"
    per_judge = run_study.write_judgments(rows, out)

    # Combined file holds every row, with the judge_model column intact.
    assert out.exists()
    combined = pd.read_json(out, lines=True)
    assert len(combined) == 3
    assert set(combined.judge_model) == {"gemini-3.1-pro", "gpt-5.6-terra"}

    # One file per judge, named by the sanitized id, each holding only its rows.
    assert set(per_judge.keys()) == {"gemini-3.1-pro", "gpt-5.6-terra"}
    assert per_judge["gemini-3.1-pro"].name == "judgments.gemini_3_1_pro.jsonl"
    assert per_judge["gpt-5.6-terra"].name == "judgments.gpt_5_6_terra.jsonl"
    for jm, path in per_judge.items():
        assert path.exists()
        g = pd.read_json(path, lines=True)
        assert list(g.judge_model.unique()) == [jm]
    assert len(pd.read_json(per_judge["gemini-3.1-pro"], lines=True)) == 2


# ---------------------------------------------------------------------------
# The per-judge summary must name each judge and never pool them; under --offline
# it is labelled a wire test (its numbers are a fixture, not a measurement).
# ---------------------------------------------------------------------------

def test_per_judge_summary_reports_each_judge_and_labels_wire_test():
    scenarios = load_all(TEST_SCENARIOS_DIR)
    base = simulate_judgments(scenarios, models={"holds": "holds", "drifts": "drifts"},
                              n_turns=6, n_replicates=2, seed=7)
    tagged = []
    for jm in PANEL:
        g = base.copy()
        g["judge_model"] = jm
        tagged.append(g)
    frame = pd.concat(tagged, ignore_index=True)

    summary = run_study.per_judge_summary(frame, scenarios, offline=True)

    assert isinstance(summary, str) and summary.strip()
    for jm in PANEL:
        assert jm in summary, f"summary did not report judge {jm}"
    assert "wire test" in summary.lower()
