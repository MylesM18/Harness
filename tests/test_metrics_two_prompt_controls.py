"""
The 2-prompt controls crash (open bug from handoff #4/#5).

`metrics.compute_all` pairs the pro and con arms on
(scenario_id, model, replicate, turn_index) but NOT on system_prompt. The
positive-controls run codes the SAME subject model under two system prompts
(anti_syco, warm) in one frame, so every pairing key appears twice and
`_paired`'s `pd.concat(axis=1)` blows up with

    ValueError: cannot handle a non-unique multi-index!

The two system prompts are distinct experimental CONDITIONS (the whole point of
the controls is warm vs anti_syco), so the fix must keep them SEPARATE - never
pool - exactly like `compute_all_per_judge` keeps two judges apart.

These tests plant a two-system_prompt frame with the simulate.py generator (two
different profiles under one model NAME, one tag each) - the exact shape of the
controls run - and assert compute_all survives it without pooling. The
conversation_id test pins the same-family latent bug at judge.py:261.

Run: .venv/bin/python -m pytest tests/test_metrics_two_prompt_controls.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from harness.scenarios import load_all, TEST_SCENARIOS_DIR
from harness.simulate import simulate_judgments
from harness.schema import TurnJudgment, RunSpec
from harness.judge import judgment_to_row
from harness import metrics


@pytest.fixture(scope="module")
def scenarios():
    return load_all(TEST_SCENARIOS_DIR)


@pytest.fixture(scope="module")
def two_prompt_frame(scenarios):
    """One subject model, coded under two system prompts in ONE frame.

    Same subject NAME ("subj") in both halves so the (scenario, model, replicate,
    turn) pairing key collides across prompts - the exact shape that crashed
    _paired on the real controls run. Different underlying profiles so the two
    prompts carry genuinely different numbers, which lets the no-pooling
    assertion mean something.
    """
    anti = simulate_judgments(scenarios, models={"subj": "holds"},
                              n_turns=6, n_replicates=2, seed=1)
    warm = simulate_judgments(scenarios, models={"subj": "drifts"},
                              n_turns=6, n_replicates=2, seed=2)
    anti["system_prompt"] = "anti_syco"
    warm["system_prompt"] = "warm"
    return pd.concat([anti, warm], ignore_index=True)


@pytest.fixture(scope="module")
def two_prompt_frame_missing_cell(two_prompt_frame):
    """The controls frame with one subject turn absent - exactly what a refusal
    or a dropped judgment leaves behind. This asymmetry is what tips the buggy
    _paired from silently MISaligning the duplicated index into the hard
    `ValueError: cannot handle a non-unique multi-index!` seen on the real run.
    """
    m = ((two_prompt_frame.system_prompt == "warm")
         & (two_prompt_frame.arm == "pro")
         & (two_prompt_frame.turn_index == 3)
         & (two_prompt_frame.replicate == 0))
    return two_prompt_frame[~m].reset_index(drop=True)


def test_two_prompts_do_not_crash_compute_all(two_prompt_frame_missing_cell,
                                              scenarios):
    """Regression: the non-unique multi-index crash. compute_all on a two-prompt
    frame with an asymmetric missing cell must return, not raise ValueError."""
    out = metrics.compute_all(two_prompt_frame_missing_cell, scenarios)
    assert set(out.keys()) == set(
        metrics.compute_all(
            two_prompt_frame_missing_cell[
                two_prompt_frame_missing_cell.system_prompt == "anti_syco"],
            scenarios).keys())


def test_two_prompts_are_kept_separate(two_prompt_frame, scenarios):
    """No pooling: the tidy metric frames carry a system_prompt column with BOTH
    conditions, and a per-(model,turn) metric has one row per prompt - not one
    pooled row that averages anti_syco and warm together."""
    out = metrics.compute_all(two_prompt_frame, scenarios)

    fric = out["friction"]
    assert "system_prompt" in fric.columns, "friction pooled the two prompts"
    assert set(fric.system_prompt.unique()) == {"anti_syco", "warm"}
    # exactly one friction row per (model, turn, system_prompt): two prompts ->
    # two rows per (model, turn), never collapsed to one.
    per_cell = fric.groupby(["model", "turn_index", "system_prompt"]).size()
    assert (per_cell == 1).all()

    div = out["divergence"]
    assert set(div.system_prompt.unique()) == {"anti_syco", "warm"}


def test_single_prompt_frame_output_is_unchanged(scenarios):
    """Guard the no-regression contract: a single-prompt frame (Stage C/D real
    data, or the legacy path) must come back exactly as before - same row counts,
    no spurious duplication from the new split."""
    base = simulate_judgments(scenarios, models={"subj": "holds"},
                              n_turns=6, n_replicates=2, seed=3)
    base["system_prompt"] = "neutral"
    out = metrics.compute_all(base, scenarios)
    # friction is one row per (model, turn); 6 turns, one model -> 6 rows.
    assert len(out["friction"]) == out["friction"][["model", "turn_index"]].drop_duplicates().shape[0]
    assert len(out["friction"]) == 6


def test_conversation_id_is_unique_per_system_prompt():
    """Same-family latent bug (judge.py:261): conversation_id omits
    system_prompt, so anti_syco and warm turns of the same
    (model, scenario, arm, replicate) collide on one id. Two specs that differ
    ONLY in system_prompt must yield DISTINCT conversation_ids."""
    j = TurnJudgment(run_key="k", turn_index=1, stance=0.0, stance_confidence=0.9,
                     considerations_present=[], contains_challenge=False,
                     challenge_strength=0.0)
    common = dict(scenario_id="S01", arm="pro", pressure="gradual",
                  model="claude-sonnet-4-6", replicate=0)
    row_anti = judgment_to_row(j, RunSpec(system_prompt="anti_syco", **common))
    row_warm = judgment_to_row(j, RunSpec(system_prompt="warm", **common))
    assert row_anti["conversation_id"] != row_warm["conversation_id"]
