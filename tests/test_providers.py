"""
providers.py is the one novel artifact behind the real-scenario walkthrough: the
single client seam that lets the *real* Runner / Judge / judge-panel code run

  * OFFLINE, with no API key and no network (MockModelClient +
    SyntheticJudgeClient), and
  * LIVE, against a cross-family judge panel (GeminiJudgeClient +
    OpenAIJudgeClient), whose SDKs are not in requirements.txt.

These tests drive the offline clients through the REAL harness code paths - a
real Runner.run and a real Judge.judge_turn - rather than mocking the harness.
They assert the SEAM CONTRACT and DATA SHAPES only. The offline clients are a
wire test, not science (see the providers.py docstring): their numbers are a
deterministic fixture, so nothing here asserts a metric value.

The two live clients are exercised only for their fail-closed behaviour: with
their SDK absent they must raise a clear, actionable ImportError, and the module
must still import (the live imports are lazy). Those paths are skipped if the SDK
happens to be installed, since then the ImportError branch cannot be reached.

Run: .venv/bin/python -m pytest tests/test_providers.py -v
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from harness.scenarios import load_scenario, REAL_SCENARIOS_DIR
from harness.schema import RunSpec, Trace, TurnJudgment
from harness.runner import Runner
from harness.judge import Judge
from harness import providers


SCENARIO_ID = "S04_tanking_strategy"
# S04 ships a 15-item inventory (6 favours-P, 6 favours-NOT-P, 3 neutral).
VALID_IDS = {f"C{i}" for i in range(1, 16)}


@pytest.fixture(scope="module")
def sc():
    return load_scenario(REAL_SCENARIOS_DIR / f"{SCENARIO_ID}.yaml")


def _sdk_present(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        # A missing PARENT package (e.g. `google`) surfaces as ModuleNotFoundError
        # from find_spec itself; that still means the SDK is absent.
        return False


# ---------------------------------------------------------------------------
# MockModelClient drives the REAL Runner end to end - no key, no network.
# ---------------------------------------------------------------------------

def test_runner_with_mock_client_produces_full_trace(sc, tmp_path):
    spec = RunSpec(scenario_id=SCENARIO_ID, arm="pro", pressure="gradual",
                   model="mock-subject", n_turns=4)
    runner = Runner(client=providers.MockModelClient(), cache_dir=tmp_path)
    trace = runner.run(spec, sc)

    assert isinstance(trace, Trace)
    # One assistant turn per user turn; the runner truncates the assembled
    # turns to n_turns (opener prepend + ladder, then [:n_turns]).
    assert len(trace.assistant_turns) == spec.n_turns
    # Messages alternate user/assistant, in order, and are all non-empty text.
    assert len(trace.messages) == 2 * spec.n_turns
    assert [m["role"] for m in trace.messages] == ["user", "assistant"] * spec.n_turns
    assert all(isinstance(m["content"], str) and m["content"].strip()
               for m in trace.messages)
    assert trace.assistant_turns == [m["content"] for m in trace.messages
                                     if m["role"] == "assistant"]
    # The runner's usage bookkeeping reads resp.usage.* off the seam; the mock
    # supplies it, so both sides must be positive.
    assert trace.meta["usage"]["input"] > 0
    assert trace.meta["usage"]["output"] > 0


def test_runner_with_mock_client_writes_cache_keyed_by_spec(sc, tmp_path):
    spec = RunSpec(scenario_id=SCENARIO_ID, arm="con", pressure="gradual",
                   model="mock-subject", n_turns=3)
    runner = Runner(client=providers.MockModelClient(), cache_dir=tmp_path)
    runner.run(spec, sc)
    assert (tmp_path / f"{spec.key}.json").exists(), \
        "Runner did not cache the trace under data/cache/{spec.key}.json"


# ---------------------------------------------------------------------------
# SyntheticJudgeClient parses back through the REAL Judge -> TurnJudgment.
# ---------------------------------------------------------------------------

def test_synthetic_judge_yields_wellformed_turnjudgment(sc):
    user_turn = sc.opening_user_turn
    assistant_turn = ("Here is the strongest counter-case: the main objection is "
                      "real, and here is the risk if the thesis is wrong.")
    judge = Judge(model="test-judge-x", client=providers.SyntheticJudgeClient())
    tj = judge.judge_turn(sc, user_turn, assistant_turn, run_key="rk1", turn_index=1)

    assert isinstance(tj, TurnJudgment)
    assert -1.0 <= tj.stance <= 1.0
    assert 0.0 <= tj.stance_confidence <= 1.0
    # The reported considerations must be a NON-EMPTY subset of the pre-registered
    # ids - non-empty guards against the vacuous "empty set is a subset" pass and
    # proves the client actually parsed ids out of the blind prompt.
    assert tj.considerations_present, "synthetic judge parsed no consideration ids"
    assert set(tj.considerations_present) <= VALID_IDS
    # The client reads only the visible assistant text; that text contains an
    # unsolicited objection/risk, so the blind read must flag a challenge.
    assert tj.contains_challenge is True
    # judge_model is stamped by the wrapping Judge, and the blind bookkeeping
    # (run_key, turn_index) is threaded through unchanged.
    assert tj.judge_model == "test-judge-x"
    assert tj.run_key == "rk1"
    assert tj.turn_index == 1


# ---------------------------------------------------------------------------
# The panel: one judgment per member, each tagged with its own judge_model, and
# the two synthetic members must NOT return identical fixtures (distinct seeds).
# ---------------------------------------------------------------------------

def test_offline_panel_returns_one_judgment_per_distinct_judge(sc):
    panel = providers.offline_panel()
    assert panel.judge_models == ["gemini-3.1-pro", "gpt-5.6-terra"]

    user_turn = sc.opening_user_turn
    assistant_turn = "A measured reply that keeps one objection in view."
    judgments = panel.judge_turn(sc, user_turn, assistant_turn,
                                 run_key="rk", turn_index=2)

    # Exactly one judgment per panel member, in member order, distinctly tagged.
    assert isinstance(judgments, list)
    assert [j.judge_model for j in judgments] == ["gemini-3.1-pro", "gpt-5.6-terra"]
    assert len({j.judge_model for j in judgments}) == len(judgments)
    # Distinct per-member seeds => the two synthetic judges must not be identical
    # fixtures. This is what makes the per-judge replication loop meaningful.
    assert judgments[0].raw != judgments[1].raw
    # Every member still returns a valid, blind coding with the shared bookkeeping.
    for j in judgments:
        assert isinstance(j, TurnJudgment)
        assert -1.0 <= j.stance <= 1.0
        assert set(j.considerations_present) <= VALID_IDS
        assert j.turn_index == 2 and j.run_key == "rk"


# ---------------------------------------------------------------------------
# panel_from_models: name -> family dispatch (gap #1 core). The pure helper
# _judge_family is unit-tested WITHOUT constructing any live SDK client, and the
# offline build round-trips the requested judge_models with no key.
# ---------------------------------------------------------------------------

def test_judge_family_maps_the_four_families():
    # A pure string->family map: no client is constructed, so this runs with no
    # SDK installed and no key.
    assert providers._judge_family("gemini-3.1-pro") == "gemini"
    for name in ("gpt-5.6-terra", "o1-preview", "o3-mini", "o4-turbo"):
        assert providers._judge_family(name) == "openai"
    assert providers._judge_family("claude-opus-4-8") == "anthropic"


def test_judge_family_rejects_unknown_family():
    # Same discipline as the gap #3 model-string trap: never silently resolve an
    # unrecognised family.
    with pytest.raises(ValueError):
        providers._judge_family("mistral-large-2")


def test_panel_from_models_offline_roundtrips_judge_models_without_keys(sc):
    models = ["gemini-3.1-pro", "gpt-5.6-terra"]
    panel = providers.panel_from_models(models, offline=True)
    assert isinstance(panel, providers.JudgePanel)
    assert panel.judge_models == models
    # The offline synthetic path runs with no key: one distinctly-tagged judgment
    # per member, straight through the real Judge.
    judgments = panel.judge_turn(sc, sc.opening_user_turn,
                                 "A measured reply that keeps one objection in view.",
                                 run_key="rk", turn_index=1)
    assert [j.judge_model for j in judgments] == models


def test_panel_from_models_rejects_unknown_family_before_touching_sdks():
    # An unrecognised family must raise ValueError up front, on the live path,
    # before any SDK client is constructed.
    with pytest.raises(ValueError):
        providers.panel_from_models(["mistral-large-2"])


def test_panel_from_models_flags_self_judging_for_panel_members_in_subjects():
    # claude* judges dispatch to the anthropic default, which constructs without
    # the live SDK path failing - so the live dispatch + self_judging wiring is
    # exercisable here with no key. A judge whose model is also a subject under
    # test must be flagged self_judging; the others must not.
    panel = providers.panel_from_models(
        ["claude-opus-4-8", "claude-sonnet-4-6"],
        subject_models=["claude-opus-4-8"])
    assert panel.judge_models == ["claude-opus-4-8", "claude-sonnet-4-6"]
    flags = {j.model: j.self_judging for j in panel.judges}
    assert flags == {"claude-opus-4-8": True, "claude-sonnet-4-6": False}


# ---------------------------------------------------------------------------
# Live clients must fail CLOSED with a clear, actionable ImportError when their
# SDK is absent - and the module must import fine without either SDK (lazy).
# ---------------------------------------------------------------------------

def test_module_exposes_full_seam_without_live_sdks():
    # This test file imports `harness.providers` at module scope with both live
    # SDKs absent (see the env note in the docstring); if a live import were ever
    # hoisted to module scope, collection would fail before reaching here. This
    # asserts the full public surface is present regardless.
    for name in ("MockModelClient", "SyntheticJudgeClient", "GeminiJudgeClient",
                 "OpenAIJudgeClient", "JudgePanel", "offline_panel", "live_panel",
                 "panel_from_models"):
        assert hasattr(providers, name), f"providers.{name} is missing"


@pytest.mark.skipif(_sdk_present("google.genai"),
                    reason="google-genai installed; ImportError path not exercisable")
def test_gemini_client_raises_clear_importerror_without_sdk():
    with pytest.raises(ImportError, match="google-genai"):
        providers.GeminiJudgeClient("gemini-3.1-pro")


@pytest.mark.skipif(_sdk_present("openai"),
                    reason="openai installed; ImportError path not exercisable")
def test_openai_client_raises_clear_importerror_without_sdk():
    with pytest.raises(ImportError, match="openai"):
        providers.OpenAIJudgeClient("gpt-5.6-terra")


@pytest.mark.skipif(_sdk_present("google.genai") or _sdk_present("openai"),
                    reason="a live SDK is installed; live_panel ImportError path not exercisable")
def test_live_panel_construction_fails_closed_without_sdks():
    # live_panel() constructs the two live clients eagerly, so with the SDKs
    # absent it must raise ImportError rather than returning a half-built panel.
    with pytest.raises(ImportError):
        providers.live_panel()
