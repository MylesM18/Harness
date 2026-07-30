"""
Scenario loading and validation.

The loader is deliberately strict. A scenario that fails validation is not
loaded, because a malformed scenario silently produces plausible-looking numbers
that mean nothing.

The validation gates encode the design's non-negotiables:

  MIRROR SYMMETRY   the pro and con user scripts must be generated from one
                    template. If they are authored separately, wording differences
                    confound the stance difference and the whole identification
                    strategy is gone.

  BALANCED INVENTORY the consideration inventory must contain at least three
                    items on each side. Asymmetric attrition is a difference of
                    two proportions; with one item on a side, that proportion is
                    0 or 1 and the metric is noise.

  LIVE PROPOSITION   the focal proposition must be genuinely contestable. If
                    reasonable people all land the same way, the con arm is a
                    request to defend nonsense and the model is right to refuse
                    it, which reads as sycophancy resistance when it is just
                    knowing the answer.

  DIVERGING HORIZON  the stated objective must come apart from what the user
                    starts wanting. If they never diverge, the horizon metric
                    measures nothing.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from .schema import Scenario, Consideration, UserTurn


# The scenario corpus is split into two buckets. `real-scenarios/` is the study
# set (the unit of generalisation - see stats.py); `test-scenarios/` holds the
# small demo fixtures the pipeline and test suite run against. Keep the paths in
# one place so a re-layout is a one-line change, not a codebase-wide grep.
SCENARIOS_ROOT = Path(__file__).resolve().parents[1] / "scenarios"
REAL_SCENARIOS_DIR = SCENARIOS_ROOT / "real-scenarios"
TEST_SCENARIOS_DIR = SCENARIOS_ROOT / "test-scenarios"


class ScenarioValidationError(ValueError):
    pass


def _validate(sc: Scenario) -> list[str]:
    problems = []

    by_val = sc.inventory_by_valence()
    if len(by_val["favors_p"]) < 3:
        problems.append(f"{sc.id}: needs >=3 considerations favouring P, has {len(by_val['favors_p'])}")
    if len(by_val["favors_not_p"]) < 3:
        problems.append(f"{sc.id}: needs >=3 considerations favouring NOT-P, has {len(by_val['favors_not_p'])}")

    if not sc.user_turns_pro:
        problems.append(f"{sc.id}: no user turn ladder")

    levels = [t.escalation_level for t in sc.user_turns_pro]
    if levels != sorted(levels):
        problems.append(f"{sc.id}: escalation levels are not monotonic: {levels}")

    # Every turn must be mirror-safe. Two legitimate routes: carry the stance in
    # a placeholder, or carry no stance and declare it. A turn that asserts a
    # position in bare prose is neither, and it breaks the mirror silently.
    for t in sc.user_turns_pro:
        has_placeholder = "{P}" in t.text or "{NOT_P}" in t.text
        if not has_placeholder and not t.stance_free:
            problems.append(
                f"{sc.id} turn {t.index}: no {{P}}/{{NOT_P}} placeholder and not "
                "declared stance_free. Either carry the stance in a placeholder so "
                "the arms mirror, or set stance_free: true to assert this turn takes "
                "no side.")
        if has_placeholder and t.stance_free:
            problems.append(
                f"{sc.id} turn {t.index}: declared stance_free but contains a stance "
                "placeholder. One or the other.")

    if not sc.stated_objective:
        problems.append(f"{sc.id}: missing stated_objective, horizon metric undefined")

    if not sc.nosource_framing:
        problems.append(f"{sc.id}: missing nosource_framing, speaker-free floor cannot be estimated")

    if not sc.neutral_turns:
        problems.append(f"{sc.id}: missing neutral_turns, length-matched placebo unavailable")

    return problems


def load_scenario(path: str | Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())

    sc = Scenario(
        id=raw["id"],
        domain=raw["domain"],
        title=raw["title"],
        proposition_p=raw["proposition_p"],
        proposition_not_p=raw["proposition_not_p"],
        stated_objective=raw["stated_objective"],
        opening_user_turn=raw["opening_user_turn"],
        user_turns_pro=[UserTurn(**t) for t in raw["user_turns"]],
        considerations=[Consideration(**c) for c in raw["considerations"]],
        nosource_framing=raw.get("nosource_framing", ""),
        neutral_turns=[UserTurn(**t) for t in raw.get("neutral_turns", [])],
        notes=raw.get("notes", ""),
    )
    problems = _validate(sc)
    if problems:
        raise ScenarioValidationError("\n".join(problems))
    return sc


def load_all(directory: str | Path = SCENARIOS_ROOT,
             strict: bool = True) -> dict[str, Scenario]:
    # rglob, not glob: scenarios live in sub-buckets (test/real), so the loader
    # must descend. This also means load_all(SCENARIOS_ROOT) picks up everything.
    out, errs = {}, []
    for p in sorted(Path(directory).rglob("*.yaml")):
        try:
            sc = load_scenario(p)
            out[sc.id] = sc
        except ScenarioValidationError as e:
            errs.append(str(e))
    if errs and strict:
        raise ScenarioValidationError("\n\n".join(errs))
    return out


def build_turns(sc: Scenario, arm: str, n_turns: int, pressure: str = "gradual") -> list[str]:
    """
    Render the user side of the conversation for one arm.

    pressure="gradual" walks the escalation ladder as authored.
    pressure="abrupt" holds level 0 until turn 3, then jumps straight to the top
    rung and stays there - the classic single-pivot paradigm, included so results
    can be compared against the existing literature rather than floating free of it.
    pressure="none" uses the neutral turn list.
    """
    if arm == "neutral" or pressure == "none":
        pool = sc.neutral_turns or sc.user_turns_pro
        return [sc.fill(t.text, "neutral") for t in pool][:n_turns]

    ladder = sorted(sc.user_turns_pro, key=lambda t: t.index)

    if pressure == "abrupt":
        top = max(ladder, key=lambda t: t.escalation_level)
        low = [t for t in ladder if t.escalation_level == 0] or [ladder[0]]
        seq = (low * 3)[:2] + [top] * (n_turns - 2)
    else:
        seq = [ladder[min(i, len(ladder) - 1)] for i in range(n_turns)]

    render_arm = "pro" if arm in ("pro", "nosource_pro") else "con"
    texts = [sc.fill(t.text, render_arm) for t in seq][:n_turns]

    if arm.startswith("nosource"):
        # Strip first-person conviction; present the identical claims as found
        # material. Content held constant, speaker removed.
        texts = [sc.nosource_framing.replace("{CLAIM}", t) for t in texts]

    return texts
