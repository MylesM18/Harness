"""
Synthetic judgment generator.

Purpose: the analysis pipeline, the figures, and the tests all have to be
runnable and reviewable before a single API credit is spent, and by anyone who
clones the repo without a key.

This is a data generator with known ground truth, not a model of anything. It
exists so that:

  - the metric code can be validated against a signal that was planted on
    purpose. If asymmetric attrition is set to zero in the generator and AAI
    comes back non-zero, the metric is broken, and that is a much cheaper thing
    to discover here than after a thousand real conversations.
  - figures render on `git clone && python scripts/demo_synthetic.py`.
  - the power analysis is grounded. Vary n_scenarios and n_replicates here, run
    the real statistics on the output, and read off what design is needed to
    detect the effect size that matters. Do this BEFORE the real study.

Two profiles are provided. Neither is a claim about any released model.

  "holds"   content invariant across arms, delivery adapts, friction persists.
            The behaviour the premise argues for: vary approach, hold standard.
  "drifts"  no flip ever occurs and no false statement is made, but friction
            decays, the inventory attrits asymmetrically, and stance slides.
            The failure that turn-of-flip metrics cannot see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Scenario


PROFILES = {
    "holds": dict(
        stance_drift=0.004,        # stance movement toward the user, per turn
        floor_drift=0.006,         # movement from context alone (no speaker)
        friction_open=0.82,
        friction_decay=0.012,      # per-turn proportional decay
        aai_open=0.02,
        aai_drift=0.004,
        coverage_open=0.72,
        coverage_decay=0.006,
        warmth_gap=0.22,           # pro-vs-con delivery divergence: allowed to be large
        content_gap=0.03,          # pro-vs-con content divergence: should stay small
        disclosure=0.31,
        horizon=0.86,
    ),
    "drifts": dict(
        stance_drift=0.052,
        floor_drift=0.014,
        friction_open=0.78,
        friction_decay=0.115,
        aai_open=0.04,
        aai_drift=0.031,
        coverage_open=0.70,
        coverage_decay=0.030,
        warmth_gap=0.26,
        content_gap=0.24,
        disclosure=0.06,
        horizon=0.51,
    ),
}


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_judgments(scenarios: dict[str, Scenario],
                       models: dict[str, str] | None = None,
                       n_turns: int = 12,
                       n_replicates: int = 4,
                       noise: float = 0.06,
                       seed: int = 11) -> pd.DataFrame:
    """
    Emit a judgments frame with the same columns the real judge produces, so
    metrics.py and plots.py cannot tell the difference.

    models maps a display name to a profile key, e.g.
        {"model-A": "holds", "model-B": "drifts"}
    """
    models = models or {"model-A": "holds", "model-B": "drifts"}
    rng = np.random.default_rng(seed)
    rows = []

    arms = ["pro", "con", "neutral", "nosource_pro", "nosource_con"]

    for model_name, profile_key in models.items():
        p = PROFILES[profile_key]
        for sid, sc in scenarios.items():
            inv = sc.inventory_by_valence()
            sc_effect = rng.normal(0, 0.35)          # scenario random effect
            for rep in range(n_replicates):
                for arm in arms:
                    for t in range(1, n_turns + 1):

                        # direction the user is pushing in this arm
                        if arm in ("pro", "nosource_pro"):
                            sgn = +1.0
                        elif arm in ("con", "nosource_con"):
                            sgn = -1.0
                        else:
                            sgn = 0.0

                        under_pressure = arm in ("pro", "con")
                        speakerless = arm.startswith("nosource")

                        drift = p["floor_drift"] if speakerless else p["stance_drift"]
                        if arm == "neutral":
                            drift = p["floor_drift"] * 0.4

                        stance = sgn * (drift * t * (1 + 0.25 * sc_effect))
                        stance += rng.normal(0, noise)
                        stance = float(np.clip(stance, -1, 1))

                        # friction only decays where there is a user to defer to
                        if under_pressure:
                            fr = p["friction_open"] * np.exp(-p["friction_decay"] * (t - 1))
                        else:
                            fr = p["friction_open"] * np.exp(-0.004 * (t - 1))
                        fr = float(np.clip(fr + rng.normal(0, 0.04), 0.01, 0.99))
                        challenge = rng.random() < fr

                        # inventory retention, asymmetric under pressure
                        aai_now = (p["aai_open"] + p["aai_drift"] * (t - 1)) if under_pressure else 0.0
                        base_cov = p["coverage_open"] - p["coverage_decay"] * (t - 1)
                        base_cov = float(np.clip(base_cov, 0.05, 1.0))

                        user_side = inv["favors_p"] if sgn >= 0 else inv["favors_not_p"]
                        other_side = inv["favors_not_p"] if sgn >= 0 else inv["favors_p"]

                        present = []
                        for cid in user_side:
                            if rng.random() < np.clip(base_cov + aai_now / 2, 0, 1):
                                present.append(cid)
                        for cid in other_side:
                            if rng.random() < np.clip(base_cov - aai_now / 2, 0, 1):
                                present.append(cid)
                        for cid in inv["neutral"]:
                            if rng.random() < base_cov:
                                present.append(cid)

                        # delivery channel: adapts to the user in both profiles
                        warm_base = 0.42 + (p["warmth_gap"] / 2) * sgn * min(t / n_turns, 1.0)
                        validation = np.clip(warm_base + rng.normal(0, 0.07), 0, 1)
                        warmth = np.clip(warm_base + 0.05 + rng.normal(0, 0.07), 0, 1)
                        praise = np.clip(0.12 + 0.10 * sgn * (t / n_turns) + rng.normal(0, 0.05), 0, 1)
                        mirroring = np.clip(0.30 + (p["warmth_gap"] / 2) * sgn + rng.normal(0, 0.06), 0, 1)

                        # content channel bleed: this is the thing that should NOT move
                        bleed = p["content_gap"] * sgn * (t / n_turns)
                        directness = np.clip(0.62 - abs(bleed) + rng.normal(0, 0.06), 0, 1)
                        hedging = np.clip(0.28 + abs(bleed) * 0.5 + rng.normal(0, 0.06), 0, 1)

                        serves_obj = rng.random() < (p["horizon"] - 0.012 * (t - 1) * under_pressure)
                        serves_want = rng.random() < _logistic(0.5 + 0.10 * t * under_pressure)

                        rows.append(dict(
                            scenario_id=sid, arm=arm, model=model_name,
                            replicate=rep, turn_index=t,
                            conversation_id=f"{model_name}|{sid}|{arm}|{rep}",
                            stance=stance,
                            stance_confidence=float(np.clip(0.8 + rng.normal(0, .08), 0, 1)),
                            considerations_present=present,
                            contains_challenge=bool(challenge),
                            challenge_strength=float(fr if challenge else 0.0),
                            recommendation=("toward_p" if stance > .25 else
                                            "toward_not_p" if stance < -.25 else "neither"),
                            accepts_premise_uncorrected=bool(rng.random() < 0.12 + 0.02 * t * under_pressure),
                            warmth=float(warmth), validation_language=float(validation),
                            praise_of_user=float(praise), emotional_mirroring=float(mirroring),
                            directness=float(directness), hedging=float(hedging),
                            discloses_accommodation=bool(rng.random() < p["disclosure"]),
                            serves_stated_objective=bool(serves_obj),
                            serves_immediate_want=bool(serves_want),
                            persona="none",
                            judge_model="synthetic",
                        ))

    return pd.DataFrame(rows)
