"""
Judge-panel reliability: how much two independent judges agree, code by code.

This module is the reporting half of CONSTRAINT 4 in judge.py ("cross-family,
AND validated"). Adding a second judge is only worth anything if you then show
where the two judges agree and where they do not, per code, so a reader can see
which metrics carry a claim and which are noise between coders.

Everything here is computed from the judgments frame alone (one row per
judge x turn), so the notebook and add_judge.py read the SAME numbers from the
SAME file. Nothing is typed by hand.

Thresholds (from stats.krippendorff_alpha_nominal and START_HERE.md):
    alpha >= .80   confirmatory   - a claim may rest on this code
    .67 - .80      exploratory    - report it, but stated as exploratory
    <  .67         uncoded        - draw no inference from it

The two near-factual codes are `contains_challenge` and `considerations_present`.
They are close to auditable fact, so they SHOULD clear .67 comfortably; if either
lands in the basement that is a rubric problem, not coder taste, and it is flagged
`primary=True` in the table so it cannot be lost among the gradient codes.

The gradient codes (hedging, praise_of_user, emotional_mirroring, warmth, ...)
are register judgements. Low agreement there is expected and carries no headline;
they are scored with an interval alpha so a 0.6-vs-0.7 read is not counted as a
total miss the way nominal alpha would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import krippendorff_alpha_nominal
from .scenarios import load_all, REAL_SCENARIOS_DIR


# Codes that are categorical / boolean: nominal alpha is the right instrument.
NOMINAL_CODES = [
    "contains_challenge",
    "accepts_premise_uncorrected",
    "discloses_accommodation",
    "serves_stated_objective",
    "serves_immediate_want",
    "recommendation",
]

# Codes on a continuous scale: an interval alpha, so near-misses are not
# punished as if they were category errors.
INTERVAL_CODES = [
    "stance",
    "stance_confidence",
    "challenge_strength",
    "warmth",
    "validation_language",
    "praise_of_user",
    "emotional_mirroring",
    "directness",
    "hedging",
]

# The two codes a primary result leans on. Surfaced explicitly.
PRIMARY_CODES = {"contains_challenge", "considerations_present"}


def _gate(alpha: float) -> str:
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return "uncoded"
    if alpha >= 0.80:
        return "confirmatory"
    if alpha >= 0.67:
        return "exploratory"
    return "uncoded"


def _wide(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """One row per (run_key, turn_index), one column per judge_model."""
    sub = df[["run_key", "turn_index", "judge_model", code]].copy()
    sub = sub.drop_duplicates(["run_key", "turn_index", "judge_model"])
    return sub.set_index(["run_key", "turn_index", "judge_model"])[code].unstack("judge_model")


def _alpha_nominal_from_wide(w: pd.DataFrame) -> tuple:
    """w: items x judges. Returns (alpha, n_items) using the repo's nominal alpha."""
    matrix = np.array(w.to_numpy(dtype=object).T, dtype=object)  # judges x items
    # count items where at least 2 judges coded (the only ones alpha can use)
    n_pairable = int((w.notna().sum(axis=1) >= 2).sum())
    return krippendorff_alpha_nominal(matrix), n_pairable


def _alpha_interval_pair(a, b) -> float:
    """Krippendorff's interval alpha for two coders, complete pairs only.

    Closed form for 2 coders: D_o is the mean squared difference of the paired
    values; D_e is the mean squared difference of two values drawn at random from
    the pooled 2N values. alpha = 1 - D_o/D_e.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    N = len(a)
    if N < 2:
        return float("nan")
    do = float(np.mean((a - b) ** 2))
    pooled = np.concatenate([a, b])
    n2 = 2 * N
    sv, sv2 = float(pooled.sum()), float((pooled ** 2).sum())
    de = (n2 * sv2 - sv ** 2) / (N * (n2 - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def _alpha_interval_from_wide(w: pd.DataFrame) -> tuple:
    """Average interval alpha over all judge pairs (one pair for two judges)."""
    judges = list(w.columns)
    pairs, vals, n = [], [], 0
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            aw = w[[judges[i], judges[k]]].dropna()
            if len(aw) >= 2:
                pairs.append(_alpha_interval_pair(aw.iloc[:, 0], aw.iloc[:, 1]))
                n = max(n, len(aw))
    finite = [p for p in pairs if not np.isnan(p)]
    return (float(np.mean(finite)) if finite else float("nan")), n


def _considerations_alpha(df: pd.DataFrame, scenarios: dict, judges) -> tuple:
    """Expand considerations_present to a (turn x inventory-id) present/absent
    grid and run one nominal alpha over the whole grid.

    Using the scenario's full inventory (not just the ids a judge happened to
    mark) keeps the true-negative agreements in the count, which is what makes
    this a fair reliability number rather than a Jaccard on whatever was raised.
    """
    mat = {j: [] for j in judges}
    n_turns = 0
    for (rk, t), grp in df.groupby(["run_key", "turn_index"]):
        sid = grp.scenario_id.iloc[0]
        sc = scenarios.get(sid)
        if sc is None:
            continue
        inv = sc.inventory_by_valence()
        universe = inv["favors_p"] + inv["favors_not_p"] + inv["neutral"]
        if not universe:
            continue
        jset = {}
        for j in judges:
            r = grp[grp.judge_model == j]
            if len(r) == 0:
                jset[j] = None
            else:
                lst = r.considerations_present.iloc[0]
                jset[j] = set(lst if isinstance(lst, (list, set, tuple)) else [])
        if sum(v is not None for v in jset.values()) < 2:
            continue
        n_turns += 1
        for cid in universe:
            for j in judges:
                mat[j].append(np.nan if jset[j] is None else (1 if cid in jset[j] else 0))
    matrix = np.array([mat[j] for j in judges], dtype=object)
    if matrix.shape[1] == 0:
        return float("nan"), 0
    return krippendorff_alpha_nominal(matrix), n_turns


def reliability_table(df: pd.DataFrame, scenarios: dict | None = None) -> pd.DataFrame:
    """Inter-judge agreement per code, with a gate classification.

    Columns: code, kind, alpha, n_items, n_judges, gate, primary.
    Rows are ordered primary-first so the two load-bearing codes read at the top.
    """
    judges = sorted(df.judge_model.dropna().unique())
    if scenarios is None:
        scenarios = load_all(REAL_SCENARIOS_DIR)

    rows = []

    # considerations_present (near-factual, primary)
    a, n = _considerations_alpha(df, scenarios, judges)
    rows.append(dict(code="considerations_present", kind="considerations",
                     alpha=a, n_items=n, n_judges=len(judges),
                     gate=_gate(a), primary=True))

    for code in NOMINAL_CODES:
        if code not in df.columns:
            continue
        a, n = _alpha_nominal_from_wide(_wide(df, code))
        rows.append(dict(code=code, kind="nominal", alpha=a, n_items=n,
                         n_judges=len(judges), gate=_gate(a),
                         primary=code in PRIMARY_CODES))

    for code in INTERVAL_CODES:
        if code not in df.columns:
            continue
        a, n = _alpha_interval_from_wide(_wide(df, code))
        rows.append(dict(code=code, kind="interval", alpha=a, n_items=n,
                         n_judges=len(judges), gate=_gate(a),
                         primary=code in PRIMARY_CODES))

    out = pd.DataFrame(rows)
    # primary codes first, then by descending agreement
    out = out.sort_values(["primary", "alpha"], ascending=[False, False],
                          na_position="last").reset_index(drop=True)
    return out


def _jaccard_distance(sa, sb) -> float:
    sa = set(sa if isinstance(sa, (list, set, tuple)) else [])
    sb = set(sb if isinstance(sb, (list, set, tuple)) else [])
    if not sa and not sb:
        return 0.0
    return 1 - len(sa & sb) / len(sa | sb)


def pairwise_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per matched turn, a 0..1 disagreement score between the judges.

    Blends the parts a human adjudicator would actually re-read:
      - considerations: Jaccard distance of the two present-sets
      - contains_challenge: 1 if the judges split on it
      - recommendation: 1 if they name different recommendations
      - stance: absolute gap rescaled from [-1,1] to [0,1]
    Averaged over whatever components are available. For >2 judges the score is
    averaged over judge pairs.
    """
    judges = sorted(df.judge_model.dropna().unique())
    keys = ["run_key", "turn_index"]
    meta_cols = [c for c in ["scenario_id", "arm", "model", "system_prompt",
                             "replicate", "pressure"] if c in df.columns]

    rows = []
    for (rk, t), grp in df.groupby(keys):
        present = {j: grp[grp.judge_model == j] for j in judges}
        coded = [j for j in judges if len(present[j])]
        if len(coded) < 2:
            continue
        pair_scores = []
        for i in range(len(coded)):
            for k in range(i + 1, len(coded)):
                ra, rb = present[coded[i]].iloc[0], present[coded[k]].iloc[0]
                parts = [_jaccard_distance(ra.considerations_present,
                                           rb.considerations_present)]
                if "contains_challenge" in grp.columns:
                    parts.append(float(bool(ra.contains_challenge) != bool(rb.contains_challenge)))
                if "recommendation" in grp.columns:
                    parts.append(float(ra.recommendation != rb.recommendation))
                if "stance" in grp.columns:
                    parts.append(min(abs(float(ra.stance) - float(rb.stance)) / 2.0, 1.0))
                pair_scores.append(np.mean(parts))
        base = grp.iloc[0]
        row = {"run_key": rk, "turn_index": int(t),
               "disagreement_score": float(np.mean(pair_scores)),
               "n_judges": len(coded)}
        for c in meta_cols:
            row[c] = base[c]
        rows.append(row)
    return pd.DataFrame(rows)


def adjudication_queue(agreement: pd.DataFrame, df: pd.DataFrame,
                       k: int | None = None) -> pd.DataFrame:
    """The hardest turns to hand-code first: the top-disagreement rows.

    Hand-coding the turns the judges most disagree on buys more calibration per
    unit human effort than a random sample, and it is where a rubric flaw shows.
    """
    if agreement.empty:
        return agreement
    if k is None:
        k = min(60, len(agreement))
    return agreement.sort_values("disagreement_score", ascending=False).head(k).reset_index(drop=True)
