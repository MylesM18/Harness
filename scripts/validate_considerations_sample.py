#!/usr/bin/env python3
"""
Human-coding validation for `considerations_present`  (CONSTRAINT 4 in judge.py).

The judge extracts, per turn, WHICH pre-registered considerations appear. The
inter-judge reliability of that code is atomised to a (turn x consideration-cell)
present/absent grid (see harness/panel.py::_considerations_alpha). This script
prepares — and later scores — a HUMAN validation of that same grid, so we can
report a confirmatory human-vs-machine Krippendorff's alpha rather than only the
machine-vs-machine number.

The human coder is the system-under-validation's independent check: they must be
given the IDENTICAL task the judge had (single assistant turn + the one user turn
before it + the consideration's detection_hint), BLIND to model / arm / turn index
/ pressure / the judges' own codes. That blinding is why the answer key is written
to a SEPARATE file the coder never opens.

    Unit    one (run_key, turn_index, consideration_id) cell, binary present/absent
    Strata  model family {opus, sonnet}  x  judge-agreement bucket {both_yes, split, both_no}
    Alloc   equal, --n-per-stratum (default 25)  ->  6 strata x 25 = n=150
    Seed    fixed, recorded in the manifest and stamped into both output files

Two subcommands:

    sample   draw the stratified subsample, emit the blank coding sheet + the
             hidden key + a provenance manifest.
    score    after the coder returns the filled sheet, compute human-vs-consensus
             Krippendorff's alpha (+ % agreement), a 3-coder alpha that includes
             the split cells honestly, and the per-stratum / per-model breakdown.

This file is standalone. It never imports from, and never modifies, run_study.py
or analyze.py. It reuses only harness.stats.krippendorff_alpha_nominal so the
human figure is produced by the SAME estimator as the inter-judge figure.

Usage
-----
    python scripts/validate_considerations_sample.py sample
    python scripts/validate_considerations_sample.py sample --n-per-stratum 25 --seed 20260804
    # ... human fills coding_sheet_v1.csv -> coding_sheet_v1.filled.csv ...
    python scripts/validate_considerations_sample.py score \
        --sheet data/human_validation/coding_sheet_v1.filled.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGMENTS = os.path.join(REPO, "data", "judgments.stageD_full.jsonl")
CACHE_DIR = os.path.join(REPO, "data", "cache")
SCEN_DIRS = [
    os.path.join(REPO, "scenarios", "real-scenarios"),
    os.path.join(REPO, "scenarios", "test-scenarios"),
]
OUT_DIR = os.path.join(REPO, "data", "human_validation")

# The two judges of record for stageD_full (report.json::_scope).
JUDGE_PRIMARY = "gpt-5.6-terra"          # primary judge
JUDGE_SECONDARY = "gemini-3.1-pro-preview"

BUCKETS = ["both_yes", "split", "both_no"]


# ---------------------------------------------------------------------------
# scenario inventories
# ---------------------------------------------------------------------------
def load_scenarios() -> dict:
    """scenario_id -> {title, prop_p, prop_not_p, considerations:{cid:{text,hint,valence}}, universe:[cid]}.

    Uses PyYAML when available (the repo's own parser); falls back to a light
    regex reader so `sample` works even in a bare interpreter.
    """
    import importlib.util
    loader = _load_scen_yaml if importlib.util.find_spec("yaml") else _load_scen_regex

    out = {}
    for d in SCEN_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            sid, rec = loader(p)
            if sid:
                out[sid] = rec
    return out


def _load_scen_yaml(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw or "considerations" not in raw:
        return None, None
    cons = {}
    for c in raw["considerations"]:
        cons[c["id"]] = {
            "text": (c.get("text") or "").strip(),
            "hint": (c.get("detection_hint") or "").strip(),
            "valence": c.get("valence", "neutral"),
        }
    universe = _universe_order(cons)
    return raw["id"], {
        "title": (raw.get("title") or "").strip(),
        "prop_p": (raw.get("proposition_p") or "").strip(),
        "prop_not_p": (raw.get("proposition_not_p") or "").strip(),
        "considerations": cons,
        "universe": universe,
    }


def _load_scen_regex(path):                               # pragma: no cover
    import re
    with open(path) as f:
        txt = f.read()

    def _scalar(key):
        m = re.search(rf'^{key}:\s*(?:"([^"]*)"|>?\s*\n?\s*(.*))', txt, re.MULTILINE)
        if not m:
            return ""
        return (m.group(1) or m.group(2) or "").strip()

    sid = _scalar("id")
    cons = {}
    block = txt.split("considerations:", 1)[-1]
    for m in re.finditer(
        r'-\s*id:\s*(C\d+)\s*\n\s*valence:\s*(\w+)\s*\n\s*text:\s*"([^"]*)"\s*\n\s*detection_hint:\s*"([^"]*)"',
        block,
    ):
        cid, val, text, hint = m.groups()
        cons[cid] = {"text": text.strip(), "hint": hint.strip(), "valence": val}
    if not sid or not cons:
        return None, None
    return sid, {
        "title": _scalar("title"),
        "prop_p": _scalar("proposition_p"),
        "prop_not_p": _scalar("proposition_not_p"),
        "considerations": cons,
        "universe": _universe_order(cons),
    }


def _universe_order(cons: dict) -> list:
    """favors_p, then favors_not_p, then neutral — mirrors panel.inventory_by_valence."""
    order = {"favors_p": 0, "favors_not_p": 1, "neutral": 2}
    return sorted(cons, key=lambda c: (order.get(cons[c]["valence"], 3), c))


# ---------------------------------------------------------------------------
# judgments -> per-cell judge codes and agreement buckets
# ---------------------------------------------------------------------------
def load_cells(scenarios: dict):
    """Yield one dict per (run_key, turn_index, cid): the two judge codes + bucket.

    Only turns coded by BOTH judges of record and belonging to a known scenario
    are used — identical to the population panel.py atomises for the inter-judge
    alpha, so the human sample is drawn from exactly that grid.
    """
    by_turn = defaultdict(dict)   # (rk, ti) -> judge -> (set, model, scenario_id)
    with open(JUDGMENTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            j = d.get("judge_model")
            if j not in (JUDGE_PRIMARY, JUDGE_SECONDARY):
                continue
            s = set(d.get("considerations_present") or [])
            by_turn[(d["run_key"], d["turn_index"])][j] = (s, d.get("model"), d.get("scenario_id"))

    cells = []
    for (rk, ti), jd in by_turn.items():
        if JUDGE_PRIMARY not in jd or JUDGE_SECONDARY not in jd:
            continue
        s_pri, model, sid = jd[JUDGE_PRIMARY]
        s_sec, _, _ = jd[JUDGE_SECONDARY]
        sc = scenarios.get(sid)
        if not sc:
            continue
        for cid in sc["universe"]:
            in_pri, in_sec = cid in s_pri, cid in s_sec
            if in_pri and in_sec:
                bucket = "both_yes"
            elif in_pri or in_sec:
                bucket = "split"
            else:
                bucket = "both_no"
            cells.append(dict(
                run_key=rk, turn_index=ti, scenario_id=sid, model=model,
                consideration_id=cid,
                judge_gpt5_code=int(in_pri), judge_gemini_code=int(in_sec),
                bucket=bucket,
            ))
    return cells


# ---------------------------------------------------------------------------
# transcript reconstruction (exactly what the judge saw)
# ---------------------------------------------------------------------------
_cache_memo: dict = {}


def turn_text(run_key: str, turn_index: int):
    """(user_message, assistant_response) for turn_index (1-based), from the cache."""
    if run_key not in _cache_memo:
        path = os.path.join(CACHE_DIR, f"{run_key}.json")
        with open(path) as f:
            d = json.load(f)
        users = [m["content"] for m in d["messages"] if m["role"] == "user"]
        _cache_memo[run_key] = (users, d["assistant_turns"])
    users, assistants = _cache_memo[run_key]
    i = turn_index - 1
    if i < 0 or i >= len(assistants) or i >= len(users):
        raise IndexError(f"{run_key} turn {turn_index}: out of range")
    return users[i].strip(), assistants[i].strip()


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------
def cmd_sample(args):
    import random

    scenarios = load_scenarios()
    cells = load_cells(scenarios)

    strata = defaultdict(list)                     # (model, bucket) -> [cell]
    for c in cells:
        strata[(c["model"], c["bucket"])].append(c)

    models = sorted({c["model"] for c in cells})
    rng = random.Random(args.seed)

    chosen, shortfalls = [], []
    for model in models:
        for bucket in BUCKETS:
            pool = strata.get((model, bucket), [])
            k = min(args.n_per_stratum, len(pool))
            if k < args.n_per_stratum:
                shortfalls.append(dict(model=model, bucket=bucket,
                                       requested=args.n_per_stratum, available=len(pool)))
            for c in rng.sample(pool, k):
                c = dict(c, stratum=f"{model}|{bucket}")
                chosen.append(c)

    rng.shuffle(chosen)                            # kill any strata/model ordering signal
    for i, c in enumerate(chosen, start=1):
        c["row_id"] = i

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sheet_path = os.path.join(OUT_DIR, "coding_sheet_v1.csv")
    key_path = os.path.join(OUT_DIR, "hidden_key_v1.csv")
    man_path = os.path.join(OUT_DIR, "sample_manifest.json")

    # blank coding sheet — coder-visible ONLY. No condition, no judge codes.
    with open(sheet_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "scenario_topic", "proposition_p", "proposition_not_p",
                    "user_message", "assistant_response",
                    "consideration_id", "detection_hint",
                    "human_code", "notes"])
        for c in chosen:
            sc = scenarios[c["scenario_id"]]
            user_msg, asst = turn_text(c["run_key"], c["turn_index"])
            w.writerow([c["row_id"], sc["title"], sc["prop_p"], sc["prop_not_p"],
                        user_msg, asst,
                        c["consideration_id"], sc["considerations"][c["consideration_id"]]["hint"],
                        "", ""])                    # human_code left EMPTY

    # hidden key — researcher-only. Never given to the coder while coding.
    with open(key_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "run_key", "turn_index", "scenario_id", "model",
                    "consideration_id", "valence", "consideration_full_text",
                    "judge_gpt5_code", "judge_gemini_code",
                    "bucket", "stratum", "consensus_code"])
        for c in chosen:
            sc = scenarios[c["scenario_id"]]
            consensus = {"both_yes": 1, "both_no": 0}.get(c["bucket"], "")   # split -> undefined
            w.writerow([c["row_id"], c["run_key"], c["turn_index"], c["scenario_id"],
                        c["model"], c["consideration_id"],
                        sc["considerations"][c["consideration_id"]]["valence"],
                        sc["considerations"][c["consideration_id"]]["text"],
                        c["judge_gpt5_code"], c["judge_gemini_code"],
                        c["bucket"], c["stratum"], consensus])

    counts = defaultdict(int)
    for c in chosen:
        counts[c["stratum"]] += 1
    manifest = dict(
        created_utc=stamp, seed=args.seed, n_per_stratum=args.n_per_stratum,
        n_total=len(chosen), unit="(run_key, turn_index, consideration_id) cell",
        code="considerations_present", present_definition_source="harness/judge.py JUDGE_TEMPLATE",
        judges=dict(primary=JUDGE_PRIMARY, secondary=JUDGE_SECONDARY),
        strata=dict(sorted(counts.items())),
        population_cells=len(cells),
        shortfalls=shortfalls,
        source_files=dict(judgments=os.path.relpath(JUDGMENTS, REPO),
                          cache=os.path.relpath(CACHE_DIR, REPO)),
        outputs=dict(coding_sheet=os.path.relpath(sheet_path, REPO),
                     hidden_key=os.path.relpath(key_path, REPO)),
    )
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"seed={args.seed}  n={len(chosen)}  ({args.n_per_stratum} x {len(models)} models x {len(BUCKETS)} buckets)")
    for k in sorted(counts):
        print(f"  {k:32s} {counts[k]}")
    if shortfalls:
        print("SHORTFALLS (stratum pool smaller than requested):")
        for s in shortfalls:
            print(f"  {s['model']}|{s['bucket']}: wanted {s['requested']}, had {s['available']}")
    print(f"\ncoding sheet : {os.path.relpath(sheet_path, REPO)}")
    print(f"hidden key   : {os.path.relpath(key_path, REPO)}   <-- do NOT open while coding")
    print(f"manifest     : {os.path.relpath(man_path, REPO)}")


# ---------------------------------------------------------------------------
# score  (run after the coder returns the filled sheet)
# ---------------------------------------------------------------------------
def cmd_score(args):
    import numpy as np
    sys.path.insert(0, REPO)
    from harness.stats import krippendorff_alpha_nominal

    key = {r["row_id"]: r for r in _read_csv(args.key)}
    sheet = _read_csv(args.sheet)

    rows, unlabeled, bad = [], 0, 0
    for r in sheet:
        rid = r["row_id"]
        raw = (r.get("human_code") or "").strip()
        if raw == "":
            unlabeled += 1
            continue
        hv = _parse_code(raw)
        if hv is None:
            bad += 1
            print(f"  ! row {rid}: uninterpretable human_code {raw!r}", file=sys.stderr)
            continue
        k = key.get(rid)
        if not k:
            print(f"  ! row {rid}: not in hidden key", file=sys.stderr)
            continue
        rows.append(dict(
            row_id=rid, human=hv,
            gpt5=int(k["judge_gpt5_code"]), gemini=int(k["judge_gemini_code"]),
            bucket=k["bucket"], model=k["model"], stratum=k["stratum"],
            consensus=(None if k["consensus_code"] == "" else int(k["consensus_code"])),
        ))

    n = len(rows)
    print(f"scored rows: {n}   (unlabeled: {unlabeled}, unparseable: {bad})")
    if n == 0:
        print("nothing to score.")
        return

    def alpha(coder_lists):
        return krippendorff_alpha_nominal(np.array(coder_lists, dtype=object))

    def pct_agree(pairs):
        return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")

    # (1) headline: human vs judge CONSENSUS, over cells where the judges agreed.
    agree = [r for r in rows if r["consensus"] is not None]
    hc = [(r["human"], r["consensus"]) for r in agree]
    a_consensus = alpha([[h for h, _ in hc], [c for _, c in hc]]) if len(hc) >= 2 else float("nan")
    p_consensus = pct_agree(hc)

    # (2) companion: 3-coder alpha over ALL cells — includes split cells honestly.
    a_three = alpha([[r["gpt5"] for r in rows],
                     [r["gemini"] for r in rows],
                     [r["human"] for r in rows]])

    # (3) diagnostics: human vs each judge individually, over all cells.
    a_gpt5 = alpha([[r["gpt5"] for r in rows], [r["human"] for r in rows]])
    a_gem = alpha([[r["gemini"] for r in rows], [r["human"] for r in rows]])
    p_gpt5 = pct_agree([(r["human"], r["gpt5"]) for r in rows])
    p_gem = pct_agree([(r["human"], r["gemini"]) for r in rows])

    gate = "confirmatory" if a_consensus >= 0.80 else ("exploratory" if a_consensus >= 0.67 else "uncoded")

    # bootstrap CI for the headline consensus alpha (item resample; clustering caveat noted).
    ci = _bootstrap_alpha(hc, alpha, seed=args.seed, reps=args.boot) if len(hc) >= 2 else (float("nan"), float("nan"))

    print("\n=== considerations_present: human validation ===")
    print(f"(1) human vs judge-consensus  alpha = {a_consensus:.3f}   "
          f"[boot95 {ci[0]:.3f}, {ci[1]:.3f}]   %agree = {p_consensus:.1%}   n={len(hc)} (agreement cells)")
    print(f"    gate @0.80 -> {gate.upper()}")
    print(f"(2) 3-coder alpha (gpt5, gemini, human), all cells = {a_three:.3f}   n={n}")
    print(f"(3) human vs gpt-5.6-terra   alpha = {a_gpt5:.3f}  %agree = {p_gpt5:.1%}")
    print(f"    human vs gemini-3.1-pro  alpha = {a_gem:.3f}  %agree = {p_gem:.1%}")

    # per-stratum: %agreement (consensus buckets) / adjudication direction (split)
    print("\n--- per stratum ---")
    strata = defaultdict(list)
    for r in rows:
        strata[r["stratum"]].append(r)
    per_stratum = {}
    for s in sorted(strata):
        rs = strata[s]
        bucket = rs[0]["bucket"]
        if bucket == "split":
            present = sum(r["human"] == 1 for r in rs)
            per_stratum[s] = dict(n=len(rs), bucket=bucket,
                                  human_present_rate=present / len(rs), consensus="undefined")
            print(f"  {s:32s} n={len(rs):3d}  split: human=present on {present}/{len(rs)} "
                  f"({present/len(rs):.0%})  [consensus undefined]")
        else:
            agr = sum(r["human"] == r["consensus"] for r in rs)
            per_stratum[s] = dict(n=len(rs), bucket=bucket, pct_agree=agr / len(rs))
            print(f"  {s:32s} n={len(rs):3d}  {bucket}: human matches consensus {agr}/{len(rs)} "
                  f"({agr/len(rs):.0%})")

    report = dict(
        n_scored=n, unlabeled=unlabeled, unparseable=bad,
        human_vs_consensus=dict(alpha=a_consensus, boot95_lo=ci[0], boot95_hi=ci[1],
                                pct_agree=p_consensus, n=len(hc), gate=gate, threshold=0.80),
        three_coder_alpha=a_three,
        human_vs_gpt5=dict(alpha=a_gpt5, pct_agree=p_gpt5),
        human_vs_gemini=dict(alpha=a_gem, pct_agree=p_gem),
        per_stratum=per_stratum,
    )
    out = os.path.join(OUT_DIR, "validation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


def _bootstrap_alpha(pairs, alpha_fn, seed, reps):
    import random
    rng = random.Random(seed)
    N = len(pairs)
    vals = []
    for _ in range(reps):
        samp = [pairs[rng.randrange(N)] for _ in range(N)]
        a = alpha_fn([[h for h, _ in samp], [c for _, c in samp]])
        if a == a:                                   # not NaN
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi


def _parse_code(raw: str):
    t = raw.strip().lower()
    if t in ("1", "y", "yes", "present", "p", "true", "t"):
        return 1
    if t in ("0", "n", "no", "absent", "a", "false", "f"):
        return 0
    return None


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw the stratified subsample + emit sheet/key/manifest")
    s.add_argument("--n-per-stratum", type=int, default=25)
    s.add_argument("--seed", type=int, default=20260804)
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="score the returned filled coding sheet")
    c.add_argument("--sheet", required=True, help="the FILLED coding sheet CSV")
    c.add_argument("--key", default=os.path.join(OUT_DIR, "hidden_key_v1.csv"))
    c.add_argument("--seed", type=int, default=20260804, help="bootstrap seed")
    c.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
