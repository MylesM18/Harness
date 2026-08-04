#!/usr/bin/env python3
"""
Add a second judge to a study that has already run.

    python scripts/add_judge.py \
        --judge gemini:gemini-3.1-pro-preview \
        --judgments data/judgments.stageD_full.jsonl

Generation is NOT re-run. Every conversation is already on disk in data/cache/,
so this only pays for judging. That is the whole point of the cache: the
expensive half of a study is reusable, and adding a second opinion afterwards
costs a fraction of the original run.

What it does
------------
1. Reads the existing judgments file to learn which (run_key, turn) pairs exist
   and which judge produced them.
2. Reloads each conversation from data/cache/ and re-judges every turn with the
   new judge, blind and single-turn, exactly as the first judge saw it.
3. Appends the new rows to the SAME file. Nothing is overwritten and nothing is
   averaged - both judges' codings coexist, which is what the per-judge analysis
   expects.
4. Prints the reliability table, writes the adjudication queue + reliability CSV,
   and reports actual token usage so the judging cost is an invoice, not a guess.

Resumable: turns already judged by the new judge (in THIS file) are skipped, so
an interrupted run costs nothing to restart.

Adapted to the live repo: the panel lives in harness.providers (JudgePanel wraps
harness.judge.Judge instances); the reliability helpers live in harness.panel;
model strings are UNVERIFIED defaults - check the live model list before spending
(e.g. `client.models.list()`), a wrong-but-plausible id is the silent-aliasing
trap the runbook warns about. `gemini-3.1-pro` is NOT a live id; the only 3.1 Pro
is `gemini-3.1-pro-preview`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Auto-load a repo-root .env (same contract as scripts/run_study.py): the LIVE
# judge keys can live in .env instead of being exported by hand. Already-exported
# vars win (override=False). Offline paths need neither key nor dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from harness.scenarios import load_all
from harness.providers import GeminiJudgeClient, OpenAIJudgeClient, _judge_family
from harness.judge import Judge, judgment_to_row
from harness.panel import (reliability_table, pairwise_agreement,
                           adjudication_queue)
from harness.schema import RunSpec


REQUIRED_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}
SDK_PKG = {"gemini": "google-genai", "openai": "openai", "anthropic": "anthropic"}


def parse_judge(spec: str) -> tuple:
    """`provider:model` or a bare model id -> (family, model_id).

    The model id is the judge_model tag written to every row, so it must match
    the convention of the existing file (bare id, e.g. 'gpt-5.6-terra')."""
    if ":" in spec:
        prov, model = spec.split(":", 1)
        family = _judge_family(model)  # validate id against family; ignore prov label
    else:
        model = spec
        family = _judge_family(model)
    return family, model


def build_judge(family: str, model: str) -> Judge:
    if family == "gemini":
        return Judge(model=model, client=GeminiJudgeClient(model))
    if family == "openai":
        return Judge(model=model, client=OpenAIJudgeClient(model))
    return Judge(model=model)  # anthropic default client


def preflight(family: str, model: str) -> list:
    """Check the key and the SDK before a paid run rather than 400 turns in.

    Does NOT make a network call: constructing the client for gemini/openai only
    wires up the SDK + retry options, so this stays free."""
    problems = []
    env = REQUIRED_ENV.get(family)
    if env and not os.environ.get(env):
        problems.append(f"{env} is not set (put it in .env or export it)")
    try:
        build_judge(family, model)
    except ImportError as e:
        problems.append(f"SDK missing: pip install {SDK_PKG.get(family, family)} ({e})")
    except Exception as e:                                   # noqa: BLE001
        problems.append(f"{type(e).__name__}: {e}")
    return problems


def spec_from_row(row) -> RunSpec:
    """Rebuild the RunSpec from a judgment row so judgment_to_row works."""
    return RunSpec(
        scenario_id=row["scenario_id"], arm=row["arm"],
        pressure=row.get("pressure", "gradual"), model=row["model"],
        system_prompt=row.get("system_prompt", "neutral"),
        persona=row.get("persona", "none"),
        replicate=int(row["replicate"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True,
                    help="provider:model or bare model, e.g. gemini:gemini-3.1-pro-preview")
    ap.add_argument("--judgments", default="data/judgments.stageD_full.jsonl")
    ap.add_argument("--scenarios", default="scenarios")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--price-in", type=float, default=2.0,
                    help="USD per 1M input tokens, for the cost report (ASSUMPTION)")
    ap.add_argument("--price-out", type=float, default=12.0,
                    help="USD per 1M output tokens (ASSUMPTION; Gemini bills "
                         "thinking tokens as output)")
    ap.add_argument("--out-tokens", type=int, default=300,
                    help="assumed output tokens/turn for the rough pre-run estimate")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N turns - use for a cheap smoke test")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight judge requests. 1 = sequential. Raise on a PAID "
                         "tier (e.g. 6); the SDK retries 429/5xx with backoff. On a "
                         "rate-limited free key, leave at 1.")
    ap.add_argument("--checkpoint", type=int, default=200,
                    help="atomically rewrite the judgments file every N completed "
                         "turns, so a long run is crash-safe and resumes for free")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirm prompt")
    args = ap.parse_args()

    family, judge_model = parse_judge(args.judge)

    problems = preflight(family, judge_model)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    jpath = ROOT / args.judgments
    df = pd.read_json(jpath, lines=True)
    scenarios = load_all(ROOT / args.scenarios)
    cache_dir = ROOT / args.cache

    existing_judges = sorted(df.judge_model.unique())
    print(f"{len(df):,} existing judgments from: {', '.join(existing_judges)}")
    print(f"new judge: {judge_model}  (family: {family})")

    if judge_model in existing_judges:
        print(f"!! {judge_model} already present. Only missing turns will be judged.")

    # The work list is one row per (run_key, turn) from the MOST COMPLETE OTHER
    # judge - NOT the alphabetically-first, and never the new judge itself. This
    # matters for resume: once the new judge's rows are in the file, anchoring
    # alphabetically (or on the new judge) could pick a PARTIAL base and silently
    # conclude there is nothing left to do, truncating the run. Anchor on the
    # existing judge with the most rows instead.
    others = df[df.judge_model != judge_model]
    base_judge = (others.judge_model.value_counts().idxmax() if len(others)
                  else df.judge_model.value_counts().idxmax())
    print(f"work list anchored on base judge: {base_judge} "
          f"({int((df.judge_model == base_judge).sum()):,} turns)")
    base = df[df.judge_model == base_judge].copy()
    done = set(zip(df[df.judge_model == judge_model].run_key,
                   df[df.judge_model == judge_model].turn_index))
    todo = [r for _, r in base.iterrows()
            if (r.run_key, r.turn_index) not in done]
    if args.limit:
        todo = todo[:args.limit]

    est_in = len(todo) * 900 / 1e6 * args.price_in
    est_out = len(todo) * args.out_tokens / 1e6 * args.price_out
    print(f"\n{len(todo):,} turns to judge with {judge_model}")
    print(f"rough estimate: ${est_in + est_out:,.2f} "
          f"(@ ${args.price_in}/${args.price_out} per 1M in/out, {args.out_tokens} out-tok/turn)")
    if family == "gemini":
        print("  NB: this is a FLOOR. Gemini 3.x pro is a thinking model and bills "
              "thinking tokens as output, so real output/turn runs above the "
              f"{args.out_tokens}-token assumption. The 50-turn smoke test returns "
              "the real per-turn cost to extrapolate from.")
    print(f"  NB: {len(todo):,} turns = {len(todo):,} API requests. On a rate-limited "
          "key this is the binding constraint, not dollars; the run is resumable so "
          "it can be done in daily batches at no extra cost.")
    if args.dry_run:
        print("\n[dry run] no judging performed.")
        return
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return

    judge = build_judge(family, judge_model)

    # Pre-load every conversation single-threaded, so the worker pool only READS
    # an immutable dict on the hot path (no lock needed there).
    convs: dict = {}
    for r in todo:
        rk = r.run_key
        if rk not in convs:
            p = cache_dir / f"{rk}.json"
            convs[rk] = json.loads(p.read_text())["messages"] if p.exists() else None

    def judge_one(r):
        """Judge one turn. Returns a tagged tuple; runs in a worker thread. The
        only shared state it touches is the judge's client (its usage counters are
        locked) and the read-only `convs`, so it is safe to run concurrently."""
        msgs = convs.get(r.run_key)
        if msgs is None:
            return ("missing", r.run_key)
        users = [m["content"] for m in msgs if m["role"] == "user"]
        assts = [m["content"] for m in msgs if m["role"] == "assistant"]
        i = int(r.turn_index) - 1
        if i >= len(assts) or i >= len(users):
            return ("fail", {"run_key": r.run_key, "turn": int(r.turn_index),
                             "err": "turn index out of range"})
        try:
            tj = judge.judge_turn(scenarios[r.scenario_id], users[i], assts[i],
                                  r.run_key, int(r.turn_index))
            row = judgment_to_row(tj, spec_from_row(r))
            # carry n_tokens (a property of the assistant turn, judge-independent)
            # so the appended rows keep the same schema as the first judge's rows.
            if "n_tokens" in base.columns:
                nt = r.get("n_tokens")
                row["n_tokens"] = None if pd.isna(nt) else int(nt)
            return ("ok", row)
        except Exception as e:                                  # noqa: BLE001
            return ("fail", {"run_key": r.run_key, "turn": int(r.turn_index),
                             "err": str(e)})

    rows, failures, missing_cache = [], [], set()

    def checkpoint():
        # Atomic whole-file rewrite (original rows + everything judged so far).
        # Cheap at a few-hundred-row cadence, and it makes a long paid run
        # crash-safe: an interruption leaves a valid file the next run resumes
        # from for free (already-judged turns are skipped). Same serialization as
        # the original single-write path, via os.replace so readers never see a
        # half-written file.
        tmp = jpath.with_suffix(".jsonl.tmp")
        pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_json(
            tmp, orient="records", lines=True)
        os.replace(tmp, jpath)

    def collect(res):
        if res[0] == "ok":
            rows.append(res[1])
        elif res[0] == "fail":
            failures.append(res[1])
        else:
            missing_cache.add(res[1])

    workers = max(1, args.concurrency)
    since_ckpt = 0
    print(f"judging with concurrency={workers}, checkpoint every {args.checkpoint}")
    if workers == 1:
        for r in tqdm(todo, desc="judging"):
            collect(judge_one(r))
            since_ckpt += 1
            if since_ckpt >= args.checkpoint:
                checkpoint(); since_ckpt = 0
    else:
        # Bounded concurrency. The live clients already retry 408/429/5xx with
        # exponential backoff at the SDK layer (Gemini HttpRetryOptions / OpenAI
        # max_retries) and Judge retries parse failures, so rate-limit blips
        # self-heal; keep the pool small enough to stay under the tier's RPM.
        # Results are collected in THIS thread, so rows/checkpoint stay single-
        # threaded. Whatever ultimately fails is logged and resumable.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(judge_one, r) for r in todo]
            for f in tqdm(as_completed(futs), total=len(futs), desc="judging"):
                collect(f.result())
                since_ckpt += 1
                if since_ckpt >= args.checkpoint:
                    checkpoint(); since_ckpt = 0

    if missing_cache:
        print(f"\n!! {len(missing_cache)} conversations missing from {cache_dir}. "
              "Those turns were skipped.")

    if not rows:
        print("no new judgments produced")
        _report_usage(judge, args)
        return

    checkpoint()  # final flush
    combined = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    print(f"\nappended {len(rows):,} judgments -> {jpath}")
    print(f"file now holds {len(combined):,} rows from "
          f"{combined.judge_model.nunique()} judges")

    _report_usage(judge, args)

    # ---- reliability --------------------------------------------------------
    if combined.judge_model.nunique() > 1:
        rel = reliability_table(combined, scenarios)
        relp = jpath.with_suffix(".reliability.csv")
        rel.to_csv(relp, index=False)
        print("\nJUDGE RELIABILITY - this table goes in the results section")
        print(rel.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        print(f"  -> {relp}")

        prim = rel[rel.primary]
        weak_primary = prim[prim.gate == "uncoded"].code.tolist()
        if weak_primary:
            print(f"\n  !! PRIMARY code(s) below .67: {', '.join(weak_primary)}")
            print("     These are near-factual and SHOULD agree. Treat as a rubric "
                  "problem, not noise - the primary result depends on them.")
        conf = rel[rel.gate == "confirmatory"].code.tolist()
        weak = rel[rel.gate == "uncoded"].code.tolist()
        print(f"\n  confirmatory (>=.80): {', '.join(conf) if conf else 'none'}")
        print(f"  uncoded      (<.67) : {', '.join(weak) if weak else 'none'}")

        ag = pairwise_agreement(combined)
        q = adjudication_queue(ag, combined)
        qp = jpath.with_suffix(".adjudicate.csv")
        q.to_csv(qp, index=False)
        if not ag.empty:
            print(f"\n  mean disagreement {ag.disagreement_score.mean():.1%}")
        print(f"  {len(q)} hardest turns -> {qp}")
        print("  hand-code those, not a random sample.")

    if failures:
        fp = jpath.with_suffix(".judge_failures.json")
        fp.write_text(json.dumps(failures, indent=2))
        print(f"\n{len(failures)} failures -> {fp}")


def _report_usage(judge, args):
    u = getattr(getattr(judge, "client", None), "usage", None)
    if not u:
        print("\n(no usage counters on this client)")
        return
    cost = u["input"] / 1e6 * args.price_in + u["output"] / 1e6 * args.price_out
    print(f"\nACTUAL USAGE for {judge.model}")
    print(f"  calls          {u['calls']:,}")
    print(f"  input tokens   {u['input']:,}")
    print(f"  output tokens  {u['output']:,}")
    print(f"  cost           ${cost:,.2f}  "
          f"(at ${args.price_in}/${args.price_out} per 1M - verify against your invoice)")


if __name__ == "__main__":
    main()
