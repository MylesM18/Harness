#!/usr/bin/env python3
"""
Run the real study.

    export ANTHROPIC_API_KEY=...
    python scripts/run_study.py --models claude-sonnet-4-6 claude-opus-4-6 \
                                --turns 12 --replicates 4

Resumable. Every conversation is cached by a hash of its RunSpec, so an
interrupted run picks up where it stopped and re-running costs nothing for work
already done.

Run the positive controls FIRST:

    python scripts/run_study.py --controls-only --replicates 2

If an explicit anti-sycophancy system prompt does not move the metrics and a
warmth prompt does not move them the other way, the instrument is not measuring
what it claims and the main study should not proceed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from harness.schema import RunSpec
from harness.scenarios import load_all, REAL_SCENARIOS_DIR
from harness.runner import Runner, estimate_cost
from harness.judge import Judge, judgment_to_row


ARMS = ["pro", "con", "neutral", "nosource_pro", "nosource_con"]


def build_specs(args, scenarios) -> list[RunSpec]:
    specs = []
    arms = args.arms or ARMS
    prompts = ["anti_syco", "warm"] if args.controls_only else [args.system_prompt]
    for sp in prompts:
        for sid in scenarios:
            for arm in arms:
                for model in args.models:
                    for rep in range(args.replicates):
                        specs.append(RunSpec(
                            scenario_id=sid,
                            arm=arm,
                            pressure="none" if arm == "neutral" else args.pressure,
                            model=model,
                            system_prompt=sp,
                            persona=args.persona,
                            n_turns=args.turns,
                            replicate=rep,
                            temperature=args.temperature,
                        ))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-4-6"])
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--pressure", default="gradual", choices=["gradual", "abrupt"])
    ap.add_argument("--system-prompt", default="neutral",
                    choices=["none", "neutral", "anti_syco", "warm"])
    ap.add_argument("--persona", default="none",
                    choices=["none", "novice", "expert", "high_status"])
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="run at deployment temperature; variance across replicates "
                         "is part of the phenomenon, not noise around it")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--controls-only", action="store_true",
                    help="run only the two positive controls, before spending on Study 1")
    ap.add_argument("--scenarios", nargs="+", metavar="ID", default=None,
                    help="restrict to these scenario ids (default: the whole real set). "
                         "Use for the staged screen, e.g. --scenarios S04_tanking_strategy")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=None,
                    help="restrict to these arms (default: all five)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="data/judgments.jsonl")
    args = ap.parse_args()

    scenarios = load_all(REAL_SCENARIOS_DIR)
    if args.scenarios:
        missing = [s for s in args.scenarios if s not in scenarios]
        if missing:
            sys.exit(f"unknown scenario id(s): {', '.join(missing)}\n"
                     f"available: {', '.join(sorted(scenarios))}")
        scenarios = {s: scenarios[s] for s in args.scenarios}
    specs = build_specs(args, scenarios)

    est = estimate_cost(len(scenarios), len(args.arms or ARMS), len(args.models),
                        args.replicates, args.turns)
    print(json.dumps(est, indent=2))
    if args.dry_run:
        return
    if input(f"\nrun {len(specs)} conversations? [y/N] ").strip().lower() != "y":
        return

    runner = Runner(cache_dir=ROOT / "data/cache")
    judge = Judge(model=args.judge_model,
                  self_judging=args.judge_model in args.models)
    if judge.self_judging:
        print("\n!! judge model is also under test. self_judging flag will "
              "propagate to the results table.\n")

    rows, failures = [], []
    for spec in tqdm(specs, desc="conversations"):
        sc = scenarios[spec.scenario_id]
        try:
            trace = runner.run(spec, sc)
        except Exception as e:                              # noqa: BLE001
            failures.append({"spec": spec.key, "stage": "run", "err": str(e)})
            continue

        user_turns = [m["content"] for m in trace.messages if m["role"] == "user"]
        for i, (ut, at) in enumerate(zip(user_turns, trace.assistant_turns), start=1):
            try:
                j = judge.judge_turn(sc, ut, at, spec.key, i)
                rows.append(judgment_to_row(j, spec))
            except Exception as e:                          # noqa: BLE001
                failures.append({"spec": spec.key, "stage": "judge",
                                 "turn": i, "err": str(e)})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records", lines=True)
    print(f"\nwrote {len(rows):,} judgments to {out}")

    if failures:
        fp = out.with_suffix(".failures.json")
        fp.write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} failures logged to {fp}")
        # Failures are not silently dropped. A run with a non-trivial failure
        # rate is not a clean run, and which arm the failures came from matters:
        # if the con arm fails more often, that is a refusal signal and the
        # scenario contestability assumption is in trouble.


if __name__ == "__main__":
    main()
