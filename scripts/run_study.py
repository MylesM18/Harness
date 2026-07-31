#!/usr/bin/env python3
"""
Run the real study.

    export ANTHROPIC_API_KEY=...          # subject side
    export GEMINI_API_KEY=... OPENAI_API_KEY=...   # judge panel
    python scripts/run_study.py --models claude-sonnet-4-6 claude-opus-4-6 \
                                --turns 12 --replicates 4

The judges are a cross-family PANEL by default (--judge-models
gemini-3.1-pro gpt-5.6-terra), each dispatched to its SDK by family prefix. Every
metric is reported SEPARATELY per judge - the driver writes a combined
data/judgments.jsonl (with a judge_model column) plus one
data/judgments.<judge>.jsonl per judge, and prints a per-judge summary. Judges
are never pooled.

Resumable. Every conversation is cached by a hash of its RunSpec, so an
interrupted run picks up where it stopped and re-running costs nothing for work
already done.

Wire test (no key, no spend). Use >= 2 replicates so the per-judge summary's
noise-floor step (metrics.denoise) has same-arm pairs to work with; with a single
replicate the summary is skipped (best-effort, judgments still written):

    python scripts/run_study.py --offline --scenarios S04_tanking_strategy \
                                --turns 4 --replicates 2

Run the positive controls FIRST:

    python scripts/run_study.py --controls-only --replicates 2

If an explicit anti-sycophancy system prompt does not move the metrics and a
warmth prompt does not move them the other way, the instrument is not measuring
what it claims and the main study should not proceed.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:                                    # auto-load a repo-root .env into os.environ, so the
    from dotenv import load_dotenv      # LIVE panel keys can live in .env instead of being
    load_dotenv(ROOT / ".env")          # exported by hand. Already-exported vars win (override=False).
except ImportError:                     # offline / wire-test path needs no keys and no dotenv.
    pass

import pandas as pd
from tqdm import tqdm

from harness.schema import RunSpec
from harness.scenarios import load_all, REAL_SCENARIOS_DIR
from harness.runner import Runner, estimate_cost
from harness.judge import judgment_to_row
from harness import providers, metrics


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


def resolve_judge_models(args) -> list[str]:
    """The panel (--judge-models) is the default; the singular --judge-model
    alias overrides it when explicitly passed (default None)."""
    if getattr(args, "judge_model", None):
        return [args.judge_model]
    return list(args.judge_models)


def _safe_judge_id(judge_model: str) -> str:
    """Filename-safe judge id: any run of non-alphanumerics collapses to '_'."""
    return re.sub(r"[^0-9A-Za-z]+", "_", judge_model)


def run_conversations(specs, scenarios, panel, runner):
    """Run every spec through the runner, then judge each turn with EVERY panel
    member independently. Iterating panel.judges (rather than panel.judge_turn)
    means one judge failing on a turn does not drop the others' codings; each
    failure is logged against its own judge model id. Rows carry judge_model via
    judgment_to_row, so the combined frame is separable downstream."""
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
            for judge in panel.judges:
                try:
                    j = judge.judge_turn(sc, ut, at, spec.key, i)
                    rows.append(judgment_to_row(j, spec))
                except Exception as e:                      # noqa: BLE001
                    failures.append({"spec": spec.key, "stage": "judge",
                                     "turn": i, "judge": judge.model, "err": str(e)})
    return rows, failures


def write_judgments(rows, out_path) -> dict:
    """Write the combined frame (all judges, judge_model column) to out_path, and
    one out_path.stem.<safe-judge-id>.suffix per judge. Returns {judge_model: path}."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(out_path, orient="records", lines=True)

    per_judge = {}
    if "judge_model" in df.columns:
        for jm, g in df.groupby("judge_model"):
            p = out_path.with_name(
                f"{out_path.stem}.{_safe_judge_id(jm)}{out_path.suffix}")
            g.to_json(p, orient="records", lines=True)
            per_judge[jm] = p
    return per_judge


def per_judge_summary(frame: pd.DataFrame, scenarios, offline: bool = False) -> str:
    """A compact, per-judge friction table - friction half-life and terminal
    friction for each subject, computed SEPARATELY per judge (never pooled). Under
    --offline the numbers are a synthetic fixture, so the header says so."""
    batteries = metrics.compute_all_per_judge(frame, scenarios)
    lines = ["per-judge summary (WIRE TEST - fixture numbers, not a measurement):"
             if offline else "per-judge summary:"]
    for jm, battery in batteries.items():
        lines.append(f"\njudge = {jm}")
        fsc = battery["friction"]
        subjects = sorted(fsc.model.unique()) if not fsc.empty else []
        if not subjects:
            lines.append("  (no pressure-arm friction data)")
            continue
        lines.append(f"  {'subject':<24}{'friction_half_life':>20}"
                     f"{'terminal_friction':>20}")
        for m in subjects:
            fhl = metrics.friction_half_life(fsc, model=m)
            hl = fhl.get("empirical_half_life")
            tr = fhl.get("terminal_rate")
            hl_s = "n/a" if hl is None else str(hl)
            tr_s = "n/a" if tr is None else f"{tr:.3f}"
            lines.append(f"  {m:<24}{hl_s:>20}{tr_s:>20}")
    return "\n".join(lines)


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
    ap.add_argument("--judge-models", nargs="+",
                    default=["gemini-3.1-pro", "gpt-5.6-terra"],
                    help="cross-family judge panel; each id dispatched to its SDK by "
                         "family prefix (gemini*/gpt*|o1*|o3*|o4*/claude*). Metrics "
                         "are reported PER judge, never pooled.")
    ap.add_argument("--judge-model", default=None,
                    help="single-judge alias; overrides --judge-models when passed")
    ap.add_argument("--offline", action="store_true",
                    help="MockModelClient subject + keyless synthetic panel: exercises "
                         "the whole new code path with no key and no spend (a wire test)")
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

    judge_models = resolve_judge_models(args)
    panel = providers.panel_from_models(judge_models, subject_models=args.models,
                                        offline=args.offline)
    self_judges = [j.model for j in panel.judges if j.self_judging]
    if self_judges:
        print(f"\n!! judge model(s) also under test: {', '.join(self_judges)}. "
              f"self_judging flag propagates to the results table.\n")

    est = estimate_cost(len(scenarios), len(args.arms or ARMS), len(args.models),
                        args.replicates, args.turns)
    print(json.dumps(est, indent=2))
    print(f"note: estimate_cost counts the SUBJECT side only; judge-side cost scales "
          f"x {len(judge_models)} judges ({', '.join(judge_models)}).")
    if args.dry_run:
        return

    if args.offline:
        print(f"\n[offline] wire test: {len(specs)} conversations x "
              f"{len(judge_models)} synthetic judges, no key, no spend.")
        # A distinct cache dir: mock traces must never poison the live trace cache
        # (same spec key, fabricated content).
        runner = Runner(client=providers.MockModelClient(),
                        cache_dir=ROOT / "data/cache-offline")
    else:
        if input(f"\nrun {len(specs)} conversations x {len(judge_models)} judges? "
                 f"[y/N] ").strip().lower() != "y":
            return
        runner = Runner(cache_dir=ROOT / "data/cache")

    rows, failures = run_conversations(specs, scenarios, panel, runner)

    out = ROOT / args.out
    per_judge = write_judgments(rows, out)
    print(f"\nwrote {len(rows):,} judgments to {out}")
    for jm, p in per_judge.items():
        print(f"  judge {jm}: {p}")

    if failures:
        fp = out.with_suffix(".failures.json")
        fp.write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} failures logged to {fp}")
        # Failures are not silently dropped. A run with a non-trivial failure
        # rate is not a clean run, and which arm the failures came from matters:
        # if the con arm fails more often, that is a refusal signal and the
        # scenario contestability assumption is in trouble.

    # Per-judge summary. The judgments are already on disk; a metrics error here
    # must never lose them, so this is strictly best-effort.
    try:
        print("\n" + per_judge_summary(pd.DataFrame(rows), scenarios,
                                       offline=args.offline))
    except Exception as e:                                  # noqa: BLE001
        print(f"\n(per-judge summary skipped: {e})")


if __name__ == "__main__":
    main()
