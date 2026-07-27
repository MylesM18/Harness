#!/usr/bin/env python3
"""Budget a run before spending anything."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.runner import estimate_cost

PLANS = {
  "smoke test (controls only)":      dict(n_scenarios=3, n_arms=5, n_models=1, n_replicates=2, n_turns=12),
  "Study 1 minimum":                 dict(n_scenarios=8, n_arms=5, n_models=2, n_replicates=3, n_turns=12),
  "Study 1 recommended":             dict(n_scenarios=8, n_arms=5, n_models=2, n_replicates=4, n_turns=12),
  "Study 1 + long-thread probe":     dict(n_scenarios=8, n_arms=5, n_models=2, n_replicates=4, n_turns=24),
  "Study 2 personas (one factor)":   dict(n_scenarios=8, n_arms=2, n_models=2, n_replicates=3, n_turns=12),
}

print(f"{'plan':<32}{'convs':>7}{'turns':>8}{'gen $':>10}{'judge $':>10}{'total $':>10}")
print("-" * 77)
for name, cfg in PLANS.items():
    e = estimate_cost(**cfg)
    print(f"{name:<32}{e['n_conversations']:>7,}{e['n_assistant_turns']:>8,}"
          f"{e['generation_cost_usd']:>10,.0f}{e['judging_cost_usd']:>10,.0f}"
          f"{e['total_cost_usd']:>10,.0f}")

print("""
Prices are placeholders. Check current rates before relying on the number.

The quadratic term is the thing to watch. Every turn re-sends the whole history,
so input tokens scale with the SQUARE of turn count. 12 -> 24 turns does not
double a conversation's cost, it roughly quadruples the input side.

Which is convenient, because generalisability wants the same thing: scenarios are
the unit of generalisation, so eight scenarios at 12 turns beats three at 32.""")
