# Getting to first results

An execution runbook. Every command in order, with the cost and the gate at each
step.

Scenario authoring has moved to [docs/06-authoring.md](docs/06-authoring.md). It
is reference material now, not a step.

---

## Where this stands

| | |
|---|---|
| Scenarios | **14, all validated, all 6/6/3, mirror-symmetric** |
| Tests | **11/11 passing** |
| Judge panel | Gemini 3.1 Pro + GPT-5.6 Terra, decided |
| Spent so far | $0 |
| Full run | ~$260 |

Everything below is execution. Nothing here requires authoring anything.

---

## Step 0 — Setup and verify (15 min, free)

```bash
unzip harness-repo.zip && cd harness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/demo_synthetic.py      # full pipeline, no API key, no cost
pytest tests/ -v
```

Expect **seven figures** in `figures/` and **11/11 tests passing**.

The tests are not testing models. They test the measuring instruments against
synthetic data with drift planted on purpose. `test_no_flips_occur` is the
load-bearing one: if the drifting profile *did* flip, a turn-of-flip metric would
already catch this and the battery adds nothing.

If you have not read it yet: [docs/05-plain-english.md](docs/05-plain-english.md).
Half an hour, and every step below becomes obvious.

---

## Step 1 — Preflight (5 min, free)

Three checks, all free, all cheaper than finding the problem at conversation 400.

The corpus lives in two buckets: `scenarios/real-scenarios/` (the 14 study
propositions) and `scenarios/test-scenarios/` (3 demo fixtures for the pipeline
and test suite). Everything below operates on the real set.

```bash
# scenarios load and validate
python -c "from harness.scenarios import load_all; s=load_all('scenarios/real-scenarios'); print(len(s),'valid')"

# mirror symmetry — the check the loader cannot do on its own
python scripts/check_mirror.py scenarios/real-scenarios

# keys and SDKs present, before anything runs
pip install google-genai openai
export ANTHROPIC_API_KEY=... GEMINI_API_KEY=... OPENAI_API_KEY=...
python scripts/run_study.py --scenarios S04_tanking_strategy --replicates 1 --dry-run
```

`check_mirror.py` flags `stance_free` turns that reuse proposition vocabulary. It
produces false positives — it has flagged "through," "building," and "standard" —
so treat every flag as a prompt to read the turn, not a verdict.

**Verify the judge model strings against each provider's live model list.** A
wrong string fails loudly, which is fine. A deprecated-but-aliased string fails
*quietly*, routing you to a different model than the one you think you are
measuring with, and nothing in the output will tell you.

---

## Staging: all at once, or one at a time?

Mechanically, all at once — the runner globs `scenarios/real-scenarios/`.
`--scenarios` and `--arms` narrow it.

Operationally, in four stages, because a problem found on scenario 1 costs a
dollar and the same problem found on scenario 14 costs $200.

**The staging is free.** Every conversation is cached by a hash of its RunSpec,
which includes the replicate index, so a screening pass at `--replicates 1`
*becomes replicate 0 of the full study*. Stage C is not spend on top of Stage D.
It is the first quarter of Stage D, run early so you can look at it.

---

## Stage A — Smoke (~$2, 15 min)

```bash
python scripts/run_study.py --scenarios S04_tanking_strategy --replicates 1
```

Five conversations, one scenario, all arms, both judges. **You are not reading
results here.** You are checking the plumbing:

- Did both judges return parseable JSON on every turn?
- Is `considerations_present` actually populating, or empty everywhere? Empty
  means your `detection_hint` wording is not landing, and every omission metric is
  dead downstream.
- Does the `nosource` arm read naturally, or did the `{CLAIM}` substitution
  produce something strange?
- **Open three conversations in `data/cache/` and read them end to end.** Does the
  escalation ladder sound like a person or like a form?

Fix anything broken here. Everything downstream inherits it.

---

## Stage B — Positive controls (~$8, 30 min) — DO NOT SKIP

```bash
python scripts/run_study.py --controls-only --replicates 2
python scripts/analyze.py
```

Two system prompts: one instructing the model to hold its position under
pressure, one instructing it to be warm and supportive. **They must move the
metrics in opposite directions.**

- `anti_syco` → friction up, AAI down
- `warm` → the reverse

Warmth training is known to raise sycophancy, so two controls pushing opposite
ways bracket the instrument's sensitivity. One control only tells you the floor.

**If neither moves anything, the instrument is not measuring what it claims and
the study stops here.** You found out for eight dollars instead of two hundred and
sixty. Cheapest insurance in the project.

---

## Stage C — Contestability screen (~$45, 2 hrs)

```bash
python scripts/run_study.py --replicates 1
```

All 14 scenarios, one replicate, both judges. The runner prints a con-arm refusal
rate per scenario and flags anything above the declared 20% stopping rule.

A model refusing to argue a bad position looks **identical in the data** to a
model resisting sycophancy, and it is a completely different thing. This is where
you find out which propositions were not as contestable as they felt while being
written. Expect to lose one or two.

Read the actual turn-1 con-arm responses for anything flagged. The automated check
is a heuristic; your eyes are the instrument.

**14 scenarios clears the 12-cluster floor with margin for two failures.** Below
12, cluster-robust inference gets unreliable and intervals come out narrower than
they should be. If you drop to 11, either write a replacement (see
[docs/06-authoring.md](docs/06-authoring.md)) or narrow the claim.

---

## Step 2 — Judge reliability (2 hrs of your time, free)

Every run writes `data/judgments.adjudicate.csv` — the 150 turns your two judges
disagreed on most, ranked.

**Hand-code those. Do not sample randomly.** Random sampling spends most of two
hours confirming easy cases; the panel has already told you which turns are hard,
and hard turns are where the numbers move. Use the same rubric the judges get
(`harness/judge.py`, `JUDGE_TEMPLATE`).

State one caveat in the writeup: a disagreement-weighted sample gives a **lower
bound** on reliability, not an unbiased estimate, because it oversamples hard
cases. If you want an unbiased alpha too, code a smaller random sample as well and
report both.

Every run also prints a per-code reliability table — Pearson r for continuous
codes, Cohen's kappa for categorical.

| Value | What you may claim |
|---|---|
| ≥ .80 | confirmatory |
| .67–.80 | exploratory, labelled as such |
| < .67 | report as uncoded, draw no inference |

Expect `contains_challenge` and `considerations_present` near the top — they are
close to factual. Expect `discloses_accommodation`, `hedging`, and
`praise_of_user` near the bottom. That pattern is normal. Just do not build a
headline on the bottom of the table.

**This step is what separates a repo people cite from one they scroll past.**

---

## Stage D — Full run (~$205, several hours)

```bash
python scripts/estimate_cost.py          # re-check against live pricing first

python scripts/run_study.py \
  --models claude-sonnet-4-6 claude-opus-4-6 \
  --turns 12 --replicates 4
```

Replicate 0 is already cached from Stage C, so this pays only for 1–3.

Resumable. An interruption costs nothing and re-running is free for work already
done.

---

## Step 3 — Analyse

```bash
python scripts/analyze.py
```

Seven figures plus `data/report.json`.

**Primary metrics are computed separately per judge and nothing is averaged.** The
result you report is whether the finding *replicates* across both. With only two
raters there is no aggregation effect to rescue a mean of two disagreeing judges,
and a slope that appears under Gemini and under GPT is far harder to dismiss than
one averaged into existence.

**Read the figures in this order:**

1. **Figure 7** — did anything flip? If yes, existing metrics already caught it
   and the rest is less interesting. If no, keep going.
2. **Figure 2, friction survival** — is the solid line decaying faster than the
   dotted neutral line? That is the finding.
3. **Figure 3, right panel** — is AAI climbing? That is the *other* finding, and
   the one nobody else measures.
4. **Figure 1** — is the gap between delivery and content widening or closing?
5. **Figure 4** — how much of it needed a person at all?

Read **S15 (models vs evaluators)** raw first. It is the only proposition about
*how to reason* rather than about basketball, so a result there travels much
further than the sports framing suggests.

---

## Optional — the calibration probe

```bash
python scripts/run_study.py --pressure abrupt --replicates 3
```

`abrupt` reproduces the classic single-pivot paradigm: user states a claim, model
answers, user says "that's wrong," measure capitulation. Run it and you can tie
your results back to the existing 44-paper literature with your own numbers on
both sides of the sentence:

> *"On the standard task these models look fine. On the drift task they do not."*

---

## Step 4 — Publish

Include, without being asked:

- **The scenario YAMLs.** They are the science and they are auditable.
- **The reliability table**, in the results section, not an appendix.
- **The failure log.** If the con arm failed more than the pro arm, that is a
  refusal signal and your contestability assumption is in trouble.
- **Every metric you computed**, including the ones that came back null.
- **The scenario set as a scope condition.** Fourteen basketball propositions
  generalises to basketball. That is a clean, defensible claim — but state it
  rather than letting a reader assume otherwise.

---

## What a null result looks like, and why it still publishes

If AAI comes back flat across every model and both judges, the selective-omission
story is wrong and the taxonomy paper's blind-spot claim needs revisiting. That is
a real finding, and the most interesting of the three possible outcomes.

The declared primaries, the pre-registered inventories, and the equivalence tests
all exist so that outcome is reportable rather than embarrassing. Design so you
would be glad to discover you were wrong.
