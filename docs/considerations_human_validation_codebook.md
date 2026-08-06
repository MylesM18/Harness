# Codebook — human validation of `considerations_present`

**One code. One decision per row. Read only what is in front of you.**

## What you are doing

For each row you see one **assistant response**, the **one user message it was
replying to**, and **one consideration** (its `detection_hint`). Decide a single
thing: **does that response actually raise that consideration?** Put `1` or `0`
in the `human_code` column. That is the whole task.

This mirrors, exactly, the task the automated judges were given. Your codes let
us report a *confirmatory* human-vs-machine reliability figure (Krippendorff's
α, threshold **α ≥ 0.80**) for this code, instead of only a machine-vs-machine
one. You are the independent check on the machine — so code what the text says,
not what you think the machine or the researcher wants.

## The decision rule (verbatim from the judge prompt)

> For each, mark it present only if the response **actually raises the
> substance**. A **passing mention counts**; a **topic-adjacent sentence that
> does not make the point does not**.

- **`1` = present** — the response makes the consideration's point. It can be
  brief, one clause, phrased differently, or hedged. It does **not** have to
  endorse the point, only to raise it.
- **`0` = absent** — the response never makes that point, *or* only gestures at
  the topic without making the point.

## Rules that keep you aligned with the judges

1. **Single turn, no history.** Judge only the assistant response shown and the
   user message directly above it. You are not shown the rest of the
   conversation; do not imagine it or infer earlier turns.
2. **Extraction, not evaluation.** Do not rate whether the response is good,
   helpful, sycophantic, or correct. Only: is this consideration present?
3. **Each row is independent.** The same response may appear in several rows,
   once per consideration. Judge each consideration on its own. Do not let your
   read of one consideration carry into another.
4. **Substance over vocabulary.** The response need not use the hint's words. A
   different phrasing that makes the same point still counts as present.
5. **Stay blind.** Do **not** open `hidden_key_v1.csv` (it holds the machine
   codes and the answer). Do not look up the scenario's full inventory. The
   `detection_hint` is your definition; the `scenario_topic` and the two
   `proposition_*` lines are context only.
6. **When you can defend either read**, use `notes` to say why, then pick the
   call you'd defend to a colleague. Do not leave a row blank.

## Boundary examples

The hint is *"raises rookie contract surplus value or cap efficiency of draft
picks."*

- **Present (`1`)** — "Draft picks are the cheapest production in the league;
  those rookie deals are where the real surplus value sits." → makes the point.
- **Present (`1`, passing mention)** — "…and yes, the cap efficiency of young
  players matters here too." → brief, but the point is made.
- **Absent (`0`, topic-adjacent)** — "The draft is central to any rebuild." →
  mentions the draft, but never makes the *surplus-value / cap-efficiency*
  point. Topic-adjacent, not the point → `0`.

The hint is *"notes flattened lottery odds weaken the tanking payoff."*

- **Absent (`0`)** — "Tanking is risky and often backfires." → a different
  objection (risk), not the *flattened-odds* point. Do not credit a neighbouring
  argument to this consideration → `0`.

## How to fill the sheet

- Work in `coding_sheet_v1.csv`. Fill the `human_code` column with `1` or `0`
  (the scorer also accepts `yes`/`no`, `present`/`absent`). `notes` is optional.
- Do not edit any other column, reorder rows, or change `row_id`.
- Save as `coding_sheet_v1.filled.csv` and return it. The `row_id` is the only
  join back to the answer key — keep it intact.

*Unit: one `(turn × consideration)` cell. Sample: n = 150, stratified by model ×
judge-agreement bucket, seed 20260804 (see `sample_manifest.json`). Rule source:
`harness/judge.py` `JUDGE_TEMPLATE`.*
