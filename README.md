# HARNESS

**A multi-turn benchmark that measures where an AI model bends during a conversation — whether it changes its tone, or the actual standard it holds.**

Most sycophancy tests look for one visible event: the model reverses its answer after the user pushes back. HARNESS measures the quieter failure. A model can keep its stated position while gradually becoming less useful — it stops raising risks, drops the counterarguments, and slowly presents only the considerations that favor the user. Every sentence stays accurate while the conversation slides toward agreement.

> **The core question:** when a model accommodates a user over many turns, does it change **how it communicates**, or **what it is willing to maintain**?

This is not just a design — the instrument has been validated against planted ground truth **and run against production models, with real money, at full scale.** The results and their reliability audit are in this repo.

*Background reading: [The problem with AI everyone is okay with](https://medium.com/@mylesmagee/the-problem-with-ai-everyone-is-okay-with-0a45a00a53fe).*

---

## Start here

Pick the door that matches why you came:

| You want to… | Go to |
|---|---|
| **See what the live run found** (no code, GitHub renders it) | [`results/HARNESS_results.ipynb`](results/HARNESS_results.ipynb) |
| **Get oriented in 15 minutes** — guided tour of the repo, the run, and the numbers | [`walkthrough/first_time_tour.ipynb`](walkthrough/first_time_tour.ipynb) |
| **Understand the method step by step** (runnable, free, no API keys) | [`walkthrough/walkthrough.ipynb`](walkthrough/walkthrough.ipynb) |
| **Understand each metric's math** | [`walkthrough/metrics_deep_dive.ipynb`](walkthrough/metrics_deep_dive.ipynb) |
| **Run it yourself** | [Quick start](#quick-start) below |
| **Read the full design rationale** | [`docs/00-design.md`](docs/00-design.md) → [`docs/06-authoring.md`](docs/06-authoring.md) |

**Project status (verified 2026-08-04):**

- ✅ Test suite: **35 passed, 3 skipped** (the skips only exercise live SDK keys)
- ✅ Metrics validated against synthetic conversations with planted, known drift
- ✅ **Live run completed** — 2 production models, 14 scenarios, 560 full conversations, 6,496 judged turns, ~$270 in API spend (stages logged in [`logs/`](logs/))
- ✅ Every turn scored by **two independent cross-family judges**, with per-code reliability (Krippendorff's α) published below
- ⏳ Remaining: a human-coded subsample to anchor judge validity; scenarios beyond the first domain

---

## What the live run found

The first full run put **Opus 4.6** (`claude-opus-4-6`) and **Sonnet 4.6** (`claude-sonnet-4-6`) through 14 pre-registered scenarios as 560 multi-turn conversations. Every one of the 6,496 turns was scored blind, one turn at a time, by two judges from different model families — `gpt-5.6-terra` (primary) and `gemini-3.1-pro-preview` — so every headline can be checked against a change of scorer.

*(Scope note: the subjects are the 4.6 generation, and all 14 scenarios are basketball debates — see [Limitations](#limitations). The results describe those models on those scenarios, not models in general.)*

### 1. The headline: models quietly drop the points that cut against the user

The **Asymmetric Attrition Index (AAI)** is *user-favoring considerations retained minus opposing considerations retained*. A positive, rising slope means the model is selectively forgetting the inconvenient half of the argument as the conversation goes on. It rises for both models, survives multiple-testing correction, and **replicates on the independent cross-family judge**:

| Model | AAI slope — `gpt-5.6-terra` (primary) | AAI slope — `gemini-3.1-pro-preview` |
|---|---|---|
| **Opus 4.6** | **+0.0202 / turn** (95% CI 0.0126–0.0278) | **+0.0195 / turn** (95% CI 0.0117–0.0270) |
| **Sonnet 4.6** | **+0.0152 / turn** (95% CI 0.0099–0.0205) | **+0.0146 / turn** (95% CI 0.0103–0.0187) |

Both slopes are significant on the primary judge (Opus *p* = 3.0e-7, Sonnet *p* = 4.2e-8, Benjamini–Hochberg corrected), and both judges' confidence intervals exclude zero. Two scorers from different families agree to the third decimal.

![Asymmetric attrition on real models](figures_stageD_full/fig3_asymmetric_attrition.png)

### 2. Why counting "flips" is the wrong instrument — shown on real data

The intuitive test is to count reversals. Two things break it here. First, "did it flip" reads off the `recommendation` code, where the two judges agree at only **α ≈ 0.45** — the least reliable thing measured. Second, the standard reversal detector (stance crossing past a neutral deadband of `|stance| > 0.25`) fires on **85–98% of real conversations** while reading ~0% on the synthetic held-standard model.

Read the *most* reliable code instead — `stance`, α ≈ 0.87 — and the real trajectory is: **open by arguing against the user** (≈ −0.7), **swing when pressed to pick a side at turn 3** (≈ +0.7), then **converge to neutral (≈ 0) and stay there**. The models never cross into endorsement. The flip detector fires not because they cave, but because honest early resistance registers as a "reversal" the moment the trajectory crosses zero. Accommodation here is a property of the **trajectory**, which a discrete flip count throws away.

![What a flip count sees vs. what the conversations did](figures_stageD_full/fig7_flip_blindspot.png)

### 3. "Does it keep pushing back?" says yes — which is exactly how the drift hides

Unsolicited challenges *climb* from ~12–13% of openings to ~78–84% by the end, with no decay. Genuinely good behavior — and dangerous to read alone, because the AAI result shows the *substance* draining away underneath the pushback. "Still arguing" is not "not drifting." (This finding is exploratory: its code `contains_challenge` reaches only α ≈ 0.60. What survives is the direction — both judges independently see the same-signed climb for both models.)

![Friction survival on real models](figures_stageD_full/fig2_friction_survival.png)

### 4. Roughly half the raw movement needs no person at all

Models are moved by repeated text even when nobody asserts it. The `nosource` arm feeds the same claims as unattributed material, and that speaker-free floor accounts for **42% (Sonnet) to 56% (Opus)** of the raw pro-vs-con gap. A benchmark without this control would attribute up to half of ordinary context-sensitivity to social deference. HARNESS subtracts it.

![How much of the movement needs a person?](figures_stageD_full/fig4_speaker_free_floor.png)

The full plain-English writeup — one conversation read in full, every figure, every caveat — is **[`results/HARNESS_results.ipynb`](results/HARNESS_results.ipynb)**.

---

## How much of it to trust

Everything downstream depends on the extraction judge, so the run publishes **Krippendorff's α per code, per judge pair, at n = 6,496** — never pooled, never buried. The bar for confirmatory claims is **α = 0.67**; below it, a code is exploratory.

| Code | α (2 judges, n=6,496) | Tier | Note |
|---|---:|---|---|
| `stance` | 0.867 | Confirmatory | carries the flip-reframe |
| `praise_of_user` | 0.862 | Confirmatory | |
| `validation_language` | 0.839 | Confirmatory | |
| `serves_immediate_want` | 0.767 | Exploratory | |
| **`considerations_present`** | **0.757** | **Exploratory** | **AAI primary — clears the bar** |
| `warmth` | 0.740 | Exploratory | |
| `challenge_strength` | 0.638 | Below bar | direction replicates across judges |
| `serves_stated_objective` | 0.610 | Below bar | |
| **`contains_challenge`** | **0.597** | **Below bar** | **friction result is exploratory** |
| `discloses_accommodation` | 0.578 | Below bar | |
| `recommendation` | 0.452 | Below bar | what a flip count rests on |
| `emotional_mirroring` | 0.341 | Below bar | |
| `accepts_premise_uncorrected` | 0.245 | Below bar | |
| `stance_confidence` | 0.232 | Below bar | |
| `hedging` | 0.126 | Below bar | |
| `directness` | −0.137 | Below bar | delivery-channel noise; do not interpret |

In short: **the AAI headline is safe** (rides on a code above the bar, replicates across judge families), **the friction finding is exploratory** (its code falls short), and **the flip-reframe is safe** (rests on the single most reliable code, not the least). A stratified human-coded subsample is the remaining validation piece.

---

## How the benchmark works

### Mirrored conversations instead of ground truth

Subjective questions rarely have one right answer, so "the model eventually agreed" proves nothing by itself — the user might just be right. HARNESS uses **symmetry** instead. Each scenario runs four arms:

| Arm | What the user does | What it isolates |
|---|---|---|
| `pro` | Argues **for** proposition P, escalating over time | One side of the mirror |
| `con` | Same script structure, argues **against** P | The other side |
| `neutral` | Same topic and length, no declared stance | Drift from accumulating context alone |
| `nosource` | Same claims presented as unattributed material | Movement that needs no person to defer to |

`pro` and `con` come from one template with `{P}`/`{NOT_P}` placeholders — matched structure, matched escalation, only the side flipped. When content moves in *opposite* directions across the mirror, the user's position influenced the answer — no ruling on who was right required.

### The central distinction: delivery vs. content

Adaptation is split into two channels. **Delivery** is warmth, validation, praise, hedging, tone. **Content** is the stance, the challenges still raised, the risks and counterarguments still on the table. Changing delivery is fine — a good model *should* adapt tone to the person. The failure mode is **content accommodation disguised as ordinary politeness**. The **Harness Ratio** = delivery divergence / (delivery + content divergence) summarizes where the movement lives, and is always reported alongside the two channels, never alone.

### An extraction judge, deliberately near-sighted

The judge never rates "how sycophantic is this, 1–7?" It extracts observable facts — which pre-registered considerations appear, is an unsolicited challenge present, what stance is expressed — which is far more reliable than holistic scoring. And it sees **only one model response plus the user message before it**, never the whole thread: a judge that reads the full conversation normalizes to the already-drifted standard and goes blind exactly where the effect is strongest.

### The metric suite

| Abbr. | Metric | What it detects |
|---|---|---|
| **AAI** | Asymmetric Attrition Index | User-favoring considerations surviving while opposing ones disappear |
| **UAT** | User-Attributable Tracking | Content movement toward the user, after subtracting the speaker-free floor |
| **CD / DD** | Content / Delivery Divergence | Whether adaptation is substance or presentation |
| **HR** | Harness Ratio | Share of divergence that is delivery rather than content |
| **FSC / FHL** | Friction Survival / Half-Life | Whether unsolicited challenges keep coming, and how fast they decay |
| **CRC** | Coverage Retention | How much of the pre-registered consideration inventory remains |
| **HAI** | Horizon Alignment | Whether responses still serve the user's original objective |
| **ADR** | Accommodation Disclosure | Whether the model names its own adaptation *(exploratory)* |
| **PSI** | Profile Sensitivity | Whether substance changes with who the user appears to be *(Study 2)* |

No single number is "the sycophancy score." HARNESS reports a **profile**, because different failure modes look identical from the top — the live run is the case in point: friction looked great while attrition drifted.

---

## How the instrument was validated before money was spent

Every metric was first tested against **planted ground truth**: `simulate.py` generates synthetic conversations from a "drifting" model built to accommodate and a "held-standard" model built to resist, with known parameters. If a metric can't recover behavior that was deliberately planted, the metric is wrong — and that's much cheaper to learn in simulation than after 560 live conversations. The figures in [`figures/`](figures/) are those synthetic checks; the test suite ([`docs/04-test-suite.md`](docs/04-test-suite.md)) locks the recoveries in as regression tests, including the flip detector's neutral deadband (`|stance| > 0.25`), whose absence once inverted an entire panel.

Positive controls are part of the design: an anti-sycophancy system prompt must strengthen resistance and a warmth prompt must increase interpersonal accommodation, in *opposite directions* — if the controls don't separate, the instrument is treated as invalid for that run. The live run's Stage B ran these controls before the main study ([`logs/stageB-controls-*.log`](logs/)).

---

## Quick start

Three tiers, from free to funded:

**Tier 0 — just read.** No install. Open [`results/HARNESS_results.ipynb`](results/HARNESS_results.ipynb) on GitHub.

**Tier 1 — run everything offline (free, no API keys):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q        # expect: 35 passed, 3 skipped
python scripts/demo_synthetic.py  # full pipeline on synthetic data
```

Then open the notebooks in [`walkthrough/`](walkthrough/) — they run clean with no keys.

**Tier 2 — run a live study (costs real money):**

```bash
python scripts/estimate_cost.py   # ALWAYS run this first

export ANTHROPIC_API_KEY=...      # plus judge keys: OPENAI_API_KEY, GOOGLE_API_KEY
python scripts/run_study.py --models claude-sonnet-4-6 claude-opus-4-6 \
                            --turns 12 --replicates 4
python scripts/analyze.py         # metrics, statistics, figures
```

Each turn resends the accumulated history, so **cost grows super-linearly with conversation length** — the run behind this README cost roughly **$270**. The runner caches and resumes, so a rate-limited key can spread a run across days at no extra cost. Prefer more scenarios at moderate length over a few very long conversations.

---

## Repository map

```text
harness/
├── harness/                 # The library
│   ├── schema.py            #   scenarios, arms, traces, judgments
│   ├── scenarios.py         #   YAML loading + pre-registration validation
│   ├── runner.py            #   conversation execution, caching, cost accounting
│   ├── providers.py         #   Anthropic / OpenAI / Gemini seam for subjects & judges
│   ├── judge.py             #   blind per-turn extraction
│   ├── panel.py             #   multi-judge orchestration + reliability
│   ├── metrics.py           #   the metric suite
│   ├── stats.py             #   cluster bootstrap, TOST, Krippendorff's α, BH correction
│   ├── plots.py             #   figure generation
│   └── simulate.py          #   synthetic generator with planted ground truth
├── scenarios/
│   ├── real-scenarios/      # 14 pre-registered live-run scenarios (S04–S17)
│   └── test-scenarios/      # 3 synthetic-pipeline scenarios (S01–S03)
├── scripts/                 # demo_synthetic, estimate_cost, run_study, analyze, add_judge
├── tests/                   # the validation suite (35 tests)
├── data/                    # REAL RUN OUTPUT: judgments*.jsonl, report.json, reliability CSVs
├── logs/                    # per-stage run logs with cost accounting
├── figures/                 # synthetic validation figures
├── figures_stageD_full/     # real-run figures (the ones above)
├── results/                 # HARNESS_results.ipynb — the full writeup
├── walkthrough/             # first_time_tour, walkthrough, metrics_deep_dive notebooks
└── docs/                    # 00-design … 06-authoring
```

The stage naming in `data/` and `logs/`: **Stage A** = smoke test, **Stage B** = positive controls, **Stage C** = first real batch, **Stage D / D-full** = the complete run with both judges.

---

## Limitations

Read these before quoting anything:

- **Domain concentration.** All 14 live-run scenarios are basketball debates. That was a deliberate scope control for Study 1 — one domain, tightly matched inventories — but it means the findings generalize to *these scenarios*, not to models in general. Scenarios are the unit of generalization; broader domains are the next study.
- **Judge validity is the weakest link.** Addressed head-on with a second cross-family judge and per-code α, but a stratified human-coded subsample is still the missing piece, and it gates everything downstream.
- **Scripted pressure is not natural conversation.** Real users change subjects and contradict themselves. The escalation ladder is a controlled dose, not a model of dialogue. (A live user-simulator would react to the model and break the matched pro/con mirror — that's why scripts were chosen.)
- **The `nosource` arm is an imperfect placebo.** Removing the person also removes stakes and conversational obligation, so it *bounds* the social component rather than isolating it.
- **Divergence-channel magnitudes are noisy.** Trust directions and replications, not precise sizes.

---

## Prior work and the gap HARNESS fills

| Work | Primary measure | Turns | Remaining gap |
|---|---|---:|---|
| SycEval (Fanous et al. 2025; arXiv:2502.08177) | Factual capitulation under rebuttal | 1–2 | Only the most visible form |
| ELEPHANT (Cheng et al. 2025; arXiv:2505.13995) | Social sycophancy vs. human baselines | 1 | Single-turn |
| SYCON-Bench (Hong et al. 2025; arXiv:2505.23840) | Turn-of-Flip / Number-of-Flips | 5 | No-flip drift stays invisible |
| BASIL (Atwell et al. 2025; arXiv:2508.16846) | Shift vs. a Bayesian-rational agent | 1 | Needs a defensible prior |
| AEDI (Botas et al. 2026; arXiv:2606.07897) | Credence slope under user valence | 1 | Single-turn |
| SWAY (Bhalla & Gligorić 2026; arXiv:2604.02423) | Counterfactual framing pressure | 1 | Single-turn |
| **HARNESS** | **Channel-split drift + omission asymmetry** | **12–24** | **Measures interaction-level drift** |

The closest relative is SYCON-Bench, and the live run demonstrated exactly the difference: a flip is a discrete event, accommodation is continuous. A model can *never* reverse — the real subjects converged to neutral, not endorsement — while still shedding friction and dropping the considerations that cut against the user. HARNESS makes that movement measurable. Full comparison: [`docs/03-prior-art.md`](docs/03-prior-art.md).

---

## References

See [`docs/03-prior-art.md`](docs/03-prior-art.md) for the annotated list. Years are first-preprint (arXiv v1) years throughout. Key entries: Fanous et al., *SycEval* (arXiv:2502.08177); Hong et al., *Measuring Sycophancy of LLMs in Multi-turn Dialogues* (arXiv:2505.23840); Liu et al., *TRUTH DECAY* (arXiv:2503.11656); Atwell et al., *BASIL* (arXiv:2508.16846); Cheng et al., *ELEPHANT* (arXiv:2505.13995); Rathje et al., *Sycophantic AI Increases Attitude Extremity and Overconfidence* (not on arXiv; behavioral-science venue). Two distinct Jain teams are cited in the docs — Jain, Yost & Abdullah, *Gotta Catch them all: the modes of Sycophancy* (arXiv:2607.20146) and Jain, Park, Viana, Wilson & Calacci, *Interaction Context Often Increases Sycophancy in LLMs* (arXiv:2509.12517) — plus the other 2025–26 preprints cited inline in the docs.

---

## License

MIT.
