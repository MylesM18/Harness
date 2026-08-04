# Prior art and positioning

Where this sits relative to the existing sycophancy literature, and what is
actually new.

## The landscape

| Work | Operationalisation | Turns | Ground truth | Taxonomy cells |
|---|---|---|---|---|
| Perez et al. 2022 (arXiv:2212.09251) | endorsement of user-stated beliefs | 1 | required | Pos-Verif/Explicit |
| Sharma et al. 2023 (arXiv:2310.13548) | answer / feedback / mimicry sycophancy | 1–2 | required | Pos-Verif/Expl, Person-Traits/Expl |
| FlipFlop (Laban et al. 2023; arXiv:2311.08596) | performance drop under "are you sure?" | 2 | required | Pos-Verif/Explicit |
| SycEval (Fanous et al. 2025; arXiv:2502.08177) | progressive vs regressive capitulation | 1–2 | required | Pos-Verif/Explicit |
| TRUTH DECAY (Liu et al. 2025; arXiv:2503.11656) | multi-turn capitulation, MCQ format | ~5 | required | Pos-Verif/Explicit |
| SYCON-Bench (Hong et al. 2025; arXiv:2505.23840) | Turn-of-Flip, Number-of-Flip | 5 | expected stance | Pos-Subj/Expl + implicit |
| BASIL (Atwell et al. 2025; arXiv:2508.16846) | shift vs a Bayesian-rational agent | 1 | rational prior | Pos-Subj |
| ELEPHANT (Cheng et al. 2025; arXiv:2505.13995) | face-preservation: validation, indirectness, framing, moral endorsement | 1 | crowdsourced human baseline | Pos-Subj, Person-Emotions |
| AEDI (Botas, de Font-Reaulx & Hewitt 2026; arXiv:2606.07897) | logit-credence slope on user valence | 1 | none | Pos-Subj |
| SWAY (Bhalla & Gligorić 2026; arXiv:2604.02423) | counterfactual framing pressure, unsupervised | 1 | none | Pos-Subj/Implicit |
| Vennemeyer et al. 2025 (arXiv:2509.21305) | causal separation of agreement vs praise in activations | 1 | n/a | mechanistic |
| Jain, Yost & Abdullah 2026 (arXiv:2607.20146) | PA / SI / DCA representational modes | 1 | n/a | mechanistic |
| **HARNESS** | **channel-split drift + omission asymmetry over mirrored arms** | **12–24** | **none needed** | **Pos-Subj/Impl, Person-Traits/Impl** |

Years are first-preprint (arXiv v1) years throughout. Note there are two distinct
Jain teams in this doc: Jain, Yost & Abdullah (the modes paper above) and Jain,
Park, Viana, Wilson & Calacci (the paired-situation design under "Controls
borrowed" below).

## What each neighbour does that this does not

**ELEPHANT** is the strongest existing instrument for social sycophancy and it
covers cells this does not touch — moral endorsement, indirectness against
crowdsourced human baselines. It is single-turn. The two are complementary: run
ELEPHANT for breadth across social registers, run this for what happens to those
behaviours over twenty turns.

**BASIL** solves the no-ground-truth problem differently and more elegantly in one
respect: it defines sycophancy as movement beyond what a Bayesian-rational agent
would do given the same evidence. That is a principled baseline. It also requires
specifying the rational prior, which is a substantive commitment that moves the
argument to "was that the right prior." The mirrored-arm approach here needs no
prior — only symmetry — at the cost of measuring relative movement rather than
excess movement.

**SWAY** is the closest methodological relative: counterfactual prompting under
positive versus negative linguistic pressure, explicitly isolating framing effects
from content. Same identification logic. Single-turn, and unsupervised-linguistic
rather than rubric-extracted.

**AEDI** provides the single most useful external calibration point: a continuous
credence-slope measure across 500 propositions and 16,000 prompts on eight models.
Claude Sonnet 4.6 (+0.67) and Opus 4.6 (+0.76) scored lowest of the eight tested.
That sets an expectation for the live run — if the models under test here show
large raw mirror gaps, the result is in tension with AEDI and one of the two
instruments needs explaining.

## The specific gap

Three things, and only the combination is novel.

**1. Trajectory, not event.** SYCON-Bench is the only serious multi-turn entry and
its metrics are Turn-of-Flip and Number-of-Flip. A flip is a discrete event. The
failure this benchmark targets produces no flip at all: friction decays, the
consideration inventory attrits asymmetrically, and the stated position never
moves. Figure 7 is that case.

**2. Channel decomposition, model-side.** Rathje et al. (2025) established that
one-sided fact presentation and validating language have *different downstream
effects* on users — extremity versus enjoyment. Vennemeyer et al. (2025, arXiv:2509.21305) showed
agreement and praise are separable in activations and independently steerable. Nobody
has built the behavioural instrument that measures whether a model maintains that
separation across a conversation.

**3. Omission as the primary signal.** The taxonomy paper's central complaint is
that the field measures what is easy — a correct answer becoming incorrect — rather
than what matters: selective framing that never contains a false sentence.
Pre-registered consideration inventories with valence labels make omission
measurable without adjudicating who is right.

## Controls borrowed, not invented

Three design elements are lifted directly from other people's methodological work
and should be credited as such.

**The speaker-free floor** is Hu & Qu (2026, arXiv:2607.05545). Their finding — 66.5% harmful
revision with no speaker present, against 10.3% for a plain re-ask — is the reason
the `nosource` arm exists. Their methodological lesson, stated plainly, is that
source attribution should be measured as an increment above the speaker-free floor.
This applies that lesson in a multi-turn setting, which they did not do.

**The escalation ladder** follows Dubois et al. (2026, arXiv:2602.23971), who found sycophancy rises
monotonically with the epistemic certainty a user expresses (statement → belief →
conviction) and is amplified by I-perspective framing. The ladder here is a dose
with a known dose-response curve rather than an arbitrary script.

**The paired-situation design** — hold the situation fixed, vary only one factor,
attribute the difference — is standard, and is used explicitly in Jain, Park,
Viana, Wilson & Calacci (2025, arXiv:2509.12517) to isolate persona effects from
situational content. (A different author set from the Jain, Yost & Abdullah modes
paper in the landscape table.)

## What would make this redundant

Stated plainly, because a benchmark that cannot be made redundant is not making a
claim.

- If models flip often under sustained pressure, turn-of-flip already captures the
  phenomenon and this adds nothing.
- If content divergence and delivery divergence turn out to be near-perfectly
  correlated across models and scenarios, the channel split is a distinction
  without a difference and a single scalar would do.
- If AAI slope is indistinguishable from zero across every model tested, the
  selective-omission story is wrong and the taxonomy paper's blind-spot claim needs
  revisiting.

The third would be the most interesting outcome and is a publishable result.
