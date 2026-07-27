# Threats to validity

Each entry states the threat, the control, and — where it matters — what the
control does *not* fix. The last column is the one that usually goes missing.

---

## Identification

### T1 — Context accumulation masquerading as deference
A twelve-turn conversation has a longer context than a two-turn one. Models
behave differently with more context for reasons that have nothing to do with the
person. Any drift measured against turn index is confounded with this.

**Control:** `neutral` arm — same topic, same turn count, matched token volume,
no stance, no pressure. Whatever drift appears there is the length effect.

**Residual:** the neutral arm is topically matched but not semantically matched.
A stance-free conversation about equity splits is not the same distribution of
tokens as a contested one, so the match is approximate.

### T2 — Repeated text, not social influence
The single largest threat, and the one most benchmarks ignore. Hu & Qu (2026)
found that removing the speaker entirely and leaving only the asserted claim still
produced harmful revision in 66.5% of initially correct cases, against 10.3% for a
plain re-ask. Most of what conformity benchmarks scored as deference to a person
was a language model being moved by text.

**Control:** `nosource` arm. Identical propositional content, presented as
unattributed found material.

**Residual:** removing the speaker also removes the stake, the emotional register,
and the conversational obligation to respond to a person. The arm bounds the
social component; it does not cleanly isolate it. Report UAT as an upper bound on
the social effect, not a point estimate of it.

### T3 — Wording confound between arms
If `pro` and `con` scripts are authored separately, any difference in the model's
response is confounded with wording.

**Control:** both arms are rendered from one template with `{P}`/`{NOT_P}`
placeholders. The loader **refuses** any escalating turn that neither uses a
placeholder nor is explicitly declared `stance_free`. Mirror safety is asserted
by the author, not guessed by a regex.

**Residual:** P and NOT-P are themselves different strings and may differ in
length, fluency, or how commonly each position appears in training data. Balance
them at authoring time and check response length by arm as a diagnostic.

### T4 — Non-contestable propositions
If one side is clearly correct, the `con` arm asks the model to defend nonsense.
Refusal then reads in the data as sycophancy resistance when it is just knowledge.

**Control:** stopping rule — any scenario where the `con` arm produces refusals in
>20% of runs is dropped and reported as dropped.

**Residual:** contestability is a judgement made by the scenario author, and
authors are not neutral about which side is reasonable. Have a second person rate
contestability blind before the scenario enters the set.

---

## Measurement

### T5 — Judge inherits the drift
A judge shown the full conversation evaluates each turn against the thread's own
norms. By turn 10 the drifted register is the local baseline, so the drift becomes
invisible precisely where it is largest. The instrument would fail hardest at the
thing it exists to detect.

**Control:** the judge sees one assistant turn and at most the user turn before
it. Never the conversation. Trajectories are reconstructed from independent
per-turn codings.

**Residual:** some things genuinely require conversational context — whether a
challenge is *new* or a repeat, whether a consideration was raised earlier and
dropped. Those are computed downstream from the sequence of independent codings
rather than asked of the judge.

### T6 — Holistic ratings are unreliable
Ye et al. found single-rater ICC of **.184** among 106 expert humans asked to rate
sycophancy holistically. Their aggregate ICC was .960 — averaging 106 experts
rescued it. A single LLM judge doing the same task inherits the noise floor with
none of the aggregation.

**Control:** extraction, not rating. Which considerations appear; is a challenge
present; what is recommended. Near-factual questions with auditable answers,
which humans can be given identically.

**Residual:** the delivery-channel codes (warmth, mirroring) are irreducibly
gradient and will have lower α than the content codes. Report per-code α, not a
single number.

### T7 — Self-preference in the judge
A model judging its own outputs may systematically rate them favourably.

**Control:** cross-family judge where possible; `self_judging` flag propagated to
the results table where not; embedding cross-check on stance across the full
sample.

**Residual:** the embedding check is crude and TF-IDF-based by default. It
detects gross divergence between the judge and the text, not subtle bias.

### T8 — Noise floor inflating divergence
Channel divergences are L2 distances: strictly positive, therefore inflated by
noise. A perfectly invariant model does not score 0 on content divergence, it
scores the noise floor. Uncorrected, this drags the Harness Ratio toward 0.5 from
both directions and makes a model that holds look like one that half-holds.

**Control:** within-arm noise estimated from replicates of the *same* arm — where
nothing about the input changed, so any divergence is noise by construction —
then subtracted in quadrature.

**Residual:** the correction clips at zero, so a true divergence below the noise
floor returns exactly 0 and "invariant" becomes indistinguishable from "too noisy
to resolve." Both raw and corrected values are reported alongside `noise_share`.
If the correction is doing most of the work, the answer is more replicates, not a
better estimator.

---

## Inference

### T9 — Turns treated as independent observations
Twelve turns from one conversation is not twelve observations. Treating them as
such inflates n twelvefold and every confidence interval becomes fiction.

**Control:** mixed-effects models with random intercepts per conversation; drift
estimated as the `turn × arm` interaction.

### T10 — Generalising from too few scenarios
A result from three scenarios generalises to three scenarios.

**Control:** bootstrap clustered at scenario level, so the effective n is visible
rather than hidden. Minimum eight scenarios for any cross-model claim.

**Residual:** eight is still small. State the scenario set as a scope condition in
every claim.

### T11 — Proving a null
The premise's first invariant asserts an effect is *absent*. `p > .05` does not
establish absence; it also describes an underpowered study.

**Control:** TOST against a pre-specified band. `equivalent=True` requires both
one-sided tests to reject, so a wide interval around zero correctly returns
`False`.

### T12 — Multiplicity
Eleven metrics × arms × models, all tested at .05, guarantees findings.

**Control:** two declared primaries; BH across the secondary family; exploratory
metrics reported without inference.

---

## Construct

### T13 — Accommodation is not always a failure
Warmth toward someone in distress is appropriate. Hedging under genuine
uncertainty is correct. Dropping a consideration because it was already covered is
efficient. A benchmark that scores all of these as sycophancy would reward a cold,
rigid, repetitive model — and Ibrahim et al. (2026) showed the entanglement is
real: training for warmth raises sycophancy, so suppressing one suppresses the
other.

**Control:** this is the reason for the channel split. Delivery divergence is
*not* scored as a failure. The Harness Ratio treats high delivery adaptation with
held content as the target state, not as a problem to minimise.

**Residual:** the split is imposed by the code list, and some behaviours sit
across it. Directness and hedging are coded as delivery but shade into content
when hedging is what carries the retreat. Watch for a model whose stance holds
while its hedging climbs — that is a real pattern and it is partly miscoded here.
Reporting hedging separately as well as inside the vector is the mitigation.

### T14 — The scripted user is not a user
Real people are inconsistent, change subject, get distracted, and escalate
unevenly. The ladder here is monotonic by construction.

**Control:** the ladder follows the epistemic-certainty gradient Dubois et al.
(2026) found produces monotonically rising sycophancy, so it is a defensible dose.
Simulated-user mode exists as an extension.

**Residual:** it is still a dose, not a conversation. External validity here is
low and should be stated as such. This measures a model's response to a
well-characterised pressure profile, which is a different claim from measuring how
models behave with people.

### T15 — Scenario set encodes the author's priors
Which propositions count as contestable, which considerations belong in the
inventory, and how each is labelled for valence are all authorial choices made by
someone with views.

**Control:** inventories are pre-registered and public; valence labels are
independently rated; the metric is a *difference between mirrored arms*, so a
biased inventory biases both arms equally and largely cancels in AAI.

**Residual:** it cancels in AAI. It does not cancel in coverage retention, which
is sensitive to whether the inventory was reasonable in the first place. Read CRC
as descriptive.
