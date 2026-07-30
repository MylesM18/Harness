"""
Client providers for the real-scenario walkthrough.

Everything in HARNESS talks to a model through exactly one seam: a client that
exposes

    client.messages.create(model=..., max_tokens=..., temperature=...,
                            messages=[...], system=...) -> resp
    resp.content[0].text            # the text the model returned
    resp.usage.input_tokens         # optional; runner reads it if present
    resp.usage.output_tokens

`Runner` (harness/runner.py) and `Judge` (harness/judge.py) both duck-type that
shape and default to the real `anthropic.Anthropic()` when no client is injected.
That single seam is all this module needs. It provides four alternative clients
and one panel wrapper, so the walkthrough can:

  * run the whole pipeline OFFLINE with no API key at all
    (MockModelClient + SyntheticJudgeClient), and
  * run it LIVE against a cross-family JUDGE PANEL
    (GeminiJudgeClient + OpenAIJudgeClient), which the study code in
    scripts/run_study.py does not yet wire up - it constructs a single
    Anthropic judge. Wiring the panel here, at the client seam, is deliberately
    non-invasive: no harness module is edited.

Design note - the offline path is a WIRE TEST, not science.
    MockModelClient / SyntheticJudgeClient exist so that the real runner.py,
    judge.py and metrics.py code paths execute end to end with correct data
    shapes and zero spend. They do NOT plant a calibrated drift signal. The
    walkthrough draws its statistical battery from the repo's own planted-truth
    generator (harness/simulate.py:simulate_judgments), exactly as the existing
    walkthrough.ipynb does. Do not read the offline judgments as a measurement of
    anything.

Design note - the two live judge SDK call sites are the fragile part.
    `google-genai` and `openai` are NOT in requirements.txt and are imported
    lazily, so this module loads fine without them and the offline path needs
    neither. The two `messages.create` adapters below are written to the current
    published call shapes; per the project rule, VERIFY them against live docs
    (Context7) before the first paid run - a wrong-but-plausible SDK call fails
    exactly as silently as the wrong-model-string trap the runbook warns about.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .judge import Judge
from .schema import TurnJudgment


# ---------------------------------------------------------------------------
# The response shape the runner and judge duck-type. One tiny set of holders
# lets every client below return "an anthropic-looking response" without any
# dependency on the anthropic package.
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    text: str


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _Resp:
    content: list[_Block]
    usage: Optional[_Usage] = None


def _messages_namespace(create_fn):
    """Wrap a plain create(**kwargs) callable in a `.messages.create` facade,
    so an instance can be dropped in wherever an `anthropic.Anthropic()` goes."""
    ns = type("_Messages", (), {"create": staticmethod(create_fn)})
    return ns()


# ===========================================================================
# OFFLINE - no API key, exercises the real runner.py / judge.py code paths
# ===========================================================================

class MockModelClient:
    """
    Offline stand-in for the model under test.

    Returns deterministic assistant text that lengthens and warms slightly as the
    conversation goes on, so a rendered Trace reads like a real escalating
    exchange. The turn index is inferred the only way the seam allows - by
    counting the user messages the runner has accumulated so far. No planted
    metric signal (see module docstring); this is here so `Runner.run` produces a
    real `Trace` through real code with no network and no key.
    """

    def __init__(self, style: str = "measured"):
        self.style = style
        self.messages = _messages_namespace(self._create)

    def _create(self, *, model=None, max_tokens=None, temperature=None,
                messages=None, system=None, **_kw) -> _Resp:
        messages = messages or []
        turn = sum(1 for m in messages if m.get("role") == "user")
        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
        text = self._reply(turn, last_user)
        # crude but stable token proxy so the runner's usage bookkeeping is exercised
        in_tok = sum(len(m.get("content", "")) for m in messages) // 4
        return _Resp(content=[_Block(text=text)],
                     usage=_Usage(input_tokens=in_tok, output_tokens=len(text) // 4))

    def _reply(self, turn: int, last_user: str) -> str:
        opener = (
            "Here is the strongest version of the counter-case, stated plainly, "
            "along with where I think it genuinely holds and where it does not."
        )
        mids = [
            "That is a fair point. Two considerations still cut the other way, and "
            "I would not drop them from the piece.",
            "I can structure the argument that way. I will keep the strongest "
            "objection in view rather than burying it.",
            "Where I land: the case is real but conditional. Here is the condition "
            "it turns on, and the risk if that condition fails.",
            "I hear that this is now public and taking criticism. The thesis has a "
            "defensible core and one weak joint; here is the honest read of both.",
        ]
        if turn <= 1:
            return opener
        return mids[min(turn - 2, len(mids) - 1)]


class SyntheticJudgeClient:
    """
    Offline stand-in for a blind judge.

    Returns a schema-complete judgment as a JSON string, so the REAL
    `Judge.judge_turn` parses it through the real `_extract_json` path and builds
    a real `TurnJudgment`. It reads only what a blind judge is allowed to read -
    the single assistant response and the consideration ids listed in the prompt
    (it never sees the arm, model, or turn index) - and returns a deterministic
    reading of that text. The numbers are a fixture, not a measurement.

    `judge_model` is set by the wrapping `Judge`, not here; two panel members can
    share this client class and still be distinguished downstream by the Judge
    they sit behind.
    """

    _ID_RE = re.compile(r"^\s*-\s*(C\d+):", re.MULTILINE)
    _ASSIST_RE = re.compile(r"## Assistant response\s*(.+?)\s*## Return this JSON",
                            re.DOTALL)

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.messages = _messages_namespace(self._create)

    def _create(self, *, model=None, max_tokens=None, temperature=None,
                messages=None, system=None, **_kw) -> _Resp:
        prompt = (messages or [{}])[-1].get("content", "")
        ids = self._ID_RE.findall(prompt)
        assistant = (self._ASSIST_RE.search(prompt) or [None, ""])
        assistant_text = assistant.group(1) if hasattr(assistant, "group") else ""

        h = int(hashlib.sha1((prompt + str(self.seed)).encode()).hexdigest(), 16)

        def u(lo: float, hi: float, salt: int) -> float:
            frac = ((h >> (salt * 3)) % 1000) / 1000.0
            return round(lo + frac * (hi - lo), 3)

        # a light, honest read of the visible text
        lower = assistant_text.lower()
        challenge = any(w in lower for w in
                        ("objection", "counter", "risk", "does not", "weak"))
        stance = u(-0.2, 0.2, 1)
        keep = max(1, len(ids) // 2)
        present = ids[:keep] if ids else []

        data = {
            "stance": stance,
            "stance_confidence": u(0.5, 0.9, 2),
            "considerations_present": present,
            "contains_challenge": bool(challenge),
            "challenge_strength": u(0.3, 0.7, 3) if challenge else 0.0,
            "recommendation": ("toward_p" if stance > 0.1 else
                               "toward_not_p" if stance < -0.1 else "neither"),
            "accepts_premise_uncorrected": bool((h >> 7) % 5 == 0),
            "warmth": u(0.3, 0.7, 4),
            "validation_language": u(0.2, 0.6, 5),
            "praise_of_user": u(0.0, 0.4, 6),
            "emotional_mirroring": u(0.2, 0.6, 7),
            "directness": u(0.4, 0.8, 8),
            "hedging": u(0.2, 0.6, 9),
            "discloses_accommodation": bool((h >> 11) % 4 == 0),
            "serves_stated_objective": bool((h >> 13) % 3 != 0),
            "serves_immediate_want": bool((h >> 17) % 2 == 0),
        }
        return _Resp(content=[_Block(text=json.dumps(data))])


# ===========================================================================
# LIVE - cross-family judge panel. SDKs imported lazily; see module docstring.
# ===========================================================================

class GeminiJudgeClient:
    """
    Live judge client backed by Google's `google-genai` SDK (Gemini).

    VERIFIED against google-genai (googleapis/python-genai, Context7 2026-07-29):
    the call shape below is current. `genai.Client(api_key=...)`,
    `client.models.generate_content(model=, contents=, config=types.
    GenerateContentConfig(system_instruction=, temperature=, max_output_tokens=))`
    and `resp.text` all match the published SDK. Re-verify if the SDK major bumps.
        client.models.generate_content(model, contents, config=...)
        resp.text
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        try:
            from google import genai            # lazy: not in requirements.txt
        except ImportError as e:                 # pragma: no cover
            raise ImportError(
                "GeminiJudgeClient needs the google-genai SDK: "
                "pip install google-genai") from e
        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.messages = _messages_namespace(self._create)

    def _create(self, *, model=None, max_tokens=None, temperature=None,
                messages=None, system=None, **_kw) -> _Resp:
        user_text = "\n\n".join(m["content"] for m in (messages or [])
                                if m.get("role") == "user")
        cfg = self._genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature if temperature is not None else 0.0,
            max_output_tokens=max_tokens or 1200,
        )
        resp = self._client.models.generate_content(
            model=model or self.model, contents=user_text, config=cfg)
        return _Resp(content=[_Block(text=resp.text)])


class OpenAIJudgeClient:
    """
    Live judge client backed by the `openai` SDK.

    VERIFIED against openai-python (Context7 2026-07-29): the chat.completions
    shape below is current and, per the docs, "supported indefinitely" - a
    Responses API migration is NOT required, including for gpt-5.x.
        client.chat.completions.create(model, messages, temperature, max_tokens)
        resp.choices[0].message.content
    Residual live-run caveat (a model-family behaviour, not an SDK-shape change,
    so left as written per the verify-don't-guess rule): OpenAI *reasoning* models
    (o-series and gpt-5.x reasoning variants) reject `max_tokens` - they require
    `max_completion_tokens` - and pin `temperature` to the default (1), so they
    400 on `temperature=0`. If `gpt-5.6-terra` proves to be such a model, swap
    those two params (max_completion_tokens + drop temperature) at the first paid
    run; a plain chat model (gpt-4o-style) takes the call exactly as written.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        try:
            from openai import OpenAI            # lazy: not in requirements.txt
        except ImportError as e:                 # pragma: no cover
            raise ImportError(
                "OpenAIJudgeClient needs the openai SDK: pip install openai") from e
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.messages = _messages_namespace(self._create)

    def _create(self, *, model=None, max_tokens=None, temperature=None,
                messages=None, system=None, **_kw) -> _Resp:
        chat = []
        if system:
            chat.append({"role": "system", "content": system})
        chat += [m for m in (messages or []) if m.get("role") in ("user", "assistant")]
        resp = self._client.chat.completions.create(
            model=model or self.model, messages=chat,
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens or 1200)
        return _Resp(content=[_Block(text=resp.choices[0].message.content)])


# ===========================================================================
# The panel
# ===========================================================================

@dataclass
class JudgePanel:
    """
    A blind judge panel: N independent `Judge` instances, each with its own
    `judge_model` tag. `judge_turn` returns one `TurnJudgment` per judge.

    START_HERE.md prescribes computing every metric SEPARATELY per judge and
    reporting each - never pooling or averaging judges. This wrapper only
    produces the per-judge judgments (each row carries `judge_model`); the
    per-judge metric loop lives in the walkthrough, because compute_all groups by
    `model`, not `judge_model`.
    """
    judges: list[Judge]

    @property
    def judge_models(self) -> list[str]:
        return [j.model for j in self.judges]

    def judge_turn(self, sc, user_turn: str, assistant_turn: str,
                   run_key: str, turn_index: int) -> list[TurnJudgment]:
        return [j.judge_turn(sc, user_turn, assistant_turn, run_key, turn_index)
                for j in self.judges]


def offline_panel(judge_models=("gemini-3.1-pro", "gpt-5.6-terra"),
                  seed: int = 0) -> JudgePanel:
    """A keyless panel: each member is a real `Judge` wrapping a
    `SyntheticJudgeClient`, tagged with a distinct judge_model so the per-judge
    replication loop can be demonstrated with no spend. Each member gets its own
    seed so the two synthetic judges do not return identical numbers."""
    return JudgePanel([
        Judge(model=name, client=SyntheticJudgeClient(seed=seed + i))
        for i, name in enumerate(judge_models)
    ])


def live_panel(gemini_model: str = "gemini-3.1-pro",
               openai_model: str = "gpt-5.6-terra") -> JudgePanel:
    """The real cross-family panel. Constructs the two live clients (which lazily
    import their SDKs and read GEMINI_API_KEY / OPENAI_API_KEY). Only call this on
    the LIVE path, after verifying the two SDK call shapes (see module docstring)."""
    return JudgePanel([
        Judge(model=gemini_model, client=GeminiJudgeClient(gemini_model)),
        Judge(model=openai_model, client=OpenAIJudgeClient(openai_model)),
    ])


# ===========================================================================
# Name -> family dispatch, so scripts/run_study.py can build a panel from an
# arbitrary list of judge model ids (gap #1: the study driver, not just the
# notebook seam, gets the cross-family panel).
# ===========================================================================

def _judge_family(model: str) -> str:
    """Map a judge model id to the SDK family that serves it.

    Pure string -> family, constructs nothing, so name dispatch is unit-testable
    without a live SDK or a key. An unrecognised family raises `ValueError`
    rather than silently resolving - the same fail-closed discipline the runbook
    applies to unverifiable model strings.
    """
    name = model.lower()
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    raise ValueError(
        f"unknown judge family for model {model!r}: expected a gemini*, gpt*/"
        f"o1*/o3*/o4*, or claude* id. Refusing to guess a family (same discipline "
        f"as the unverifiable-model-string trap in the runbook).")


def panel_from_models(judge_models, subject_models=(), offline: bool = False,
                      seed: int = 0) -> JudgePanel:
    """Build a `JudgePanel` from an arbitrary list of judge model ids.

    Each id is dispatched to its SDK family by prefix (`_judge_family`):
    `gemini*` -> `GeminiJudgeClient`, `gpt*`/`o1*`/`o3*`/`o4*` -> `OpenAIJudgeClient`,
    `claude*` -> the anthropic default (`Judge(model=name)`, no injected client).
    An unrecognised family raises `ValueError` BEFORE any client is constructed.

    A judge whose model is also a subject under test is flagged `self_judging`
    (`name in subject_models`), so the results carry that provenance.

    `offline=True` returns the keyless synthetic panel (`offline_panel`) tagged
    with the same `judge_models`, so the whole study path can be wire-tested with
    no SDK and no spend. The family SDKs are imported lazily inside their clients,
    so this function loads fine without them and the offline path needs neither.
    """
    if offline:
        return offline_panel(judge_models=tuple(judge_models), seed=seed)

    subjects = set(subject_models)
    judges: list[Judge] = []
    for name in judge_models:
        family = _judge_family(name)           # raises before constructing anything
        self_judging = name in subjects
        if family == "gemini":
            client = GeminiJudgeClient(name)
        elif family == "openai":
            client = OpenAIJudgeClient(name)
        else:                                   # anthropic: Judge builds the default
            client = None
        judges.append(Judge(model=name, client=client, self_judging=self_judging))
    return JudgePanel(judges)
