#!/usr/bin/env python3
"""
Mirror-symmetry screen.

The loader (harness/scenarios.py) enforces that every escalating user turn is
either stance-carrying (via a {P}/{NOT_P} placeholder) or explicitly declared
`stance_free`. What it CANNOT check is whether a turn declared stance_free
quietly smuggles a side back in through bare prose - e.g. by reusing a
distinctive word from the focal proposition. That silently breaks the pro/con
mirror, and the whole identification strategy with it.

This is a lexical screen, not a proof. It flags stance_free turns whose wording
overlaps the proposition vocabulary. It over-flags on purpose and false-positives
on common words (it has flagged "through", "building", "standard"), so treat
every flag as a prompt to read the turn and render both arms aloud - never as a
verdict. It never gates the pipeline: it always exits 0.

    python scripts/check_mirror.py                        # the real (study) set
    python scripts/check_mirror.py scenarios/test-scenarios
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.scenarios import load_all, REAL_SCENARIOS_DIR

# Deliberately tiny. A larger stoplist would hide real leaks; the screen is meant
# to over-flag and let a human adjudicate, not to be clever.
_STOP = {
    "the", "and", "for", "with", "without", "into", "onto", "from", "that",
    "this", "these", "those", "than", "then", "are", "was", "were", "been",
    "its", "you", "your", "they", "them", "their", "not", "does", "did", "have",
    "has", "had", "would", "should", "could", "will", "can", "about", "more",
    "most", "some", "any", "here", "there", "what", "which", "who", "how", "why",
    "when", "right", "call", "think", "want", "just", "really", "actually",
}

_WORD = re.compile(r"[a-z][a-z'-]{2,}")   # >= 3 letters


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def check_dir(directory: str | Path) -> int:
    # strict=False so an unrelated validation error elsewhere does not abort the
    # screen; the load-and-validate step is a separate preflight.
    scenarios = load_all(directory, strict=False)
    if not scenarios:
        print(f"  (no scenarios loaded from {directory})")
        return 0
    flags = 0
    for sid in sorted(scenarios):
        sc = scenarios[sid]
        vocab = _content_words(sc.proposition_p) | _content_words(sc.proposition_not_p)
        for t in sc.user_turns_pro:
            if not t.stance_free:
                continue
            overlap = sorted(_content_words(t.text) & vocab)
            if overlap:
                flags += 1
                print(f"  {sid} turn {t.index}: stance_free but reuses proposition "
                      f"vocabulary {overlap}")
                print(f"      {' '.join(t.text.split())[:160]}")
    return flags


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else REAL_SCENARIOS_DIR
    directory = Path(arg)
    if not directory.exists():
        sys.exit(f"no such directory: {directory}")

    print(f"mirror screen over {directory}\n")
    flags = check_dir(directory)
    if flags:
        print(f"\n{flags} turn(s) flagged. A flag is a prompt to read the turn and "
              "render\nboth arms aloud, not a verdict - the screen false-positives "
              "on common words.")
    else:
        print("no stance_free turns reuse proposition vocabulary.")
    return 0   # informational screen; never gates the pipeline


if __name__ == "__main__":
    raise SystemExit(main())
