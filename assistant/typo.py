"""Conservative typo autocorrection for the user's input.

The LLM already tolerates typos, so this exists mainly to help the deterministic layers
(memory extraction, the regex intent detectors) which are typo-brittle. It is deliberately
cautious — it only fixes clearly-misspelled lowercase common words at edit-distance 1, and
PROTECTS everything that could be a name, code, or domain term:
  - any word containing an uppercase letter (proper nouns: Ixtlalli, Regenics, …)
  - words on config.AUTOCORRECT_KEEP
  - contractions (anything with an apostrophe)
  - short words (< 3 chars) and non-alphabetic tokens (numbers, emails, paths, code)
The spellchecker is offline; if it isn't installed, correction is a no-op.
"""
import logging
import re

import config

log = logging.getLogger("assistant.typo")

_spell = False  # lazy: False = not yet built, None = unavailable
_TOKEN = re.compile(r"[A-Za-z']+")


def _speller():
    global _spell
    if _spell is False:
        try:
            from spellchecker import SpellChecker
            sp = SpellChecker(distance=1)            # distance 1 = conservative
            sp.word_frequency.load_words(config.AUTOCORRECT_KEEP)  # never "correct" these
            _spell = sp
        except Exception as e:  # noqa: BLE001
            log.debug("spellchecker unavailable (%s) — autocorrect disabled", e)
            _spell = None
    return _spell


def _fix_word(word: str, sp) -> str:
    if (any(c.isupper() for c in word) or len(word) < 3 or "'" in word
            or word.lower() in config.AUTOCORRECT_KEEP):
        return word                                  # protected — leave as-is
    low = word.lower()
    if sp.known([low]):
        return word                                  # already a real word
    cand = sp.correction(low)
    return cand if (cand and cand != low) else word  # only swap for a real correction


def correct(text: str) -> str:
    """Return `text` with obvious typos fixed (no-op if disabled/unavailable)."""
    if not config.AUTOCORRECT or not (text or "").strip():
        return text
    sp = _speller()
    if sp is None:
        return text
    out = _TOKEN.sub(lambda m: _fix_word(m.group(0), sp), text)
    if out != text:
        log.debug("autocorrect: %r -> %r", text, out)
    return out
