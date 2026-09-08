"""Turn a flat word stream into sentences and then into candidate clips.

Clip boundaries are the single biggest quality lever in a clipper: a short
that starts mid-sentence reads as broken no matter how good the content is.
So candidates are only ever cut on sentence boundaries.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from clipper.config import PlatformPreset
from clipper.models import Candidate, Sentence, Word

TERMINAL_PUNCTUATION = (".", "!", "?", "…")
#: Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "approx.", "inc.", "ltd.", "co.", "u.s.", "u.k.", "a.m.", "p.m.",
}
_INITIAL_RE = re.compile(r"^[A-Z]\.$")


def is_terminal_token(token: str) -> bool:
    """Does this token end a sentence?"""
    stripped = token.strip().strip('"”’\')]')
    if not stripped.endswith(TERMINAL_PUNCTUATION):
        return False
    lowered = stripped.lower()
    if lowered in _ABBREVIATIONS:
        return False
    if _INITIAL_RE.match(stripped):  # "J." in "J. Smith"
        return False
    # A lone decimal point inside a number ("3.") is not a sentence end.
    if stripped[:-1].isdigit() and len(stripped) <= 3:
        return False
    return True


def split_sentences(
    words: Sequence[Word],
    *,
    max_pause: float = 0.7,
    max_words: int = 42,
    min_words: int = 3,
) -> List[Sentence]:
    """Group words into sentences using punctuation, pauses and a hard cap.

    ``max_pause`` catches speech that a caption source stripped of
    punctuation; ``max_words`` stops a rambling monologue from becoming one
    unusable mega-sentence.
    """
    sentences: List[Sentence] = []
    current: List[Word] = []

    for index, word in enumerate(words):
        if current:
            gap = word.start - current[-1].end
            if gap >= max_pause and len(current) >= min_words:
                sentences.append(Sentence(words=current, terminal=False))
                current = []
        current.append(word)

        terminal = is_terminal_token(word.text)
        overlong = len(current) >= max_words
        if (terminal and len(current) >= min_words) or overlong:
            # Prefer breaking at a comma when we hit the hard cap mid-thought.
            if overlong and not terminal:
                pivot = _last_soft_break(current, min_words)
                if pivot is not None:
                    sentences.append(Sentence(words=current[: pivot + 1], terminal=False))
                    current = current[pivot + 1 :]
                    continue
            sentences.append(Sentence(words=current, terminal=terminal))
            current = []
        del index

    if current:
        sentences.append(
            Sentence(words=current, terminal=is_terminal_token(current[-1].text))
        )
    return sentences


def _last_soft_break(words: Sequence[Word], min_words: int) -> Optional[int]:
    """Index of the last comma/semicolon that leaves both halves usable."""
    for i in range(len(words) - min_words, min_words - 1, -1):
        if words[i].text.rstrip('"”').endswith((",", ";", ":", "—", "-")):
            return i
    return None


def build_candidates(
    sentences: Sequence[Sentence],
    preset: PlatformPreset,
    *,
    max_candidates: int = 400,
    max_gap: float = 2.5,
) -> List[Candidate]:
    """Enumerate every run of consecutive sentences that fits the duration window.

    Runs are dropped when they contain a long silent gap: stitching across a
    ten second pause produces a clip that feels edited even though the cut is
    on a sentence boundary.
    """
    candidates: List[Candidate] = []
    total = len(sentences)

    for start_index in range(total):
        run: List[Sentence] = []
        for end_index in range(start_index, total):
            sentence = sentences[end_index]
            if run:
                gap = sentence.start - run[-1].end
                if gap > max_gap:
                    break
            run.append(sentence)
            duration = run[-1].end - run[0].start
            if duration < preset.min_duration:
                continue
            if duration > preset.max_duration:
                break
            lead = (
                run[0].start - sentences[start_index - 1].end if start_index else 6.0
            )
            trail = (
                sentences[end_index + 1].start - run[-1].end
                if end_index + 1 < total
                else 6.0
            )
            candidates.append(
                Candidate(
                    start=run[0].start,
                    end=run[-1].end,
                    sentences=list(run),
                    lead_silence=max(0.0, lead),
                    trail_silence=max(0.0, trail),
                )
            )

    # A single sentence can already exceed max_duration (long unpunctuated
    # captions).  Trim those to the target length on a word boundary rather
    # than losing the moment entirely.
    if not candidates:
        candidates = _trimmed_fallbacks(sentences, preset)

    if len(candidates) > max_candidates:
        # Keep an even spread across the timeline instead of the first N.
        step = len(candidates) / max_candidates
        candidates = [candidates[int(i * step)] for i in range(max_candidates)]
    return candidates


def _trimmed_fallbacks(
    sentences: Sequence[Sentence], preset: PlatformPreset
) -> List[Candidate]:
    """Cut over-long sentences down to the target duration at word boundaries."""
    out: List[Candidate] = []
    for sentence in sentences:
        if sentence.duration < preset.min_duration:
            continue
        kept: List[Word] = []
        for word in sentence.words:
            if kept and word.end - kept[0].start > preset.target_duration:
                break
            kept.append(word)
        if not kept or kept[-1].end - kept[0].start < preset.min_duration:
            continue
        trimmed = Sentence(words=kept, terminal=False)
        out.append(Candidate(start=trimmed.start, end=trimmed.end, sentences=[trimmed]))
    return out


def pad_candidate(
    candidate: Candidate, source_duration: float, *, lead: float = 0.18, tail: float = 0.32
) -> Candidate:
    """Add a little breathing room so the first word is not clipped.

    The tail is larger than the lead because a short that cuts the instant the
    last syllable ends feels abrupt.
    """
    start = max(0.0, candidate.start - lead)
    end = min(source_duration, candidate.end + tail) if source_duration else candidate.end + tail
    return Candidate(
        start=start,
        end=end,
        sentences=candidate.sentences,
        score=candidate.score,
        lead_silence=candidate.lead_silence,
        trail_silence=candidate.trail_silence,
    )
