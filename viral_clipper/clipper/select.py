"""Pick a final set of clips from the scored candidates.

Two candidates that share 90% of their words are the same moment cut two
ways, so a naive "top N by score" returns five near-identical clips.  This
module applies non-maximum suppression on the timeline *and* on the text.
"""

from __future__ import annotations

from typing import List, Sequence, Set

from clipper.lexicon import content_words
from clipper.models import Candidate
from clipper.utils import overlap


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Set overlap in ``[0, 1]``; two empty sets are treated as unrelated."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


def temporal_overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Shared seconds as a fraction of the *shorter* candidate.

    Using the shorter side means a 20s clip fully contained in a 60s one
    scores 1.0, which is what we want: it is the same moment.
    """
    shortest = min(a.duration, b.duration)
    if shortest <= 0:
        return 0.0
    return overlap(a.start, a.end, b.start, b.end) / shortest


def select_clips(
    candidates: Sequence[Candidate],
    *,
    max_clips: int = 5,
    max_overlap: float = 0.25,
    min_score: float = 0.0,
    similarity_limit: float = 0.65,
    min_gap: float = 0.0,
) -> List[Candidate]:
    """Greedy, score-ordered selection with timeline and content suppression.

    ``candidates`` may be in any order; they are re-sorted by score here so the
    function is safe to call on raw output.
    """
    ranked = sorted(candidates, key=lambda c: c.score.total, reverse=True)
    chosen: List[Candidate] = []
    chosen_words: List[Set[str]] = []

    for candidate in ranked:
        if len(chosen) >= max_clips:
            break
        if candidate.score.total < min_score:
            break  # ranked order means everything after this is worse too
        if candidate.duration <= 0:
            continue

        words = set(content_words(candidate.text))
        conflict = False
        for accepted, accepted_words in zip(chosen, chosen_words):
            if temporal_overlap_ratio(candidate, accepted) > max_overlap:
                conflict = True
                break
            if min_gap > 0 and _gap(candidate, accepted) < min_gap:
                conflict = True
                break
            if jaccard(words, accepted_words) > similarity_limit:
                conflict = True
                break
        if conflict:
            continue

        chosen.append(candidate)
        chosen_words.append(words)

    chosen.sort(key=lambda c: c.start)
    return chosen


def _gap(a: Candidate, b: Candidate) -> float:
    """Seconds between two non-overlapping candidates (0 if they touch)."""
    if a.start >= b.end:
        return a.start - b.end
    if b.start >= a.end:
        return b.start - a.end
    return 0.0
