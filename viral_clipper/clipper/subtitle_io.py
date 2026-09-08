"""Reading and writing SRT / WebVTT.

Caption files are both an input (a fast, free transcript when the source
already has captions) and an output (an SRT next to every rendered clip).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, NamedTuple, Sequence

from clipper.models import Word
from clipper.utils import normalize_whitespace, srt_timecode

_TIME_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})"
)
_ARROW_RE = re.compile(r"-->")
_TAG_RE = re.compile(r"<[^>]+>")
_VTT_INLINE_TS_RE = re.compile(r"<(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})>")
_INDEX_RE = re.compile(r"^\d+$")


class Cue(NamedTuple):
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    match = _TIME_RE.search(value)
    if not match:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    millis = match.group("ms").ljust(3, "0")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(millis) / 1000.0
    )


def parse_cues(text: str) -> List[Cue]:
    """Parse SRT or WebVTT content into cues, ignoring styling blocks."""
    cues: List[Cue] = []
    block: List[str] = []

    def flush(lines: Sequence[str]) -> None:
        timing_index = next(
            (i for i, line in enumerate(lines) if _ARROW_RE.search(line)), None
        )
        if timing_index is None:
            return
        left, _, right = lines[timing_index].partition("-->")
        try:
            start = parse_timestamp(left)
            end = parse_timestamp(right)
        except ValueError:
            return
        body_lines = [line for line in lines[timing_index + 1 :] if line.strip()]
        body = " ".join(body_lines)
        body = _VTT_INLINE_TS_RE.sub(" ", body)
        body = _TAG_RE.sub("", body)
        body = html.unescape(body)
        body = normalize_whitespace(body)
        if body and end > start:
            cues.append(Cue(start, end, body))

    for raw_line in text.splitlines():
        line = raw_line.rstrip("﻿").rstrip()
        if line.strip():
            block.append(line)
            continue
        flush(block)
        block = []
    flush(block)

    # YouTube auto-captions repeat the previous cue as a rolling window; drop
    # any cue whose text is fully contained in the one before it.
    deduped: List[Cue] = []
    for cue in cues:
        if deduped and cue.text and cue.text in deduped[-1].text:
            continue
        if deduped and deduped[-1].text and deduped[-1].text in cue.text:
            deduped[-1] = Cue(deduped[-1].start, cue.end, cue.text)
            continue
        deduped.append(cue)
    return deduped


def cues_to_words(cues: Sequence[Cue]) -> List[Word]:
    """Approximate word timings by spreading each cue over its own words.

    Time is allocated proportionally to word length, which tracks real speech
    far better than an even split.  Confidence is lowered so downstream
    consumers know these timings are estimates.
    """
    words: List[Word] = []
    for cue in cues:
        tokens = [t for t in cue.text.split() if t]
        if not tokens:
            continue
        span = max(0.05, cue.end - cue.start)
        total = sum(len(t) for t in tokens) or len(tokens)
        cursor = cue.start
        for token in tokens:
            share = span * (len(token) / total)
            end = min(cue.end, cursor + share)
            words.append(Word(text=token, start=cursor, end=max(cursor + 0.02, end), confidence=0.5))
            cursor = end
    words.sort(key=lambda w: w.start)
    return words


def load_words(path: Path) -> List[Word]:
    """Read an SRT/VTT file into approximate word timings."""
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    return cues_to_words(parse_cues(content))


def words_to_srt(words: Sequence[Word], *, max_words: int = 8, max_gap: float = 0.8) -> str:
    """Render words back out as an SRT, grouping them into readable cues."""
    groups: List[List[Word]] = []
    current: List[Word] = []
    for word in words:
        if current and (
            len(current) >= max_words or word.start - current[-1].end > max_gap
        ):
            groups.append(current)
            current = []
        current.append(word)
        if word.text.endswith((".", "!", "?")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    lines: List[str] = []
    for index, group in enumerate(groups, start=1):
        if not group:
            continue
        text = normalize_whitespace(" ".join(w.text for w in group))
        lines.append(str(index))
        lines.append(f"{srt_timecode(group[0].start)} --> {srt_timecode(group[-1].end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
