"""Suggested title, caption and hashtags for each clip.

Deterministic and offline: the hook the clipper already identified is the
best raw material for a caption, so the copy is derived from it rather than
invented.  When ``--llm`` is enabled these fields are overwritten by
:mod:`clipper.llm`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence

from clipper.lexicon import (
    FILLER_WORDS,
    PLATFORM_HASHTAGS,
    STOPWORDS,
    content_words,
)
from clipper.models import Candidate, SocialCopy
from clipper.utils import normalize_whitespace, unique

MAX_TITLE_CHARS = 70
MAX_CAPTION_CHARS = 220
MAX_HOOK_CHARS = 42

_LEADING_FILLER_RE = re.compile(
    r"^(?:(?:so|and|but|okay|ok|well|yeah|um|uh|like|right|now|anyway|i mean)[,\s]+)+",
    re.IGNORECASE,
)
_TRAILING_CONJUNCTION_RE = re.compile(
    r"[\s,]+(?:and|but|so|because|which|that|or|then)\s*$", re.IGNORECASE
)
_HASHTAG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def clean_sentence(text: str) -> str:
    """Trim the verbal throat-clearing off the front and back of a line."""
    cleaned = normalize_whitespace(text)
    cleaned = _LEADING_FILLER_RE.sub("", cleaned)
    cleaned = _TRAILING_CONJUNCTION_RE.sub("", cleaned)
    cleaned = cleaned.strip(" ,;:-")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def truncate(text: str, limit: int, *, ellipsis: str = "...") -> str:
    """Cut at a word boundary, never mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ellipsis)].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or text[: limit - len(ellipsis)]) + ellipsis


def make_title(candidate: Candidate, fallback: str = "") -> str:
    """A short, specific title built from the clip's own hook."""
    hook = clean_sentence(candidate.hook)
    if len(hook) < 12 and candidate.sentences[1:]:
        hook = clean_sentence(f"{hook} {candidate.sentences[1].text}")
    if not hook:
        hook = clean_sentence(fallback) or "Clip"
    title = truncate(hook, MAX_TITLE_CHARS)
    return title.rstrip(".") if title.endswith(".") else title


def make_on_screen_hook(candidate: Candidate) -> str:
    """A very short line to overlay on the first second of the clip."""
    hook = clean_sentence(candidate.hook)
    # Prefer the question or the clause before the first comma: it is the part
    # that actually creates the open loop.
    for separator in ("?", ",", " - ", ":"):
        if separator in hook:
            head = hook.split(separator, 1)[0]
            if separator == "?":
                head += "?"
            if 12 <= len(head) <= MAX_HOOK_CHARS:
                return head
    return truncate(hook, MAX_HOOK_CHARS)


def extract_keywords(text: str, limit: int = 6) -> List[str]:
    """Topic words for hashtags, ranked by frequency then length."""
    words = [w for w in content_words(text) if w.isalpha() and len(w) >= 4]
    words = [w for w in words if w not in FILLER_WORDS]
    if not words:
        return []
    counts = Counter(words)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [word for word, _ in ranked[:limit]]


def extract_proper_nouns(text: str, limit: int = 3) -> List[str]:
    """Mid-sentence capitalised words are usually names worth tagging."""
    tokens = text.split()
    found: List[str] = []
    for index, token in enumerate(tokens):
        word = token.strip(".,!?;:\"'()")
        if index == 0 or not word or not word[0].isupper() or not word.isalpha():
            continue
        previous = tokens[index - 1].strip("\"'")
        if previous.endswith((".", "!", "?")):
            continue  # first word of a sentence, not a name
        if word.lower() in STOPWORDS or len(word) < 3:
            continue
        found.append(word)
    return unique(found)[:limit]


def make_hashtags(
    text: str, platform: str, *, limit: int = 8, extra: Optional[Iterable[str]] = None
) -> List[str]:
    """Topic hashtags first, platform boilerplate last."""
    tags: List[str] = []
    for source in (extract_proper_nouns(text), extract_keywords(text), extra or []):
        for word in source:
            tag = _HASHTAG_CLEAN_RE.sub("", str(word).lower())
            if len(tag) >= 3:
                tags.append(tag)
    tags.extend(PLATFORM_HASHTAGS.get(platform, PLATFORM_HASHTAGS["tiktok"]))
    return ["#" + tag for tag in unique(tags)[:limit]]


def make_caption(candidate: Candidate, title: str, hashtags: Sequence[str]) -> str:
    """Hook line, a supporting line, then the tags."""
    lines = [title]
    supporting = ""
    for sentence in candidate.sentences[1:]:
        text = clean_sentence(sentence.text)
        if len(text) >= 25:
            supporting = text
            break
    if supporting:
        lines.append(truncate(supporting, MAX_CAPTION_CHARS - len(title) - 2))
    body = "\n\n".join(lines)
    return f"{body}\n\n{' '.join(hashtags)}".strip()


def generate_copy(
    candidate: Candidate,
    *,
    platform: str = "tiktok",
    source_title: str = "",
    max_hashtags: int = 8,
) -> SocialCopy:
    """Build the full :class:`~clipper.models.SocialCopy` for one clip."""
    title = make_title(candidate, fallback=source_title)
    hashtags = make_hashtags(candidate.text, platform, limit=max_hashtags)
    return SocialCopy(
        title=title,
        caption=make_caption(candidate, title, hashtags),
        hashtags=hashtags,
        on_screen_hook=make_on_screen_hook(candidate),
    )
