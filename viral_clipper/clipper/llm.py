"""Optional LLM re-ranking of candidate moments.

The heuristic scorer in :mod:`clipper.scoring` is fast, free and explainable,
but it cannot tell that a story actually lands.  When ``--llm`` is enabled the
top candidates are sent to Claude for a second opinion, and the two scores are
blended.

This module is entirely optional: if the ``anthropic`` package or an API key
is missing, the pipeline logs a note and carries on with heuristic scores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from clipper.models import Candidate, SocialCopy
from clipper.utils import clamp, log, timecode

DEFAULT_MODEL = "claude-opus-5"
MAX_TRANSCRIPT_CHARS = 1200

SYSTEM_PROMPT = """You rate candidate clips cut from a longer video for \
short-form platforms (TikTok, Instagram Reels, YouTube Shorts).

For each candidate, judge how well it would perform as a standalone vertical \
short shown to someone who has never seen the source video. Weigh, in order:

1. Hook - does the first sentence stop a scroll within two seconds?
2. Self-containment - does it make sense with no other context?
3. Payoff - does it resolve what it opens, rather than trailing off?
4. Emotional or informational punch - is there a reason to send it to a friend?

Score 0-100. Be harsh and use the full range: a rambling or context-dependent \
excerpt is below 30, a genuinely shareable moment is above 75. Do not reward \
a clip for being about an interesting topic if the excerpt itself is flat.

Also write a title (max 70 characters, specific, no clickbait punctuation \
spam) and a short on-screen hook (max 42 characters) for each candidate."""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                    "on_screen_hook": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "title", "on_screen_hook", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ratings"],
    "additionalProperties": False,
}


@dataclass
class LLMRating:
    index: int
    score: float
    title: str = ""
    on_screen_hook: str = ""
    reason: str = ""


class LLMUnavailable(RuntimeError):
    """The Anthropic SDK or an API key is missing."""


def _client():
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise LLMUnavailable(
            "the 'anthropic' package is not installed (pip install anthropic)"
        ) from exc
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise LLMUnavailable("no ANTHROPIC_API_KEY in the environment")
    return anthropic


def build_prompt(candidates: Sequence[Candidate], source_title: str = "") -> str:
    """Render the candidates as a numbered list for the model."""
    lines: List[str] = []
    if source_title:
        lines.append(f"Source video: {source_title}")
    lines.append(f"Rate all {len(candidates)} candidates below.\n")
    for index, candidate in enumerate(candidates):
        transcript = candidate.text
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS].rsplit(" ", 1)[0] + "..."
        lines.append(
            f"[{index}] {timecode(candidate.start, 0)}-{timecode(candidate.end, 0)} "
            f"({candidate.duration:.0f}s)\n{transcript}\n"
        )
    return "\n".join(lines)


def rate_candidates(
    candidates: Sequence[Candidate],
    *,
    model: str = DEFAULT_MODEL,
    source_title: str = "",
    max_tokens: int = 8000,
) -> List[LLMRating]:
    """Ask Claude to score each candidate. Raises :class:`LLMUnavailable`."""
    if not candidates:
        return []
    anthropic = _client()
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(candidates, source_title)}],
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
    except anthropic.BadRequestError as exc:
        raise LLMUnavailable(f"request rejected: {exc}") from exc
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("the API key was rejected") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("rate limited by the Anthropic API") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"API error {exc.status_code}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("could not reach the Anthropic API") from exc

    if response.stop_reason == "refusal":
        raise LLMUnavailable("the model declined to rate this content")

    text = next((b.text for b in response.content if b.type == "text"), "")
    return parse_ratings(text, len(candidates))


def parse_ratings(text: str, count: int) -> List[LLMRating]:
    """Parse the model's JSON response, ignoring anything out of range."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LLMUnavailable(f"could not parse the model response: {exc}") from exc

    ratings: List[LLMRating] = []
    for entry in data.get("ratings", []):
        try:
            index = int(entry["id"])
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= index < count:
            continue
        ratings.append(
            LLMRating(
                index=index,
                score=clamp(score, 0.0, 100.0),
                title=str(entry.get("title", "")).strip(),
                on_screen_hook=str(entry.get("on_screen_hook", "")).strip(),
                reason=str(entry.get("reason", "")).strip(),
            )
        )
    return ratings


def apply_ratings(
    candidates: Sequence[Candidate],
    ratings: Sequence[LLMRating],
    *,
    weight: float = 0.5,
) -> Dict[int, SocialCopy]:
    """Blend LLM scores into the candidates and return their suggested copy.

    Candidates the model did not rate keep their heuristic score untouched,
    so a partial response degrades gracefully instead of reordering the list
    around missing data.
    """
    weight = clamp(weight, 0.0, 1.0)
    copy_by_index: Dict[int, SocialCopy] = {}
    for rating in ratings:
        candidate = candidates[rating.index]
        heuristic = candidate.score.total
        candidate.score.total = (1.0 - weight) * heuristic + weight * rating.score
        candidate.score.signals["llm"] = rating.score / 100.0
        candidate.score.weights["llm"] = 0.0  # informational, already blended in
        if rating.title or rating.on_screen_hook:
            copy_by_index[rating.index] = SocialCopy(
                title=rating.title[:70],
                on_screen_hook=rating.on_screen_hook[:42],
            )
    return copy_by_index


def rerank(
    candidates: Sequence[Candidate],
    *,
    model: str = DEFAULT_MODEL,
    weight: float = 0.5,
    limit: int = 12,
    source_title: str = "",
) -> Dict[int, SocialCopy]:
    """Rate the top ``limit`` candidates and blend the result in place.

    Returns suggested copy keyed by index into ``candidates``.  Never raises:
    on any failure it logs and returns an empty mapping.
    """
    if not candidates:
        return {}
    subset = list(candidates)[:limit]
    try:
        ratings = rate_candidates(subset, model=model, source_title=source_title)
    except LLMUnavailable as exc:
        log.warning("LLM re-ranking skipped: %s", exc)
        return {}
    log.info("LLM rated %d/%d candidates", len(ratings), len(subset))
    return apply_ratings(subset, ratings, weight=weight)
