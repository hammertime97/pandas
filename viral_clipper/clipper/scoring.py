"""Predict how well a candidate window would perform as a short.

Every signal returns a value in ``[0, 1]`` and is combined with the weights
from the platform preset.  Penalties are applied multiplicatively afterwards,
because a clip that starts mid-sentence is bad *regardless* of how strong its
other signals are.

The output is a 0-100 score plus a full :class:`~clipper.models.ScoreBreakdown`
so the UI can show *why* a moment was picked.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from clipper.audio import EnergyProfile, WindowEnergy
from clipper.config import PlatformPreset
from clipper.lexicon import (
    CURIOSITY_MARKERS,
    CURIOSITY_PHRASES,
    DANGLING_OPENERS,
    EMOTION_WORDS,
    FILLER_WORDS,
    HOOK_PATTERNS,
    INTENSIFIERS,
    LAUGHTER_PATTERNS,
    LIST_PATTERNS,
    PAYOFF_PHRASES,
    QUOTABLE_PATTERNS,
    content_words,
    tokenize,
)
from clipper.models import Candidate, ScoreBreakdown
from clipper.utils import clamp

HOOK_WINDOW_WORDS = 14


def saturate(value: float, half: float) -> float:
    """Map ``[0, inf)`` to ``[0, 1)``, reaching 0.5 at ``value == half``.

    Used everywhere a signal should have diminishing returns: the tenth
    emotional word matters far less than the first.
    """
    if value <= 0 or half <= 0:
        return 0.0
    return value / (value + half)


def bell(value: float, center: float, width: float) -> float:
    """Gaussian preference curve, 1.0 at ``center``."""
    if width <= 0:
        return 1.0 if value == center else 0.0
    return math.exp(-0.5 * ((value - center) / width) ** 2)


@dataclass
class ScoringContext:
    """Everything a signal function is allowed to look at."""

    candidate: Candidate
    preset: PlatformPreset
    energy: WindowEnergy
    source_duration: float
    text: str
    lowered: str
    tokens: List[str]
    hook_text: str

    @property
    def duration(self) -> float:
        return self.candidate.duration

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def words_per_second(self) -> float:
        return self.word_count / self.duration if self.duration > 0 else 0.0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def signal_hook(ctx: ScoringContext) -> float:
    """How hard the opening line stops the scroll."""
    hook = ctx.hook_text
    if not hook:
        return 0.0
    matched = [weight for pattern, weight in HOOK_PATTERNS if pattern.search(hook)]
    if not matched:
        base = 0.0
    else:
        # The strongest pattern dominates; extras add a shrinking bonus.
        matched.sort(reverse=True)
        base = matched[0] + sum(w * 0.25 for w in matched[1:3])

    if "?" in ctx.candidate.hook:
        base += 0.25
    if re.search(r"\b\d+\b", hook):
        base += 0.1
    # A hook that opens on "you" is addressed to the viewer.
    if re.match(r"^(you|your)\b", hook):
        base += 0.15
    return clamp(base, 0.0, 1.0)


def signal_curiosity(ctx: ScoringContext) -> float:
    """Open loops: questions, teases and "here's the thing" constructions."""
    phrase_hits = sum(1 for pattern in CURIOSITY_PHRASES if pattern.search(ctx.lowered))
    marker_hits = sum(1 for token in ctx.tokens if token in CURIOSITY_MARKERS)
    questions = ctx.text.count("?")
    density = marker_hits / max(1, ctx.word_count) * 100.0
    return clamp(
        0.5 * saturate(phrase_hits, 1.0)
        + 0.3 * saturate(density, 2.5)
        + 0.2 * saturate(questions, 1.0),
        0.0,
        1.0,
    )


def signal_payoff(ctx: ScoringContext) -> float:
    """Does the window resolve, or does it stop mid-argument?

    Resolution is worth more in the back half of the clip, so matches are
    weighted by where they occur.
    """
    if not ctx.lowered:
        return 0.0
    total = 0.0
    for pattern in PAYOFF_PHRASES:
        match = pattern.search(ctx.lowered)
        if not match:
            continue
        position = match.start() / max(1, len(ctx.lowered))
        total += 0.6 + 0.4 * position
    # Ending on a complete sentence is itself a form of payoff.
    if ctx.candidate.sentences and ctx.candidate.sentences[-1].terminal:
        total += 0.5
    return clamp(saturate(total, 1.1), 0.0, 1.0)


def signal_emotion(ctx: ScoringContext) -> float:
    """Density of high-arousal vocabulary and intensifiers."""
    if not ctx.tokens:
        return 0.0
    emotional = sum(1 for token in ctx.tokens if token in EMOTION_WORDS)
    intense = sum(1 for token in ctx.tokens if token in INTENSIFIERS)
    exclamations = ctx.text.count("!")
    per_hundred = (emotional + 0.4 * intense) / len(ctx.tokens) * 100.0
    return clamp(0.85 * saturate(per_hundred, 4.0) + 0.15 * saturate(exclamations, 1.0), 0.0, 1.0)


def signal_quotability(ctx: ScoringContext) -> float:
    """Short, absolute, second-person lines screenshot well."""
    hits = sum(1 for pattern in QUOTABLE_PATTERNS if pattern.search(ctx.lowered))
    punchy = 0
    for sentence in ctx.candidate.sentences:
        length = len(sentence.words)
        if 4 <= length <= 14 and sentence.terminal:
            punchy += 1
    punchy_ratio = punchy / max(1, len(ctx.candidate.sentences))
    return clamp(0.6 * saturate(hits, 1.2) + 0.4 * punchy_ratio, 0.0, 1.0)


def signal_completeness(ctx: ScoringContext) -> float:
    """Starts on a clean thought and ends on a finished one."""
    sentences = ctx.candidate.sentences
    if not sentences:
        return 0.0
    score = 0.4  # every candidate already starts on a sentence boundary
    first_token = (ctx.tokens[0] if ctx.tokens else "")
    if first_token not in DANGLING_OPENERS:
        score += 0.3
    if sentences[-1].terminal:
        score += 0.3
    # Trailing conjunction means the thought is left hanging.
    if ctx.tokens and ctx.tokens[-1] in DANGLING_OPENERS:
        score -= 0.25
    return clamp(score, 0.0, 1.0)


def signal_audio_energy(ctx: ScoringContext) -> float:
    return clamp(ctx.energy.excitement, 0.0, 1.0)


def signal_pace(ctx: ScoringContext) -> float:
    """Words per second inside the range that reads as energetic but clear."""
    if ctx.duration <= 0:
        return 0.0
    return clamp(bell(ctx.words_per_second, center=2.9, width=1.0), 0.0, 1.0)


def signal_information_density(ctx: ScoringContext) -> float:
    """Content words per token, minus filler."""
    if not ctx.tokens:
        return 0.0
    content = len(content_words(ctx.text))
    filler = sum(1 for token in ctx.tokens if token in FILLER_WORDS)
    density = content / len(ctx.tokens)
    filler_ratio = filler / len(ctx.tokens)
    unique_ratio = len(set(ctx.tokens)) / len(ctx.tokens)
    return clamp(0.55 * density + 0.25 * unique_ratio - 0.6 * filler_ratio + 0.2, 0.0, 1.0)


def signal_list_structure(ctx: ScoringContext) -> float:
    hits = sum(1 for pattern in LIST_PATTERNS if pattern.search(ctx.lowered))
    return clamp(saturate(hits, 0.8), 0.0, 1.0)


def signal_laughter(ctx: ScoringContext) -> float:
    hits = sum(len(pattern.findall(ctx.lowered)) for pattern in LAUGHTER_PATTERNS)
    return clamp(saturate(hits, 1.0), 0.0, 1.0)


def signal_topic_start(ctx: ScoringContext) -> float:
    """Does the window begin where a new thought begins?

    A clip that opens right after a pause reads as a deliberate start; one
    that opens the instant the previous sentence ended reads as a cut into the
    middle of something. Ending before a pause earns a smaller bonus for the
    same reason at the other end.
    """
    lead = saturate(ctx.candidate.lead_silence, 0.55)
    trail = saturate(ctx.candidate.trail_silence, 0.7)
    return clamp(0.7 * lead + 0.3 * trail, 0.0, 1.0)


def signal_duration_fit(ctx: ScoringContext) -> float:
    preset = ctx.preset
    width = max(4.0, (preset.max_duration - preset.min_duration) / 3.0)
    return clamp(bell(ctx.duration, preset.target_duration, width), 0.0, 1.0)


SIGNALS: Dict[str, Callable[[ScoringContext], float]] = {
    "hook": signal_hook,
    "curiosity": signal_curiosity,
    "payoff": signal_payoff,
    "emotion": signal_emotion,
    "quotability": signal_quotability,
    "completeness": signal_completeness,
    "audio_energy": signal_audio_energy,
    "pace": signal_pace,
    "information_density": signal_information_density,
    "list_structure": signal_list_structure,
    "laughter": signal_laughter,
    "duration_fit": signal_duration_fit,
    "topic_start": signal_topic_start,
}


# ---------------------------------------------------------------------------
# Penalties (multiplicative)
# ---------------------------------------------------------------------------


def penalty_dangling_start(ctx: ScoringContext) -> float:
    """A clip opening on "and so..." feels like it started by accident."""
    return 0.72 if ctx.tokens and ctx.tokens[0] in DANGLING_OPENERS else 1.0


def penalty_unfinished(ctx: ScoringContext) -> float:
    sentences = ctx.candidate.sentences
    return 1.0 if sentences and sentences[-1].terminal else 0.85


def penalty_filler(ctx: ScoringContext) -> float:
    if not ctx.tokens:
        return 1.0
    ratio = sum(1 for t in ctx.tokens if t in FILLER_WORDS) / len(ctx.tokens)
    return clamp(1.0 - 1.4 * max(0.0, ratio - 0.08), 0.55, 1.0)


def penalty_repetition(ctx: ScoringContext) -> float:
    """Punish windows where the speaker circles the same few words."""
    content = content_words(ctx.text)
    if len(content) < 12:
        return 1.0
    variety = len(set(content)) / len(content)
    return clamp(0.55 + 0.55 * variety, 0.6, 1.0)


def penalty_too_short_text(ctx: ScoringContext) -> float:
    """Long silence with almost no speech makes a dull short."""
    if ctx.duration <= 0:
        return 1.0
    return clamp(saturate(ctx.words_per_second, 0.7) + 0.25, 0.35, 1.0)


def penalty_intro_position(ctx: ScoringContext) -> float:
    """The first few seconds of a video are almost always housekeeping."""
    if ctx.source_duration <= 60:
        return 1.0
    if ctx.candidate.start < min(20.0, ctx.source_duration * 0.02):
        return 0.85
    return 1.0


PENALTIES: Dict[str, Callable[[ScoringContext], float]] = {
    "dangling_start": penalty_dangling_start,
    "unfinished": penalty_unfinished,
    "filler": penalty_filler,
    "repetition": penalty_repetition,
    "sparse_speech": penalty_too_short_text,
    "intro_position": penalty_intro_position,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context(
    candidate: Candidate,
    preset: PlatformPreset,
    *,
    energy: Optional[EnergyProfile] = None,
    source_duration: float = 0.0,
) -> ScoringContext:
    text = candidate.text
    tokens = tokenize(text)
    hook_tokens = tokens[:HOOK_WINDOW_WORDS]
    window = (
        energy.window(candidate.start, candidate.end)
        if energy is not None and energy.available
        else WindowEnergy()
    )
    return ScoringContext(
        candidate=candidate,
        preset=preset,
        energy=window,
        source_duration=source_duration,
        text=text,
        lowered=text.lower(),
        tokens=tokens,
        hook_text=" ".join(hook_tokens),
    )


def score_candidate(
    candidate: Candidate,
    preset: PlatformPreset,
    *,
    energy: Optional[EnergyProfile] = None,
    source_duration: float = 0.0,
) -> ScoreBreakdown:
    """Score one candidate, filling in and returning its breakdown."""
    ctx = build_context(candidate, preset, energy=energy, source_duration=source_duration)
    weights = preset.weights()

    signals = {name: clamp(fn(ctx), 0.0, 1.0) for name, fn in SIGNALS.items()}
    total_weight = sum(weights.get(name, 0.0) for name in signals) or 1.0
    weighted = sum(value * weights.get(name, 0.0) for name, value in signals.items())
    base = weighted / total_weight

    penalties = {name: fn(ctx) for name, fn in PENALTIES.items()}
    multiplier = 1.0
    for value in penalties.values():
        multiplier *= value

    breakdown = ScoreBreakdown(
        signals=signals,
        weights={k: weights.get(k, 0.0) for k in signals},
        penalties=penalties,
        total=clamp(base * multiplier * 100.0, 0.0, 100.0),
    )
    candidate.score = breakdown
    return breakdown


def score_candidates(
    candidates: Sequence[Candidate],
    preset: PlatformPreset,
    *,
    energy: Optional[EnergyProfile] = None,
    source_duration: float = 0.0,
) -> List[Candidate]:
    """Score every candidate and return them sorted best first."""
    for candidate in candidates:
        score_candidate(
            candidate, preset, energy=energy, source_duration=source_duration
        )
    return sorted(candidates, key=lambda c: c.score.total, reverse=True)


def explain(candidate: Candidate, limit: int = 4) -> str:
    """One-line, human readable reason a candidate scored the way it did."""
    breakdown = candidate.score
    if not breakdown.signals:
        return "not scored"
    parts = [
        f"{name} {breakdown.signals[name]:.2f}" for name in breakdown.top_signals[:limit]
    ]
    dragging = [
        f"{name} x{value:.2f}" for name, value in breakdown.penalties.items() if value < 0.95
    ]
    text = f"{breakdown.total:.1f}/100 - " + ", ".join(parts)
    if dragging:
        text += " | penalties: " + ", ".join(dragging)
    return text
