import pytest

from clipper.audio import EnergyProfile, WindowEnergy
from clipper.config import PRESETS
from clipper.models import Word
from clipper.scoring import (
    PENALTIES,
    SIGNALS,
    bell,
    build_context,
    explain,
    saturate,
    score_candidate,
    score_candidates,
)
from clipper.segment import build_candidates, split_sentences

STRONG = (
    "Nobody tells you the biggest mistake founders make when they raise money. "
    "You think a higher valuation is always better for the company. "
    "It is not, and here is the thing nobody says out loud. "
    "That is why so many companies raise once and then quietly die. "
    "The terms matter more than the headline number, every single time. "
)
WEAK = (
    "and so um you know like basically we were just kind of going through the "
    "the the normal process and it was fine i guess and then like anyway yeah "
    "um so we sort of continued on with all of that and "
)


def candidate_from(text, repeat=3):
    words, cursor = [], 0.0
    for token in (text * repeat).split():
        words.append(Word(token, cursor, cursor + 0.31))
        cursor += 0.345
    sentences = split_sentences(words)
    return build_candidates(sentences, PRESETS["tiktok"])[0]


def test_saturate_and_bell():
    assert saturate(0, 1) == 0.0
    assert saturate(1, 1) == pytest.approx(0.5)
    assert 0 < saturate(100, 1) < 1
    assert bell(5, 5, 1) == pytest.approx(1.0)
    assert bell(9, 5, 1) < 0.01


def test_every_signal_is_bounded():
    ctx = build_context(candidate_from(STRONG), PRESETS["tiktok"], source_duration=600)
    for name, fn in SIGNALS.items():
        value = fn(ctx)
        assert 0.0 <= value <= 1.0, f"{name} returned {value}"


def test_every_penalty_is_a_multiplier():
    ctx = build_context(candidate_from(WEAK), PRESETS["tiktok"], source_duration=600)
    for name, fn in PENALTIES.items():
        value = fn(ctx)
        assert 0.0 < value <= 1.0, f"{name} returned {value}"


def test_strong_content_outscores_filler():
    strong = score_candidate(candidate_from(STRONG), PRESETS["tiktok"], source_duration=600)
    weak = score_candidate(candidate_from(WEAK), PRESETS["tiktok"], source_duration=600)
    assert strong.total > weak.total + 20
    assert 0 <= weak.total <= 100 and 0 <= strong.total <= 100


def test_hook_signal_matches_both_contraction_spellings():
    """Transcribers disagree on "here's" vs "here is"; both must score."""
    contracted = build_context(
        candidate_from("Here's why most people never get good at anything hard. " * 2, 4),
        PRESETS["tiktok"],
    )
    expanded = build_context(
        candidate_from("Here is why most people never get good at anything hard. " * 2, 4),
        PRESETS["tiktok"],
    )
    assert SIGNALS["hook"](contracted) == pytest.approx(SIGNALS["hook"](expanded))
    assert SIGNALS["hook"](expanded) > 0.5


def test_filler_penalty_bites_on_filler_heavy_text():
    weak = build_context(candidate_from(WEAK), PRESETS["tiktok"], source_duration=600)
    strong = build_context(candidate_from(STRONG), PRESETS["tiktok"], source_duration=600)
    assert PENALTIES["filler"](weak) < PENALTIES["filler"](strong)


def test_dangling_start_is_penalised():
    dangling = build_context(candidate_from("And so we continued with all of it. " * 2, 6), PRESETS["tiktok"])
    assert PENALTIES["dangling_start"](dangling) < 1.0


def test_topic_start_prefers_a_real_pause():
    quiet_start = candidate_from(STRONG)
    quiet_start.lead_silence = 2.5
    abrupt = candidate_from(STRONG)
    abrupt.lead_silence = 0.0
    ctx_quiet = build_context(quiet_start, PRESETS["tiktok"])
    ctx_abrupt = build_context(abrupt, PRESETS["tiktok"])
    assert SIGNALS["topic_start"](ctx_quiet) > SIGNALS["topic_start"](ctx_abrupt)


def test_audio_energy_falls_back_when_unavailable():
    ctx = build_context(candidate_from(STRONG), PRESETS["tiktok"], energy=EnergyProfile.empty())
    assert ctx.energy == WindowEnergy()
    assert 0.0 <= SIGNALS["audio_energy"](ctx) <= 1.0


def test_score_candidates_returns_sorted_and_annotates():
    candidates = [candidate_from(WEAK), candidate_from(STRONG)]
    ranked = score_candidates(candidates, PRESETS["tiktok"], source_duration=600)
    assert [c.score.total for c in ranked] == sorted(
        (c.score.total for c in ranked), reverse=True
    )
    assert all(c.score.signals for c in ranked)


def test_breakdown_top_signals_and_explain():
    candidate = candidate_from(STRONG)
    score_candidate(candidate, PRESETS["tiktok"], source_duration=600)
    top = candidate.score.top_signals
    assert set(top) <= set(SIGNALS)
    assert "hook" in top[:5]
    assert "/100" in explain(candidate)


def test_presets_weight_every_signal():
    for preset in PRESETS.values():
        assert set(SIGNALS) <= set(preset.weights())
