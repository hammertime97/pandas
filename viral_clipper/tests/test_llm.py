"""LLM re-ranking, exercised without touching the network."""

import pytest

from clipper.llm import (
    LLMRating,
    LLMUnavailable,
    apply_ratings,
    build_prompt,
    parse_ratings,
    rerank,
)
from clipper.models import Candidate, ScoreBreakdown, Sentence, Word


def make(start=0.0, end=30.0, text="alpha beta gamma", score=60.0):
    words = [Word(t, start, start + 0.2) for t in text.split()]
    candidate = Candidate(start, end, [Sentence(words, True)])
    candidate.score = ScoreBreakdown(total=score)
    return candidate


def test_build_prompt_numbers_every_candidate():
    prompt = build_prompt([make(0, 30), make(60, 90)], source_title="Ep 12")
    assert "Ep 12" in prompt and "[0]" in prompt and "[1]" in prompt
    assert "Rate all 2 candidates" in prompt


def test_build_prompt_truncates_long_transcripts():
    prompt = build_prompt([make(text="word " * 2000)])
    assert len(prompt) < 2000
    assert prompt.rstrip().endswith("...")


def test_parse_ratings_keeps_valid_entries_only():
    ratings = parse_ratings(
        '{"ratings": ['
        '{"id": 0, "score": 90, "title": "T", "on_screen_hook": "H", "reason": "r"},'
        '{"id": 9, "score": 10, "title": "x", "on_screen_hook": "y", "reason": "z"},'
        '{"id": "bad", "score": 50},'
        '{"id": 1, "score": 250, "title": "", "on_screen_hook": "", "reason": ""}]}',
        count=2,
    )
    assert [r.index for r in ratings] == [0, 1]
    assert ratings[0].score == 90.0
    assert ratings[1].score == 100.0, "scores are clamped into 0-100"


def test_parse_ratings_rejects_non_json():
    with pytest.raises(LLMUnavailable):
        parse_ratings("sorry, I cannot do that", count=1)


def test_apply_ratings_blends_scores():
    candidates = [make(score=60.0), make(score=60.0)]
    apply_ratings(candidates, [LLMRating(index=0, score=90.0)], weight=0.5)
    assert candidates[0].score.total == pytest.approx(75.0)
    assert candidates[1].score.total == 60.0, "unrated candidates keep their score"


@pytest.mark.parametrize("weight, expected", [(0.0, 60.0), (1.0, 90.0), (0.25, 67.5)])
def test_apply_ratings_weighting(weight, expected):
    candidates = [make(score=60.0)]
    apply_ratings(candidates, [LLMRating(index=0, score=90.0)], weight=weight)
    assert candidates[0].score.total == pytest.approx(expected)


def test_apply_ratings_returns_copy_and_records_the_signal():
    candidates = [make()]
    copy = apply_ratings(
        candidates, [LLMRating(index=0, score=80.0, title="A title", on_screen_hook="Hook")]
    )
    assert copy[0].title == "A title" and copy[0].on_screen_hook == "Hook"
    assert candidates[0].score.signals["llm"] == pytest.approx(0.8)
    # Weight 0 keeps it out of the heuristic total, which already blended it in.
    assert candidates[0].score.weights["llm"] == 0.0


def test_apply_ratings_truncates_overlong_copy():
    candidates = [make()]
    copy = apply_ratings(
        candidates, [LLMRating(index=0, score=80.0, title="t" * 200, on_screen_hook="h" * 200)]
    )
    assert len(copy[0].title) == 70 and len(copy[0].on_screen_hook) == 42


def test_rerank_degrades_gracefully(monkeypatch):
    """A missing SDK or key must never fail the job."""
    def boom(*args, **kwargs):
        raise LLMUnavailable("no key")

    monkeypatch.setattr("clipper.llm.rate_candidates", boom)
    candidates = [make(score=61.0)]
    assert rerank(candidates) == {}
    assert candidates[0].score.total == 61.0


def test_rerank_with_no_candidates():
    assert rerank([]) == {}
