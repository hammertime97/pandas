import pytest

from clipper.models import Candidate, ScoreBreakdown, Sentence, Word
from clipper.select import jaccard, select_clips, temporal_overlap_ratio


def make(start, end, text, score):
    words = [Word(t, start, start + 0.2) for t in text.split()]
    candidate = Candidate(start, end, [Sentence(words, True)])
    candidate.score = ScoreBreakdown(total=score)
    return candidate


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_temporal_overlap_uses_the_shorter_clip():
    """A short clip fully inside a long one is the same moment, so 1.0."""
    short, long = make(10, 20, "x", 1), make(0, 60, "y", 1)
    assert temporal_overlap_ratio(short, long) == pytest.approx(1.0)
    assert temporal_overlap_ratio(make(0, 10, "x", 1), make(20, 30, "y", 1)) == 0.0


def test_selection_suppresses_overlapping_and_duplicate_moments():
    candidates = [
        make(0, 30, "alpha beta gamma delta epsilon", 80),
        make(2, 32, "alpha beta gamma delta epsilon", 78),    # same moment, shifted
        make(60, 90, "zeta eta theta iota kappa", 70),
        make(200, 230, "alpha beta gamma delta epsilon", 65),  # same words, elsewhere
        make(300, 330, "lambda mu nu xi omicron", 60),
    ]
    chosen = select_clips(candidates, max_clips=5)
    assert [c.start for c in chosen] == [0, 60, 300]


def test_selection_returns_clips_in_timeline_order():
    candidates = [
        make(300, 330, "lambda mu nu xi omicron", 90),
        make(0, 30, "alpha beta gamma delta epsilon", 80),
    ]
    assert [c.start for c in select_clips(candidates, max_clips=5)] == [0, 300]


def test_max_clips_is_honoured():
    candidates = [make(i * 100, i * 100 + 30, f"word{i} other{i} more{i}", 90 - i) for i in range(10)]
    assert len(select_clips(candidates, max_clips=3)) == 3


def test_min_score_filters():
    candidates = [make(0, 30, "a b c", 90), make(100, 130, "d e f", 10)]
    assert len(select_clips(candidates, max_clips=5, min_score=50)) == 1
    assert select_clips(candidates, max_clips=5, min_score=95) == []


def test_similarity_limit_can_be_relaxed():
    candidates = [
        make(0, 30, "alpha beta gamma delta epsilon", 80),
        make(200, 230, "alpha beta gamma delta epsilon", 70),
    ]
    assert len(select_clips(candidates, max_clips=5)) == 1
    assert len(select_clips(candidates, max_clips=5, similarity_limit=1.1)) == 2


def test_min_gap_spaces_clips_out():
    candidates = [make(0, 30, "a b c", 90), make(35, 65, "d e f", 80)]
    assert len(select_clips(candidates, max_clips=5)) == 2
    assert len(select_clips(candidates, max_clips=5, min_gap=30)) == 1


def test_empty_input():
    assert select_clips([], max_clips=5) == []
