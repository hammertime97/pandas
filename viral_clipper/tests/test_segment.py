import pytest

from clipper.config import PRESETS
from clipper.models import Word
from clipper.segment import (
    build_candidates,
    is_terminal_token,
    pad_candidate,
    split_sentences,
)


def words_from(text, wps=2.9, start=0.0):
    out, cursor, step = [], start, 1.0 / wps
    for token in text.split():
        out.append(Word(token, cursor, cursor + step * 0.9))
        cursor += step
    return out


@pytest.mark.parametrize(
    "token, expected",
    [("end.", True), ("yes!", True), ("what?", True), ("word", False),
     ("Dr.", False), ("e.g.", False), ("J.", False), ("3.", False),
     ('done."', True)],
)
def test_is_terminal_token(token, expected):
    assert is_terminal_token(token) is expected


def test_split_sentences_on_punctuation():
    sentences = split_sentences(words_from("One two three. Four five six! Seven eight nine?"))
    assert len(sentences) == 3
    assert sentences[0].terminal is True
    assert sentences[0].text == "One two three."


def test_split_sentences_on_long_pause():
    """Caption sources often strip punctuation; a pause has to break the run."""
    words = words_from("one two three four five")
    words += words_from("six seven eight nine", start=words[-1].end + 2.0)
    sentences = split_sentences(words, max_pause=0.7)
    assert len(sentences) == 2
    assert sentences[0].terminal is False


def test_split_sentences_caps_runaway_sentences():
    sentences = split_sentences(words_from(" ".join(["word"] * 200)), max_words=42)
    assert len(sentences) >= 4
    assert all(len(s.words) <= 42 for s in sentences)


def test_build_candidates_respects_duration_window():
    preset = PRESETS["tiktok"]
    sentences = split_sentences(words_from(("Alpha beta gamma delta epsilon zeta. " * 40)))
    candidates = build_candidates(sentences, preset)
    assert candidates
    for candidate in candidates:
        assert preset.min_duration <= candidate.duration <= preset.max_duration
        # Boundaries always land on sentence edges.
        assert candidate.start == candidate.sentences[0].start
        assert candidate.end == candidate.sentences[-1].end


def test_build_candidates_records_lead_silence():
    words = words_from("Alpha beta gamma delta. " * 12)
    tail = words_from("Brand new topic entirely here. " * 12, start=words[-1].end + 3.0)
    candidates = build_candidates(split_sentences(words + tail), PRESETS["tiktok"])
    after_gap = [c for c in candidates if c.lead_silence > 2.0]
    assert after_gap, "a candidate starting after the 3s gap should record it"


def test_build_candidates_does_not_span_long_gaps():
    words = words_from("Alpha beta gamma delta epsilon. " * 8)
    tail = words_from("Zeta eta theta iota kappa. " * 8, start=words[-1].end + 30.0)
    candidates = build_candidates(split_sentences(words + tail), PRESETS["tiktok"], max_gap=2.5)
    assert all(c.duration <= PRESETS["tiktok"].max_duration for c in candidates)
    # No candidate straddles the 30s hole.
    assert not any(c.start < words[-1].end < c.end for c in candidates)


def test_build_candidates_caps_the_list():
    sentences = split_sentences(words_from("Alpha beta gamma delta epsilon zeta. " * 300))
    candidates = build_candidates(sentences, PRESETS["tiktok"], max_candidates=25)
    assert len(candidates) == 25


def test_pad_candidate_stays_inside_the_source():
    sentences = split_sentences(words_from("Alpha beta gamma delta epsilon. " * 12, start=5.0))
    # Not the first candidate: it starts at t=0, where there is no room to pad.
    candidate = build_candidates(sentences, PRESETS["tiktok"])[1]
    padded = pad_candidate(candidate, source_duration=candidate.end + 0.1)
    assert padded.start < candidate.start
    assert padded.start >= 0
    assert padded.end <= candidate.end + 0.1
    assert padded.sentences == candidate.sentences


def test_pad_candidate_cannot_go_negative():
    sentences = split_sentences(words_from("Alpha beta gamma delta epsilon. " * 12))
    candidate = build_candidates(sentences, PRESETS["tiktok"])[0]
    assert pad_candidate(candidate, 999.0).start >= 0.0
