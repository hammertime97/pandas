import pytest

from clipper import subtitle_io
from clipper.models import Word

SRT = """1
00:00:01,000 --> 00:00:03,500
Here is the <b>first</b> line.

2
00:00:03,600 --> 00:00:06,000
And the second one?
"""

VTT_ROLLING = """WEBVTT

00:00:00.500 --> 00:00:02.000
hello there

00:00:02.000 --> 00:00:04.000
hello there friends

00:00:04.000 --> 00:00:06.000
completely new line
"""


def test_parse_timestamp_accepts_both_separators():
    assert subtitle_io.parse_timestamp("00:00:03,500") == pytest.approx(3.5)
    assert subtitle_io.parse_timestamp("00:00:03.500") == pytest.approx(3.5)
    with pytest.raises(ValueError):
        subtitle_io.parse_timestamp("nope")


def test_parse_srt_strips_markup():
    cues = subtitle_io.parse_cues(SRT)
    assert len(cues) == 2
    assert cues[0].text == "Here is the first line."
    assert cues[1].start == pytest.approx(3.6)


def test_rolling_captions_are_deduplicated():
    """YouTube auto-captions repeat the previous line; keep one merged cue."""
    cues = subtitle_io.parse_cues(VTT_ROLLING)
    assert [c.text for c in cues] == ["hello there friends", "completely new line"]
    assert cues[0].start == pytest.approx(0.5)
    assert cues[0].end == pytest.approx(4.0)


def test_cues_to_words_allocates_time_by_word_length():
    cues = subtitle_io.parse_cues(SRT)
    words = subtitle_io.cues_to_words(cues)
    assert [w.text for w in words[:3]] == ["Here", "is", "the"]
    assert words[0].start == pytest.approx(1.0)
    assert words[-1].end <= 6.0
    # Words are ordered and non-degenerate.
    assert all(w.end > w.start for w in words)
    assert words == sorted(words, key=lambda w: w.start)
    # A long word gets more time than a short one in the same cue.
    here, is_ = words[0], words[1]
    assert here.duration > is_.duration


def test_words_to_srt_breaks_on_sentence_end():
    words = [Word(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate("one two three. four".split())]
    srt = subtitle_io.words_to_srt(words)
    assert "one two three." in srt
    assert srt.count("-->") == 2


def test_load_words_roundtrip(tmp_path):
    path = tmp_path / "a.srt"
    path.write_text(SRT, encoding="utf-8")
    assert len(subtitle_io.load_words(path)) == 9
