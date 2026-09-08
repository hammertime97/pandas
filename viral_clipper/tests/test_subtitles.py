import pytest

from clipper.models import Word
from clipper.subtitles import (
    STYLE_PRESETS,
    build_ass,
    escape_ass,
    group_words,
    hex_to_ass_colour,
    scale_font_size,
)


def words_from(text, start=0.0, step=0.35):
    out, cursor = [], start
    for token in text.split():
        out.append(Word(token, cursor, cursor + step * 0.9))
        cursor += step
    return out


def dialogues(ass):
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


@pytest.mark.parametrize(
    "value, expected",
    [("#FFE14D", "&H004DE1FF"), ("#000000", "&H00000000"), ("#FFF", "&H00FFFFFF")],
)
def test_hex_to_ass_colour(value, expected):
    assert hex_to_ass_colour(value) == expected


def test_hex_to_ass_colour_rejects_garbage():
    with pytest.raises(ValueError):
        hex_to_ass_colour("not-a-colour")


def test_escape_ass_neutralises_markup():
    assert escape_ass("a{b}c\\d") == "a\\{b\\}c\\\\d"
    assert "\n" not in escape_ass("line\nbreak")


def test_group_words_respects_the_word_cap():
    chunks = group_words(words_from("one two three four five six seven eight"), max_words=3)
    assert all(len(c.words) <= 3 for c in chunks)
    assert sum(len(c.words) for c in chunks) == 8


def test_group_words_breaks_on_sentence_end():
    chunks = group_words(words_from("hello there. next line here"), max_words=6)
    assert chunks[0].text == "hello there."


def test_group_words_breaks_on_a_pause():
    words = words_from("one two") + words_from("three four", start=5.0)
    chunks = group_words(words, max_words=8, max_gap=0.5)
    assert len(chunks) == 2


def test_build_ass_structure():
    ass = build_ass(words_from("hello there friend"), width=1080, height=1920)
    assert "[Script Info]" in ass and "[V4+ Styles]" in ass and "[Events]" in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert dialogues(ass)


def test_build_ass_emits_one_event_per_word_and_tiles_them():
    """Events must butt up against each other or the caption blinks."""
    words = words_from("alpha bravo charlie")
    lines = dialogues(build_ass(words, max_words=4))
    assert len(lines) == 3
    starts_ends = [line.split(",")[1:3] for line in lines]
    for (_, end), (next_start, _) in zip(starts_ends, starts_ends[1:]):
        assert end == next_start


def test_build_ass_highlights_exactly_one_word_per_event():
    lines = dialogues(build_ass(words_from("alpha bravo charlie"), max_words=4))
    for index, line in enumerate(lines):
        assert line.count("\\fscx") == 1
        highlighted = line.split("\\fscx")[1].split("}")[1].split("{")[0]
        assert highlighted == ["ALPHA", "BRAVO", "CHARLIE"][index]


def test_offset_rebases_timings_to_the_clip():
    ass = build_ass(words_from("alpha bravo", start=100.0), offset=100.0)
    assert dialogues(ass)[0].split(",")[1] == "0:00:00.00"


def test_uppercase_can_be_disabled():
    assert "alpha" in build_ass(words_from("alpha bravo"), uppercase=False)
    assert "ALPHA" in build_ass(words_from("alpha bravo"), uppercase=True)


def test_minimal_style_emits_one_event_per_chunk():
    lines = dialogues(build_ass(words_from("alpha bravo charlie"), style="minimal", max_words=4))
    assert len(lines) == 1
    assert "\\fscx" not in lines[0]


@pytest.mark.parametrize("style", sorted(STYLE_PRESETS))
def test_every_style_renders(style):
    assert dialogues(build_ass(words_from("alpha bravo charlie"), style=style))


def test_scale_font_size():
    assert scale_font_size(78, 1080) == 78
    assert scale_font_size(78, 2160) == 156
    assert scale_font_size(78, 540) == 39


def test_no_events_for_empty_input():
    assert dialogues(build_ass([])) == []
