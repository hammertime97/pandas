from clipper.captions import (
    clean_sentence,
    extract_keywords,
    extract_proper_nouns,
    generate_copy,
    make_hashtags,
    make_on_screen_hook,
    make_title,
    truncate,
)
from clipper.models import Candidate, Word
from clipper.segment import split_sentences

TEXT = (
    "So um, nobody tells you the biggest mistake founders make when raising money. "
    "You think a bigger valuation is always better for the company. "
    "It is not, and Sequoia partners will tell you the same thing. "
    "That is why the terms matter more than the headline number."
)


def candidate_from(text):
    words, cursor = [], 0.0
    for token in text.split():
        words.append(Word(token, cursor, cursor + 0.3))
        cursor += 0.35
    return Candidate(0.0, cursor, split_sentences(words))


def test_clean_sentence_strips_leading_filler_and_trailing_conjunctions():
    assert clean_sentence("So um, well anyway this is the point and") == "This is the point"
    assert clean_sentence("  ") == ""


def test_truncate_breaks_on_a_word():
    original = "one two three four five six seven"
    result = truncate(original, 20)
    assert len(result) <= 20
    assert result.endswith("...")
    # Every kept token must be a whole word from the original, never a fragment.
    kept = result[:-3].split()
    assert kept == original.split()[: len(kept)]
    assert truncate("short", 20) == "short"


def test_make_title_uses_the_hook():
    title = make_title(candidate_from(TEXT))
    assert title.startswith("Nobody tells you")
    assert len(title) <= 70
    assert not title.endswith(".")


def test_on_screen_hook_is_short():
    assert len(make_on_screen_hook(candidate_from(TEXT))) <= 42


def test_extract_proper_nouns_skips_sentence_starts():
    nouns = extract_proper_nouns(TEXT)
    assert "Sequoia" in nouns
    assert "So" not in nouns and "You" not in nouns


def test_extract_keywords_ignores_stopwords_and_short_words():
    keywords = extract_keywords(TEXT)
    assert "founders" in keywords or "valuation" in keywords
    assert all(len(k) >= 4 for k in keywords)
    assert "the" not in keywords


def test_hashtags_are_prefixed_unique_and_capped():
    tags = make_hashtags(TEXT, "tiktok", limit=6)
    assert len(tags) == 6
    assert len(set(tags)) == 6
    assert all(t.startswith("#") and " " not in t for t in tags)


def test_hashtags_include_platform_defaults():
    assert "#shorts" in make_hashtags("a short clip about nothing", "shorts", limit=8)


def test_generate_copy_is_complete():
    copy = generate_copy(candidate_from(TEXT), platform="tiktok", source_title="Podcast 12")
    assert copy.title and copy.caption and copy.hashtags and copy.on_screen_hook
    assert copy.title in copy.caption
    assert copy.hashtags[0] in copy.caption
