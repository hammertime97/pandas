import pytest

from clipper import utils


@pytest.mark.parametrize(
    "seconds, decimals, expected",
    [
        (0, 2, "00:00:00.00"),
        (123.456, 2, "00:02:03.46"),
        (119.7, 0, "00:02:00"),  # must carry, not render 00:01:60
        (3599.6, 0, "01:00:00"),
        (-5, 2, "00:00:00.00"),
    ],
)
def test_timecode(seconds, decimals, expected):
    assert utils.timecode(seconds, decimals) == expected


def test_ass_and_srt_timecodes():
    assert utils.ass_timecode(3661.5) == "1:01:01.50"
    assert utils.srt_timecode(3661.5) == "01:01:01,500"
    assert utils.srt_timecode(1.9999) == "00:00:02,000"


def test_slugify():
    assert utils.slugify("Why NOBODY talks about this!!") == "why-nobody-talks-about-this"
    assert utils.slugify("") == "clip"
    assert len(utils.slugify("a" * 200)) <= 60


def test_overlap():
    assert utils.overlap(0, 10, 5, 20) == 5
    assert utils.overlap(0, 5, 10, 20) == 0


def test_percentile():
    values = [1, 2, 3, 4]
    assert utils.percentile(values, 0.0) == 1
    assert utils.percentile(values, 1.0) == 4
    assert utils.percentile(values, 0.5) == pytest.approx(2.5)
    assert utils.percentile([], 0.5) == 0.0


def test_moving_average_preserves_length():
    values = [1, 5, 1, 5, 1]
    smoothed = utils.moving_average(values, 3)
    assert len(smoothed) == len(values)
    assert max(smoothed) < max(values)
    assert utils.moving_average(values, 1) == values


def test_clamp_handles_inverted_range():
    assert utils.clamp(5, 0, 10) == 5
    assert utils.clamp(-1, 0, 10) == 0
    assert utils.clamp(5, 10, 0) == 10  # degenerate range collapses to `low`


def test_short_hash_is_stable_and_scoped():
    assert utils.short_hash("a", 1) == utils.short_hash("a", 1)
    assert utils.short_hash("a", 1) != utils.short_hash("a", 2)
    assert len(utils.short_hash("x", length=8)) == 8


def test_unique_preserves_order():
    assert utils.unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
