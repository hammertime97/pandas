"""Media layer tests. These need a real ffmpeg binary."""

import pytest

from clipper.errors import RenderError
from clipper.ffmpeg import FFmpeg, _parse_rate
from tests.conftest import needs_ffmpeg


@pytest.mark.parametrize(
    "value, expected",
    [("30000/1001", 29.97003), ("25", 25.0), ("0/0", None), (None, None), ("junk", None)],
)
def test_parse_rate(value, expected):
    result = _parse_rate(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-4)


@needs_ffmpeg
def test_probe_reads_dimensions_and_duration(sample_source):
    info = FFmpeg().probe(sample_source)
    assert info.width == 1280 and info.height == 720
    assert info.duration > 60
    assert info.has_audio and info.has_video
    assert info.aspect_ratio == pytest.approx(16 / 9, rel=1e-3)
    assert not info.is_vertical


@needs_ffmpeg
def test_probe_fallback_matches_ffprobe(sample_source):
    """The `ffmpeg -i` fallback must agree with ffprobe when both exist."""
    tool = FFmpeg()
    fallback = tool._probe_with_ffmpeg(sample_source)
    assert fallback.width == 1280 and fallback.height == 720
    if tool.ffprobe:
        primary = tool._probe_with_ffprobe(sample_source)
        assert primary.width == fallback.width
        assert primary.duration == pytest.approx(fallback.duration, abs=0.5)


@needs_ffmpeg
def test_probe_missing_file_raises():
    with pytest.raises(RenderError):
        FFmpeg().probe("/definitely/not/here.mp4")


@needs_ffmpeg
def test_energy_envelope_tracks_a_varying_source(sample_source):
    samples = FFmpeg().energy_envelope(sample_source, window=0.1)
    assert len(samples) > 100
    times = [t for t, _ in samples]
    assert times == sorted(times)
    levels = [lvl for _, lvl in samples]
    assert all(-90.0 <= lvl <= 0.0 for lvl in levels)
    # The fixture's volume swings, so the envelope must not be flat.
    assert max(levels) - min(levels) > 3.0


@needs_ffmpeg
def test_extract_audio_and_frame(sample_source, tmp_path):
    tool = FFmpeg()
    wav = tool.extract_audio(sample_source, tmp_path / "a.wav")
    assert wav.exists() and wav.stat().st_size > 1000
    assert tool.probe(wav).has_audio

    frame = tool.extract_frame(sample_source, 3.0, tmp_path / "f.jpg")
    assert frame.exists() and frame.stat().st_size > 500


@needs_ffmpeg
def test_read_gray_frames_returns_exact_buffer(sample_source):
    raw = FFmpeg().read_gray_frames(sample_source, 1.0, 2.0, 4, 32, 18)
    assert len(raw) % (32 * 18) == 0
    assert len(raw) // (32 * 18) >= 6
