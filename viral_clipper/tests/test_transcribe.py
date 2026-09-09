"""Transcription backend selection, caching and fallbacks."""

import json

import pytest

from clipper.config import ClipperConfig
from clipper.errors import TranscriptionError
from clipper.models import Transcript, Word
from clipper.transcribe import (
    load_cached,
    save_cached,
    transcribe,
    transcript_from_captions,
)
from tests.conftest import needs_ffmpeg


def test_transcript_cache_round_trip(tmp_path):
    transcript = Transcript([Word("hi", 0.0, 0.3), Word("there", 0.3, 0.8)], source="test")
    path = tmp_path / "t.json"
    save_cached(path, transcript)
    restored = load_cached(path)
    assert restored.text == "hi there"
    assert restored.source == "test"
    assert restored.duration == pytest.approx(0.8)


def test_load_cached_tolerates_junk(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json at all", encoding="utf-8")
    assert load_cached(path) is None
    assert load_cached(tmp_path / "missing.json") is None
    path.write_text(json.dumps({"words": []}), encoding="utf-8")
    assert load_cached(path) is None, "an empty transcript is not a usable cache"


def test_transcript_from_captions_skips_unreadable_files(tmp_path):
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    good = tmp_path / "good.srt"
    good.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello world\n", encoding="utf-8")
    transcript = transcript_from_captions([empty, tmp_path / "nope.srt", good])
    assert transcript.text == "hello world"
    assert transcript.source == "captions:good.srt"


def test_transcript_from_captions_raises_when_nothing_works():
    with pytest.raises(TranscriptionError):
        transcript_from_captions([])


def test_words_between_filters_by_midpoint():
    transcript = Transcript([Word("a", 0, 1), Word("b", 5, 6), Word("c", 10, 11)])
    assert [w.text for w in transcript.words_between(4, 7)] == ["b"]


@needs_ffmpeg
def test_explicit_transcript_path_wins(sample_source, tmp_path):
    from clipper.ingest import resolve_source

    subtitles = tmp_path / "override.srt"
    subtitles.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\ncompletely different words here\n",
        encoding="utf-8",
    )
    config = ClipperConfig(workspace=tmp_path / "ws", transcript_path=subtitles)
    media = resolve_source(str(sample_source), config.workspace)
    transcript = transcribe(media, config)
    assert transcript.text == "completely different words here"


@needs_ffmpeg
def test_missing_transcript_path_raises(sample_source, tmp_path):
    from clipper.ingest import resolve_source

    config = ClipperConfig(workspace=tmp_path / "ws", transcript_path=tmp_path / "gone.srt")
    media = resolve_source(str(sample_source), config.workspace)
    with pytest.raises(TranscriptionError, match="not found"):
        transcribe(media, config)


@needs_ffmpeg
def test_falls_back_to_the_caption_sidecar(sample_source, tmp_path):
    """With no speech model installed, the sidecar must carry the run."""
    from clipper.ingest import resolve_source

    config = ClipperConfig(workspace=tmp_path / "ws")
    media = resolve_source(str(sample_source), config.workspace)
    transcript = transcribe(media, config, use_cache=False)
    assert transcript.words
    assert transcript.source.startswith(("captions:", "faster-whisper", "whisper"))


def test_a_backend_that_fails_with_a_network_error_falls_through(monkeypatch, tmp_path):
    """A speech model that cannot reach its model host must not end the job.

    faster-whisper downloads weights on first use; behind a proxy or offline
    that surfaces as an httpx/OSError, not a TranscriptionError.
    """
    import clipper.transcribe as module

    def unreachable(*args, **kwargs):
        raise OSError("403 Forbidden while fetching the model")

    monkeypatch.setattr(module, "transcribe_with_faster_whisper", unreachable)
    monkeypatch.setattr(module, "transcribe_with_openai_whisper", unreachable)

    subtitles = tmp_path / "talk.en.srt"
    subtitles.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nthe captions still work\n", encoding="utf-8"
    )

    class FakeMedia:
        path = tmp_path / "talk.wav"
        caption_files = [subtitles]

        class info:
            duration = 10.0

    FakeMedia.path.write_bytes(b"")
    monkeypatch.setattr(
        module.FFmpeg, "extract_audio", lambda self, source, dest, **kw: dest
    )
    (tmp_path / "ws" / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ws" / "audio" / "talk.wav").write_bytes(b"")

    config = ClipperConfig(workspace=tmp_path / "ws")
    transcript = module.transcribe(FakeMedia(), config, use_cache=False)
    assert transcript.text == "the captions still work"
    assert transcript.source.startswith("captions:")


def test_every_backend_failing_reports_all_of_them(monkeypatch, tmp_path):
    import clipper.transcribe as module

    monkeypatch.setattr(
        module, "transcribe_with_faster_whisper",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no model host")),
    )
    monkeypatch.setattr(
        module, "transcribe_with_openai_whisper",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no CUDA")),
    )

    class FakeMedia:
        path = tmp_path / "talk.wav"
        caption_files = []

        class info:
            duration = 10.0

    FakeMedia.path.write_bytes(b"")
    monkeypatch.setattr(
        module.FFmpeg, "extract_audio", lambda self, source, dest, **kw: dest
    )
    (tmp_path / "ws2" / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ws2" / "audio" / "talk.wav").write_bytes(b"")

    with pytest.raises(TranscriptionError) as caught:
        module.transcribe(FakeMedia(), ClipperConfig(workspace=tmp_path / "ws2"), use_cache=False)
    message = str(caught.value)
    assert "no model host" in message and "no CUDA" in message
    assert "--transcript" in message, "the error should say how to get unstuck"
