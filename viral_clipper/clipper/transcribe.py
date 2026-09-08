"""Produce word level timestamps for a source video.

Three backends, tried in order of timing quality:

1. ``faster-whisper``  - best word timings, needs a model download
2. ``openai-whisper``   - same idea, heavier runtime
3. caption files        - instant and free, timings interpolated per word

Results are cached as JSON in the workspace so re-running a job (or tweaking
scoring weights) never pays for transcription twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from clipper.config import ClipperConfig
from clipper.errors import TranscriptionError
from clipper.ffmpeg import FFmpeg
from clipper.ingest import SourceMedia
from clipper.models import Transcript, Word
from clipper.subtitle_io import load_words
from clipper.utils import ensure_dir, log, short_hash

ProgressFn = Callable[[str, float], None]

TRANSCRIBE_HINT = "pip install faster-whisper"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def transcribe_with_faster_whisper(
    audio_path: Path,
    *,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "default",
    language: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    total_duration: float = 0.0,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TranscriptionError("faster-whisper is not installed") from exc

    if compute_type == "default":
        compute_type = "int8" if device in ("auto", "cpu") else "float16"
    log.info("transcribing with faster-whisper (%s, %s)", model_size, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )

    duration = total_duration or float(getattr(info, "duration", 0.0) or 0.0)
    words: List[Word] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            text = (word.word or "").strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(word.start),
                    end=float(word.end),
                    confidence=float(getattr(word, "probability", 1.0) or 1.0),
                )
            )
        if progress and duration:
            progress("transcribing", min(0.99, float(segment.end) / duration))

    if not words:
        raise TranscriptionError("faster-whisper returned no words")
    return Transcript(
        words=words,
        language=str(getattr(info, "language", language or "en") or "en"),
        source="faster-whisper",
    )


def transcribe_with_openai_whisper(
    audio_path: Path,
    *,
    model_size: str = "small",
    language: Optional[str] = None,
) -> Transcript:
    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TranscriptionError("openai-whisper is not installed") from exc

    log.info("transcribing with openai-whisper (%s)", model_size)
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path), word_timestamps=True, language=language)

    words: List[Word] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(word["start"]),
                    end=float(word["end"]),
                    confidence=float(word.get("probability", 1.0)),
                )
            )
    if not words:
        raise TranscriptionError("openai-whisper returned no words")
    return Transcript(
        words=words, language=str(result.get("language", language or "en")), source="whisper"
    )


def transcript_from_captions(paths: Sequence[Path]) -> Transcript:
    """Build a transcript from the first caption file that yields words."""
    for path in paths:
        try:
            words = load_words(Path(path))
        except Exception as exc:
            log.debug("could not read captions from %s: %s", path, exc)
            continue
        if words:
            log.info("using captions from %s (%d words)", path, len(words))
            return Transcript(words=words, source=f"captions:{Path(path).name}")
    raise TranscriptionError("no usable caption file found")


# ---------------------------------------------------------------------------
# Caching + orchestration
# ---------------------------------------------------------------------------


def _cache_path(workspace: Path, media_path: Path, model_size: str) -> Path:
    stat = media_path.stat()
    key = short_hash(media_path.name, stat.st_size, model_size, length=16)
    return ensure_dir(Path(workspace) / "transcripts") / f"{key}.json"


def load_cached(path: Path) -> Optional[Transcript]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        transcript = Transcript.from_dict(data)
    except (ValueError, KeyError) as exc:
        log.debug("ignoring unreadable transcript cache %s: %s", path, exc)
        return None
    return transcript if transcript.words else None


def save_cached(path: Path, transcript: Transcript) -> None:
    path.write_text(json.dumps(transcript.to_dict()), encoding="utf-8")


def transcribe(
    media: SourceMedia,
    config: ClipperConfig,
    ffmpeg: Optional[FFmpeg] = None,
    *,
    progress: Optional[ProgressFn] = None,
    use_cache: bool = True,
) -> Transcript:
    """Get word timings for ``media``, trying every backend that is available."""
    ffmpeg = ffmpeg or FFmpeg()
    workspace = ensure_dir(Path(config.workspace))

    # 1. An explicitly supplied transcript always wins.
    if config.transcript_path:
        path = Path(config.transcript_path)
        if not path.exists():
            raise TranscriptionError(f"transcript file not found: {path}")
        if path.suffix.lower() == ".json":
            cached = load_cached(path)
            if cached:
                return cached
            raise TranscriptionError(f"{path} is not a valid transcript JSON file")
        return transcript_from_captions([path])

    cache_file = _cache_path(workspace, media.path, config.whisper_model)
    if use_cache:
        cached = load_cached(cache_file)
        if cached:
            log.info("reusing cached transcript (%d words)", len(cached.words))
            if progress:
                progress("transcript cached", 1.0)
            return cached

    audio_path = ensure_dir(workspace / "audio") / f"{media.path.stem}.wav"
    if not audio_path.exists():
        if progress:
            progress("extracting audio", 0.05)
        ffmpeg.extract_audio(media.path, audio_path)

    failures: List[str] = []
    transcript: Optional[Transcript] = None

    for attempt in (
        lambda: transcribe_with_faster_whisper(
            audio_path,
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            language=config.language,
            progress=progress,
            total_duration=media.info.duration,
        ),
        lambda: transcribe_with_openai_whisper(
            audio_path, model_size=config.whisper_model, language=config.language
        ),
        lambda: transcript_from_captions(media.caption_files),
    ):
        try:
            transcript = attempt()
            break
        except TranscriptionError as exc:
            failures.append(str(exc))
            continue

    if transcript is None:
        raise TranscriptionError(
            "could not transcribe the source. Tried: "
            + "; ".join(failures)
            + f". Install a speech model ({TRANSCRIBE_HINT}) or pass --transcript "
            "with an .srt/.vtt file."
        )

    transcript.words.sort(key=lambda w: w.start)
    if use_cache:
        save_cached(cache_file, transcript)
    if progress:
        progress("transcript ready", 1.0)
    log.info("transcript: %d words via %s", len(transcript.words), transcript.source)
    return transcript
