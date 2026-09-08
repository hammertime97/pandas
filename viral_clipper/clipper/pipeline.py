"""End-to-end orchestration: a URL or file in, rendered shorts out."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from clipper import audio as audio_analysis
from clipper.captions import generate_copy
from clipper.config import ClipperConfig
from clipper.errors import NoMomentsFound
from clipper.ffmpeg import FFmpeg
from clipper.ingest import SourceMedia, resolve_source
from clipper.models import Clip, SocialCopy
from clipper.reframe import plan_crop
from clipper.render import RenderRequest, render_clip
from clipper.scoring import explain, score_candidates
from clipper.segment import build_candidates, pad_candidate, split_sentences
from clipper.select import select_clips
from clipper.subtitle_io import words_to_srt
from clipper.subtitles import build_ass, scale_font_size, write_ass
from clipper.transcribe import transcribe
from clipper.utils import ensure_dir, log, short_hash, slugify

ProgressFn = Callable[[str, float], None]

#: Fraction of total progress each stage is worth.
STAGE_WEIGHTS = {
    "source": 0.10,
    "transcribe": 0.32,
    "analyze": 0.08,
    "select": 0.05,
    "render": 0.43,
    "finish": 0.02,
}


class Progress:
    """Maps per-stage progress onto a single 0-1 job progress value."""

    def __init__(self, callback: Optional[ProgressFn] = None) -> None:
        self._callback = callback
        self._completed = 0.0
        self._stage = ""
        self._weight = 0.0

    def stage(self, name: str) -> None:
        self._completed += self._weight
        self._stage = name
        self._weight = STAGE_WEIGHTS.get(name, 0.05)
        self.update(name, 0.0)

    def update(self, message: str, fraction: float) -> None:
        if not self._callback:
            return
        overall = self._completed + self._weight * max(0.0, min(1.0, fraction))
        self._callback(message, min(1.0, overall))

    def sub(self) -> ProgressFn:
        """A callback that nested stages can pass down."""
        return self.update


@dataclass
class JobResult:
    """Everything a finished job produced."""

    source: SourceMedia
    clips: List[Clip]
    config: ClipperConfig
    output_dir: Path
    manifest_path: Optional[Path] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "config": self.config.to_dict(),
            "output_dir": str(self.output_dir),
            "stats": self.stats,
            "clips": [clip.to_dict() for clip in self.clips],
        }


def run_pipeline(
    source: str,
    config: ClipperConfig,
    *,
    progress: Optional[ProgressFn] = None,
    ffmpeg: Optional[FFmpeg] = None,
) -> JobResult:
    """Run the whole clipper for one source."""
    config.validate()
    preset = config.preset()
    ffmpeg = ffmpeg or FFmpeg()
    tracker = Progress(progress)
    started = time.monotonic()

    workspace = ensure_dir(Path(config.workspace))

    # 1. Get the media locally -------------------------------------------
    tracker.stage("source")
    media = resolve_source(source, workspace, ffmpeg, progress=tracker.sub())
    log.info("source: %s (%.1fs, %dx%d)", media.path.name, media.info.duration,
             media.info.width, media.info.height)

    # 2. Words with timings ----------------------------------------------
    tracker.stage("transcribe")
    transcript = transcribe(media, config, ffmpeg, progress=tracker.sub())

    # 3. Sentences, candidates, audio energy -----------------------------
    tracker.stage("analyze")
    sentences = split_sentences(transcript.words)
    tracker.update("finding candidate moments", 0.3)
    candidates = build_candidates(sentences, preset)
    if not candidates:
        raise NoMomentsFound(
            f"no {preset.min_duration:.0f}-{preset.max_duration:.0f}s window of "
            "connected speech was found. Try widening --min-duration/--max-duration."
        )
    log.info("%d sentences -> %d candidate windows", len(sentences), len(candidates))

    tracker.update("measuring audio energy", 0.6)
    energy = audio_analysis.analyze(media.path, ffmpeg)

    # 4. Score and choose -------------------------------------------------
    tracker.stage("select")
    ranked = score_candidates(
        candidates, preset, energy=energy, source_duration=media.info.duration
    )

    llm_copy: Dict[int, SocialCopy] = {}
    if config.use_llm:
        from clipper import llm

        tracker.update("asking Claude to re-rank", 0.4)
        llm_copy = llm.rerank(
            ranked,
            model=config.llm_model,
            weight=config.llm_weight,
            limit=config.llm_candidates,
            source_title=media.title,
        )
        ranked = sorted(ranked, key=lambda c: c.score.total, reverse=True)

    chosen = select_clips(
        ranked,
        max_clips=config.max_clips,
        max_overlap=config.max_overlap,
        min_score=config.min_score,
        similarity_limit=config.content_similarity_limit,
    )
    if not chosen:
        raise NoMomentsFound(
            "every candidate scored below --min-score; lower it to see the best available."
        )
    for candidate in chosen:
        log.info("picked %.1fs-%.1fs  %s", candidate.start, candidate.end, explain(candidate))

    # 5. Render -----------------------------------------------------------
    tracker.stage("render")
    job_dir = ensure_dir(
        config.resolved_output_dir() / f"{slugify(media.title or media.path.stem)}-{short_hash(media.path.name, length=6)}"
    )
    clips: List[Clip] = []
    llm_copy_by_id = {id(ranked[i]): copy for i, copy in llm_copy.items() if i < len(ranked)}

    for index, candidate in enumerate(chosen, start=1):
        tracker.update(f"rendering clip {index}/{len(chosen)}", (index - 1) / len(chosen))
        clip = _render_one(
            ffmpeg=ffmpeg,
            media=media,
            candidate=candidate,
            index=index,
            config=config,
            job_dir=job_dir,
            llm_copy=llm_copy_by_id.get(id(candidate)),
        )
        clips.append(clip)
        tracker.update(f"rendered clip {index}/{len(chosen)}", index / len(chosen))

    # 6. Manifest ---------------------------------------------------------
    tracker.stage("finish")
    elapsed = time.monotonic() - started
    result = JobResult(
        source=media,
        clips=clips,
        config=config,
        output_dir=job_dir,
        stats={
            "sentences": len(sentences),
            "candidates": len(candidates),
            "selected": len(clips),
            "transcript_source": transcript.source,
            "transcript_words": len(transcript.words),
            "audio_analysis": energy.available,
            "elapsed_seconds": round(elapsed, 1),
            "source_duration": round(media.info.duration, 1),
        },
    )
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    result.manifest_path = manifest_path
    tracker.update("done", 1.0)
    log.info("%d clips in %.1fs -> %s", len(clips), elapsed, job_dir)
    return result


def _render_one(
    *,
    ffmpeg: FFmpeg,
    media: SourceMedia,
    candidate,
    index: int,
    config: ClipperConfig,
    job_dir: Path,
    llm_copy: Optional[SocialCopy] = None,
) -> Clip:
    """Plan, caption and render a single clip."""
    preset = config.preset()
    padded = pad_candidate(candidate, media.info.duration)

    copy = generate_copy(
        candidate,
        platform=preset.name,
        source_title=media.title,
        max_hashtags=preset.max_hashtags,
    )
    if llm_copy:
        # Keep the heuristic caption and hashtags, take the model's headline.
        copy.title = llm_copy.title or copy.title
        copy.on_screen_hook = llm_copy.on_screen_hook or copy.on_screen_hook

    stem = f"clip-{index:02d}-{slugify(copy.title, 40)}"
    clip_id = short_hash(media.path.name, padded.start, padded.end, length=8)

    # Word timings for the burned-in captions, clipped to the padded window.
    words = [
        w for w in candidate.words if padded.start <= (w.start + w.end) / 2 <= padded.end
    ]

    subtitle_path: Optional[Path] = None
    if config.burn_subtitles and words:
        ass = build_ass(
            words,
            width=preset.width,
            height=preset.height,
            font_size=scale_font_size(preset.caption_font_size, preset.width),
            margin_v=preset.caption_margin_v,
            style=config.caption_style,
            uppercase=config.uppercase_captions,
            highlight_colour=config.highlight_color,
            max_words=preset.caption_max_words,
            offset=padded.start,
        )
        subtitle_path = write_ass(ass, job_dir / f"{stem}.ass")
        (job_dir / f"{stem}.srt").write_text(
            words_to_srt([_shifted(w, padded.start) for w in words]), encoding="utf-8"
        )

    plan = None
    if config.layout in ("auto", "center") and media.info.width and media.info.height:
        plan = plan_crop(
            ffmpeg,
            media.path,
            start=padded.start,
            duration=padded.duration,
            source_w=media.info.width,
            source_h=media.info.height,
            target_ratio=preset.aspect_ratio,
            layout=config.layout,
            sample_fps=config.tracking_fps,
            alpha=config.tracking_smoothing,
            deadzone=config.tracking_deadzone,
        )

    clip = Clip(
        clip_id=clip_id,
        index=index,
        start=padded.start,
        end=padded.end,
        score=candidate.score.total,
        breakdown=candidate.score,
        transcript_text=candidate.text,
        hook=candidate.hook,
        copy=copy,
        words=words,
        platform=preset.name,
        width=preset.width,
        height=preset.height,
        subtitle_path=str(subtitle_path) if subtitle_path else None,
    )

    if config.dry_run:
        return clip

    output = job_dir / f"{stem}.mp4"
    request = RenderRequest(
        source=media.path,
        output=output,
        start=padded.start,
        end=padded.end,
        preset=preset,
        plan=plan,
        layout=config.layout,
        subtitle_path=subtitle_path,
        sendcmd_path=job_dir / f"{stem}.cmds.txt",
        normalize_audio=config.normalize_audio,
        ffmpeg_preset=config.ffmpeg_preset,
    )
    render_clip(ffmpeg, request, has_audio=media.info.has_audio)
    clip.video_path = str(output)

    if config.thumbnails:
        thumbnail = job_dir / f"{stem}.jpg"
        try:
            ffmpeg.extract_frame(output, padded.duration * 0.25, thumbnail)
            clip.thumbnail_path = str(thumbnail)
        except Exception as exc:  # a missing thumbnail must not fail the job
            log.debug("thumbnail failed for %s: %s", stem, exc)

    if not config.keep_intermediates:
        for leftover in (job_dir / f"{stem}.cmds.txt",):
            if leftover.exists():
                leftover.unlink()

    return clip


def _shifted(word, offset: float):
    """Copy a word with its timings rebased to the start of the clip."""
    from clipper.models import Word

    return Word(
        text=word.text,
        start=max(0.0, word.start - offset),
        end=max(0.0, word.end - offset),
        confidence=word.confidence,
    )
