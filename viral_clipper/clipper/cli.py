"""Command line interface.

Uses argparse rather than a CLI framework so the tool has no hard
dependencies beyond the standard library and ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from clipper import __version__
from clipper.config import LAYOUTS, PRESETS, ClipperConfig
from clipper.errors import ClipperError
from clipper.subtitles import STYLE_PRESETS
from clipper.utils import log, timecode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description=(
            "Find the viral moments in a long video and cut them into vertical "
            "shorts for TikTok, Reels and YouTube Shorts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  clipper clip https://www.youtube.com/watch?v=... --platform tiktok\n"
            "  clipper clip talk.mp4 --max-clips 8 --layout blur\n"
            "  clipper clip talk.mp4 --transcript talk.srt --dry-run\n"
            "  clipper serve --port 8000\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"clipper {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command")

    clip = subparsers.add_parser("clip", help="cut shorts out of a video (default)")
    _add_clip_arguments(clip)

    serve = subparsers.add_parser("serve", help="run the web UI and JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workspace", type=Path, default=Path("workspace"))
    serve.add_argument("--workers", type=int, default=1, help="concurrent render jobs")

    transcribe = subparsers.add_parser("transcribe", help="only produce a transcript")
    transcribe.add_argument("source")
    transcribe.add_argument("--workspace", type=Path, default=Path("workspace"))
    transcribe.add_argument("--model", dest="whisper_model", default="small")
    transcribe.add_argument("--language", default=None)
    transcribe.add_argument("-o", "--output", type=Path, help="write SRT here")

    subparsers.add_parser("doctor", help="check that dependencies are installed")
    subparsers.add_parser("gui", help="open the desktop window")
    return parser


def _add_clip_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", help="a video URL or a local file path")
    parser.add_argument(
        "-p", "--platform", default="tiktok", choices=sorted(PRESETS),
        help="output preset (default: tiktok)",
    )
    parser.add_argument("-n", "--max-clips", type=int, default=5)
    parser.add_argument("-o", "--output", dest="output_dir", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--layout", default="auto", choices=LAYOUTS,
        help="auto follows the subject, center is a static crop, "
             "blur letterboxes over a blurred fill, fit uses black bars",
    )
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument(
        "--caption-style", default="punch", choices=sorted(STYLE_PRESETS),
    )
    parser.add_argument("--no-uppercase", action="store_true")
    parser.add_argument("--highlight-color", default="#FFE14D")
    parser.add_argument("--model", dest="whisper_model", default="small",
                        help="whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", default=None)
    parser.add_argument("--transcript", dest="transcript_path", type=Path,
                        help="use an existing .srt/.vtt/.json instead of transcribing")
    parser.add_argument("--source-height", dest="source_max_height", type=int, default=1080,
                        metavar="N",
                        help="tallest source to download (default 1080; try 2160 for "
                             "sharper vertical crops)")
    parser.add_argument("--fps", type=int, choices=range(15, 61), metavar="N",
                        help="output frame rate (default: the platform preset's 30)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip loudness normalisation")
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and print the moments without rendering")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    parser.add_argument("--llm", action="store_true",
                        help="re-rank the shortlist with Claude (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--llm-model", default="claude-opus-5")
    parser.add_argument("--llm-weight", type=float, default=0.5)


def config_from_args(args: argparse.Namespace) -> ClipperConfig:
    return ClipperConfig(
        platform=args.platform,
        workspace=args.workspace,
        output_dir=args.output_dir,
        max_clips=args.max_clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_score=args.min_score,
        layout=args.layout,
        burn_subtitles=not args.no_subtitles,
        caption_style=args.caption_style,
        uppercase_captions=not args.no_uppercase,
        highlight_color=args.highlight_color,
        whisper_model=args.whisper_model,
        language=args.language,
        transcript_path=args.transcript_path,
        source_max_height=args.source_max_height,
        fps=args.fps,
        normalize_audio=not args.no_normalize,
        ffmpeg_preset=args.ffmpeg_preset,
        dry_run=args.dry_run,
        use_llm=args.llm,
        llm_model=args.llm_model,
        llm_weight=args.llm_weight,
        verbose=args.verbose,
    ).validate()


def command_clip(args: argparse.Namespace) -> int:
    from clipper.pipeline import run_pipeline

    config = config_from_args(args)
    bar = _ProgressBar(enabled=not args.json and sys.stderr.isatty())
    result = run_pipeline(args.source, config, progress=bar)
    bar.finish()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print()
    print(f"Source : {result.source.title or result.source.path.name}")
    print(f"Output : {result.output_dir}")
    print(
        f"Found  : {result.stats['candidates']} candidate windows, "
        f"kept {len(result.clips)}  "
        f"(transcript: {result.stats['transcript_source']})"
    )
    print()
    for clip in result.clips:
        print(
            f"  [{clip.index}] {timecode(clip.start, 0)} - {timecode(clip.end, 0)}  "
            f"({clip.duration:.0f}s)   score {clip.score:.0f}/100"
        )
        print(f"      {clip.copy.title}")
        top = ", ".join(clip.breakdown.top_signals[:3])
        print(f"      strongest: {top}")
        if clip.video_path:
            print(f"      -> {Path(clip.video_path).name}")
        print()
    if config.dry_run:
        print("(dry run - nothing was rendered)")
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    from clipper.ffmpeg import FFmpeg
    from clipper.ingest import resolve_source
    from clipper.subtitle_io import words_to_srt
    from clipper.transcribe import transcribe as run_transcribe

    config = ClipperConfig(
        workspace=args.workspace,
        whisper_model=args.whisper_model,
        language=args.language,
    )
    ffmpeg = FFmpeg()
    media = resolve_source(args.source, args.workspace, ffmpeg)
    transcript = run_transcribe(media, config, ffmpeg)
    srt = words_to_srt(transcript.words)
    if args.output:
        args.output.write_text(srt, encoding="utf-8")
        print(f"{len(transcript.words)} words -> {args.output}", file=sys.stderr)
    else:
        print(srt)
    return 0


def command_gui(_: argparse.Namespace) -> int:
    from clipper.gui import launch

    return launch()


def command_doctor(_: argparse.Namespace) -> int:
    """Report which optional pieces are installed and what they unlock."""
    from clipper.ffmpeg import find_ffmpeg, find_ffprobe

    print(f"clipper {__version__}\n")
    ok = True

    ffmpeg = find_ffmpeg()
    print(_line("ffmpeg", bool(ffmpeg), ffmpeg or "REQUIRED - install ffmpeg"))
    ok = ok and bool(ffmpeg)
    ffprobe = find_ffprobe()
    print(_line("ffprobe", bool(ffprobe), ffprobe or "optional (falls back to ffmpeg -i)", required=False))

    checks = [
        ("yt-dlp", "yt_dlp", "needed to clip straight from a URL"),
        ("faster-whisper", "faster_whisper", "best transcription quality"),
        ("openai-whisper", "whisper", "alternative transcription backend"),
        ("numpy", "numpy", "faster subject tracking"),
        ("opencv", "cv2", "face-aware reframing"),
        ("fastapi", "fastapi", "needed for `clipper serve`"),
        ("uvicorn", "uvicorn", "needed for `clipper serve`"),
        ("anthropic", "anthropic", "needed for --llm re-ranking"),
    ]
    for label, module, why in checks:
        found = _module_available(module)
        print(_line(label, found, why, required=False))

    if not shutil.which("yt-dlp") and not _module_available("yt_dlp"):
        print("\nNote: without yt-dlp you can still clip local files.")
    if not _module_available("faster_whisper") and not _module_available("whisper"):
        print(
            "Note: without a speech model, pass --transcript with an .srt/.vtt "
            "file (or clip a source that already ships captions)."
        )
    print("\nInstall everything with:  pip install 'viral-clipper[full]'")
    return 0 if ok else 1


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _line(label: str, present: bool, detail: str, *, required: bool = True) -> str:
    mark = "OK  " if present else ("MISSING" if required else "  - ")
    return f"  [{mark}] {label:<16} {detail}"


def command_serve(args: argparse.Namespace) -> int:
    from clipper.server.app import serve

    serve(host=args.host, port=args.port, workspace=args.workspace, workers=args.workers)
    return 0


class _ProgressBar:
    """A single rewriting status line on stderr."""

    def __init__(self, enabled: bool = True, width: int = 28) -> None:
        self.enabled = enabled
        self.width = width
        self._last = ""

    def __call__(self, message: str, fraction: float) -> None:
        if not self.enabled:
            return
        filled = int(self.width * max(0.0, min(1.0, fraction)))
        bar = "#" * filled + "." * (self.width - filled)
        line = f"\r[{bar}] {fraction * 100:3.0f}%  {message[:44]:<44}"
        if line != self._last:
            sys.stderr.write(line)
            sys.stderr.flush()
            self._last = line

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


COMMANDS = {
    "clip": command_clip,
    "serve": command_serve,
    "transcribe": command_transcribe,
    "doctor": command_doctor,
    "gui": command_gui,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `clipper video.mp4` is shorthand for `clipper clip video.mp4`.
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "clip")

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:
        log.setLevel(logging.WARNING)

    try:
        return COMMANDS[args.command](args)
    except ClipperError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `clipper doctor | head` closes the pipe early. Point stdout at
        # devnull so the interpreter's own flush at exit does not report it
        # again, and use the conventional SIGPIPE exit status.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 128 + 13


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
