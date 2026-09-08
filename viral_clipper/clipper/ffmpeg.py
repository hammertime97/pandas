"""Thin, dependency-free wrapper around the ffmpeg toolchain.

The only hard external requirement of this project is an ``ffmpeg`` binary.
``ffprobe`` is used when present, but every probe has a fallback that parses
``ffmpeg -i`` output so the pipeline still runs on minimal installs (for
example the static build shipped by ``imageio-ffmpeg``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from clipper.errors import DependencyMissing, RenderError
from clipper.models import MediaInfo
from clipper.utils import log, run_command

INSTALL_HINT = "apt install ffmpeg / brew install ffmpeg / pip install imageio-ffmpeg"

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}\.\d+)")
_VIDEO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: ")
# `ametadata=print` emits alternating "frame:.. pts_time:.." / "key=value" lines.
_PTS_TIME_RE = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")
_RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+(?:\.\d+)?|-inf|nan)")


def _from_imageio() -> Optional[str]:
    """The ``imageio-ffmpeg`` wheel bundles a static ffmpeg; use it if present."""
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        return None


def find_ffmpeg() -> Optional[str]:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or _from_imageio()


def find_ffprobe() -> Optional[str]:
    explicit = os.environ.get("FFPROBE_BINARY")
    if explicit:
        return explicit
    found = shutil.which("ffprobe")
    if found:
        return found
    # Static bundles sometimes drop ffprobe next to ffmpeg.
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
    return None


class FFmpeg:
    """Locates the binaries once and exposes the handful of calls we need."""

    def __init__(self, ffmpeg: Optional[str] = None, ffprobe: Optional[str] = None) -> None:
        self.ffmpeg = ffmpeg or find_ffmpeg()
        if not self.ffmpeg:
            raise DependencyMissing("ffmpeg", INSTALL_HINT)
        self.ffprobe = ffprobe or find_ffprobe()

    # -- basic plumbing ---------------------------------------------------

    def version(self) -> str:
        proc = run_command([self.ffmpeg, "-version"], check=False)
        first = (proc.stdout or "").splitlines()
        return first[0] if first else "unknown"

    def run(self, args: Sequence[str], *, timeout: Optional[float] = None) -> str:
        """Run ffmpeg with sane defaults, raising :class:`RenderError`."""
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-y", *[str(a) for a in args]]
        try:
            proc = run_command(cmd, check=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            raise RenderError(f"ffmpeg failed:\n{exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RenderError(f"ffmpeg timed out after {timeout}s") from exc
        return proc.stderr or ""

    # -- probing ----------------------------------------------------------

    def probe(self, path: Path) -> MediaInfo:
        """Describe a media file, preferring ffprobe and falling back to ffmpeg."""
        path = Path(path)
        if not path.exists():
            raise RenderError(f"media file not found: {path}")
        if self.ffprobe:
            try:
                return self._probe_with_ffprobe(path)
            except Exception as exc:  # pragma: no cover - depends on install
                log.debug("ffprobe failed (%s), falling back to ffmpeg -i", exc)
        return self._probe_with_ffmpeg(path)

    def _probe_with_ffprobe(self, path: Path) -> MediaInfo:
        proc = run_command(
            [
                self.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration") or 0.0)
        if not duration and video:
            duration = float(video.get("duration") or 0.0)
        width = int(video.get("width", 0)) if video else 0
        height = int(video.get("height", 0)) if video else 0
        fps = _parse_rate(video.get("avg_frame_rate") if video else None) or 30.0
        title = str(data.get("format", {}).get("tags", {}).get("title", ""))
        return MediaInfo(
            path=str(path),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            has_audio=audio is not None,
            has_video=video is not None,
            title=title,
        )

    def _probe_with_ffmpeg(self, path: Path) -> MediaInfo:
        # `ffmpeg -i FILE` with no output exits non-zero but still prints the
        # stream table we want, so failures here are expected.
        proc = run_command(
            [self.ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)], check=False
        )
        text = proc.stderr or ""
        duration = 0.0
        match = _DURATION_RE.search(text)
        if match:
            hours, minutes, seconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        width = height = 0
        fps = 30.0
        has_video = False
        for line in text.splitlines():
            if ": Video: " not in line:
                continue
            has_video = True
            size = _VIDEO_STREAM_RE.search(line)
            if size:
                width, height = int(size.group(1)), int(size.group(2))
            rate = _FPS_RE.search(line)
            if rate:
                fps = float(rate.group(1))
            break

        has_audio = bool(_AUDIO_STREAM_RE.search(text))
        if not duration and not has_video and not has_audio:
            raise RenderError(f"could not read media info for {path}")
        return MediaInfo(
            path=str(path),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            has_audio=has_audio,
            has_video=has_video,
        )

    # -- derived media ----------------------------------------------------

    def extract_audio(
        self, source: Path, destination: Path, *, sample_rate: int = 16000, mono: bool = True
    ) -> Path:
        """Write a WAV suitable for speech recognition."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1" if mono else "2",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(destination),
            ]
        )
        return destination

    def energy_envelope(
        self, source: Path, *, window: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Per-window RMS level in dBFS, as ``(timestamp, level)`` pairs.

        This drives the "is something actually happening here" signal: laughter,
        emphasis and applause all show up as sharp excursions in the envelope.

        ``astats`` is used rather than ``ebur128`` because it writes to stdout
        via ``ametadata=print``, which makes parsing independent of ffmpeg's
        log level.  Silence is reported as ``-inf`` and clamped to a floor so
        downstream maths stays finite.
        """
        rate = 44100
        samples_per_window = max(1, int(round(rate * max(0.01, window))))
        graph = (
            f"[0:a]aresample={rate},asetnsamples=n={samples_per_window}:p=0,"
            "astats=metadata=1:reset=1,"
            "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-[out]"
        )
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-v",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and not proc.stdout:
            log.debug("energy analysis unavailable: %s", (proc.stderr or "")[-300:])
            return []

        samples: List[Tuple[float, float]] = []
        pending: Optional[float] = None
        for line in (proc.stdout or "").splitlines():
            time_match = _PTS_TIME_RE.search(line)
            if time_match:
                pending = float(time_match.group(1))
                continue
            rms_match = _RMS_RE.search(line)
            if rms_match and pending is not None:
                raw = rms_match.group(1)
                level = -90.0 if raw in ("-inf", "nan") else float(raw)
                samples.append((pending, max(-90.0, level)))
                pending = None
        return samples

    def extract_frame(self, source: Path, timestamp: float, destination: Path) -> Path:
        """Grab a single still, used for clip thumbnails."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                "-ss",
                f"{max(0.0, timestamp):.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(destination),
            ]
        )
        return destination

    def read_gray_frames(
        self, source: Path, start: float, duration: float, fps: float, width: int, height: int
    ) -> bytes:
        """Decode a window into raw 8-bit grayscale frames of ``width x height``.

        Used by the reframing tracker; returning bytes keeps this layer free of
        a numpy dependency.
        """
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{max(0.01, duration):.3f}",
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 and not proc.stdout:
            raise RenderError(
                "frame extraction failed: " + proc.stderr.decode("utf-8", "replace")[-500:]
            )
        return proc.stdout


def _parse_rate(rate: Optional[str]) -> Optional[float]:
    """``"30000/1001"`` -> ``29.97``."""
    if not rate:
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            denominator = float(den)
            return float(num) / denominator if denominator else None
        return float(rate)
    except (TypeError, ValueError):
        return None
