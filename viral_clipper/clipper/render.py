"""Build the ffmpeg filter graph for a clip and render it.

Everything happens in a single ffmpeg pass: trim, reframe, scale, burn
captions and normalise loudness.  Re-encoding once keeps quality up and
render time down compared with a chain of intermediate files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from clipper.config import PlatformPreset
from clipper.errors import RenderError
from clipper.ffmpeg import FFmpeg
from clipper.reframe import CropPlan, write_sendcmd
from clipper.utils import ensure_dir, log


def escape_ffmpeg_path(path: Path) -> str:
    """Escape a path for use as a filter argument.

    ffmpeg unescapes twice here — once splitting the filtergraph, once
    splitting a filter's ``key=value:key=value`` options — so a character that
    matters at the option level needs escaping at both. That is why the drive
    colon in ``C:\\Users`` comes out as ``\\\\:``: one level of escaping
    survives the graph parser and reaches the option parser.

    Backslashes are turned into forward slashes first. ffmpeg accepts those on
    Windows, and it removes a whole class of escaping problems.

    Prefer :func:`RenderRequest.filter_arg`, which sidesteps all of this by
    running ffmpeg from the file's own directory.
    """
    text = str(path).replace("\\", "/")
    # Order matters: escape the escape character before anything that uses it.
    text = text.replace("'", "\\\\\\'")
    text = text.replace(":", "\\\\:")
    for char in ("[", "]", ",", ";"):
        text = text.replace(char, "\\" + char)
    return text


@dataclass
class RenderRequest:
    """One clip's worth of render instructions."""

    source: Path
    output: Path
    start: float
    end: float
    preset: PlatformPreset
    plan: Optional[CropPlan] = None
    layout: str = "auto"
    subtitle_path: Optional[Path] = None
    sendcmd_path: Optional[Path] = None
    normalize_audio: bool = True
    ffmpeg_preset: str = "veryfast"
    fonts_dir: Optional[Path] = None

    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)

    @property
    def working_dir(self) -> Path:
        """Where ffmpeg is run from, so filter files need no path at all."""
        return self.output.parent

    def filter_arg(self, path: Optional[Path]) -> str:
        """How to refer to ``path`` inside the filtergraph.

        Files that sit next to the output are named directly, because ffmpeg
        runs from that folder. A Windows path in a filtergraph is a minefield
        of double-escaped colons and backslashes; not putting one there is far
        more reliable than escaping it correctly.
        """
        if path is None:
            return ""
        path = Path(path)
        if path.parent == self.working_dir:
            return escape_ffmpeg_path(Path(path.name))
        return escape_ffmpeg_path(path)


def build_video_chain(request: RenderRequest) -> List[str]:
    """The ordered list of video filters for this request.

    Kept separate from :func:`render_clip` so the graph can be unit tested
    without running ffmpeg.
    """
    preset = request.preset
    width, height = preset.width, preset.height
    filters: List[str] = ["setpts=PTS-STARTPTS"]

    layout = request.layout
    plan = request.plan

    if layout in ("auto", "center") and plan is not None:
        needs_crop = plan.crop_w != plan.source_w or plan.crop_h != plan.source_h
        if needs_crop:
            if layout == "auto" and not plan.static and request.sendcmd_path:
                # sendcmd rewrites crop's x as the clip plays, which is what
                # turns a static window into a slow pan that follows the subject.
                # No `eval=frame` here: ffmpeg 7 removed the option, and a crop
                # driven by commands re-evaluates on every version regardless.
                filters.append(f"sendcmd=f={request.filter_arg(request.sendcmd_path)}")
                filters.append(
                    f"crop=w={plan.crop_w}:h={plan.crop_h}:x={plan.x}:y={plan.y}"
                )
            else:
                filters.append(
                    f"crop=w={plan.crop_w}:h={plan.crop_h}:x={plan.x}:y={plan.y}"
                )
        filters.append(f"scale={width}:{height}:flags=lanczos")
    elif layout == "fit":
        filters.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    else:  # "blur": the source, letterboxed over a blurred fill of itself
        filters.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")

    filters.append("setsar=1")
    if request.subtitle_path:
        subtitle_arg = f"subtitles=filename={request.filter_arg(request.subtitle_path)}"
        if request.fonts_dir:
            subtitle_arg += f":fontsdir={request.filter_arg(request.fonts_dir)}"
        filters.append(subtitle_arg)
    filters.append("format=yuv420p")
    return filters


def build_filter_complex(request: RenderRequest, *, has_audio: bool = True) -> str:
    """Full ``-filter_complex`` string, including the blur background branch."""
    preset = request.preset
    width, height = preset.width, preset.height
    parts: List[str] = []

    if request.layout == "blur":
        # Split the source: one copy becomes a zoomed, blurred backdrop, the
        # other is letterboxed on top of it at its native aspect ratio.
        parts.append("[0:v]setpts=PTS-STARTPTS,split=2[bgsrc][fgsrc]")
        parts.append(
            f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=28,eq=brightness=-0.10:saturation=1.1[bg]"
        )
        parts.append(
            f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease:"
            "flags=lanczos[fg]"
        )
        overlay_chain = ["setsar=1"]
        if request.subtitle_path:
            subtitle_arg = (
                f"subtitles=filename={request.filter_arg(request.subtitle_path)}"
            )
            if request.fonts_dir:
                subtitle_arg += f":fontsdir={request.filter_arg(request.fonts_dir)}"
            overlay_chain.append(subtitle_arg)
        overlay_chain.append("format=yuv420p")
        parts.append(
            "[bg][fg]overlay=(W-w)/2:(H-h)/2," + ",".join(overlay_chain) + "[v]"
        )
    else:
        parts.append("[0:v]" + ",".join(build_video_chain(request)) + "[v]")

    if has_audio:
        audio_filters = ["asetpts=PTS-STARTPTS"]
        if request.normalize_audio:
            # Platform playback normalises to roughly -14 LUFS; matching it
            # here stops clips sounding quiet next to native uploads.
            audio_filters.append(
                f"loudnorm=I={preset.loudness_target:g}:TP=-1.5:LRA=11"
            )
        audio_filters.append("aresample=48000:first_pts=0")
        parts.append("[0:a]" + ",".join(audio_filters) + "[a]")

    return ";".join(parts)


def build_command(
    ffmpeg: FFmpeg, request: RenderRequest, *, has_audio: bool = True
) -> List[str]:
    """The full argument list handed to ffmpeg (minus the binary itself)."""
    preset = request.preset
    args: List[str] = [
        "-ss",
        f"{max(0.0, request.start):.3f}",
        "-i",
        str(Path(request.source).resolve()),
        "-t",
        f"{request.duration:.3f}",
        "-filter_complex",
        build_filter_complex(request, has_audio=has_audio),
        "-map",
        "[v]",
    ]
    if has_audio:
        args += ["-map", "[a]"]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        request.ffmpeg_preset,
        "-crf",
        str(preset.crf),
        "-maxrate",
        preset.video_bitrate,
        "-bufsize",
        "16M",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-r",
        str(preset.fps),
        "-g",
        str(preset.fps * 2),
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", preset.audio_bitrate, "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", str(Path(request.output).resolve())]
    return args


def render_clip(
    ffmpeg: FFmpeg, request: RenderRequest, *, has_audio: bool = True, timeout: float = 3600.0
) -> Path:
    """Render one clip, returning the output path."""
    ensure_dir(request.output.parent)
    if request.plan is not None and request.sendcmd_path and not request.plan.static:
        write_sendcmd(request.plan, request.sendcmd_path)

    args = build_command(ffmpeg, request, has_audio=has_audio)
    # Run from the clip's own folder so the filtergraph carries bare filenames.
    request.working_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "rendering %s (%.1fs -> %.1fs, layout=%s)",
        request.output.name,
        request.start,
        request.end,
        request.layout,
    )
    ffmpeg.run(args, timeout=timeout, cwd=request.working_dir)
    if not request.output.exists() or request.output.stat().st_size == 0:
        raise RenderError(f"ffmpeg produced no output for {request.output}")
    return request.output
