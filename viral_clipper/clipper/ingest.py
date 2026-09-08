"""Resolve whatever the user gave us into a local media file.

Accepts a local path or any URL that ``yt-dlp`` can handle.  When the source
site publishes captions we download them too: they make an excellent (and
instant) transcript fallback when no speech model is installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from clipper.errors import DependencyMissing, IngestError
from clipper.ffmpeg import FFmpeg
from clipper.models import MediaInfo
from clipper.utils import ensure_dir, log, short_hash, slugify

ProgressFn = Callable[[str, float], None]

_URL_RE = re.compile(r"^(https?|ftp)://", re.IGNORECASE)
YTDLP_HINT = "pip install yt-dlp"

#: Prefer a 1080p-or-smaller MP4: bigger sources cost render time and gain
#: nothing, since every output is at most 1080x1920.
DEFAULT_FORMAT = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
)

_CAPTION_SUFFIXES = (".vtt", ".srt")


@dataclass
class SourceMedia:
    """A local media file plus whatever we learned about it on the way in."""

    path: Path
    info: MediaInfo
    title: str = ""
    uploader: str = ""
    source_url: str = ""
    caption_files: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.title or self.path.stem)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "title": self.title,
            "uploader": self.uploader,
            "source_url": self.source_url,
            "caption_files": [str(p) for p in self.caption_files],
            "info": self.info.to_dict(),
        }


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(str(source).strip()))


def resolve_source(
    source: str,
    workspace: Path,
    ffmpeg: Optional[FFmpeg] = None,
    *,
    progress: Optional[ProgressFn] = None,
    format_selector: str = DEFAULT_FORMAT,
    cookies_file: Optional[Path] = None,
) -> SourceMedia:
    """Return a :class:`SourceMedia` for a local path or a remote URL."""
    ffmpeg = ffmpeg or FFmpeg()
    workspace = ensure_dir(Path(workspace))
    text = str(source).strip()

    if is_url(text):
        media = _download(
            text,
            workspace,
            progress=progress,
            format_selector=format_selector,
            cookies_file=cookies_file,
        )
    else:
        path = Path(text).expanduser()
        if not path.exists():
            raise IngestError(f"no such file: {path}")
        media = SourceMedia(path=path, info=MediaInfo(path=str(path), duration=0.0))
        media.title = path.stem
        media.caption_files = _sidecar_captions(path)

    media.info = ffmpeg.probe(media.path)
    media.info.title = media.title
    media.info.source_url = media.source_url
    media.info.uploader = media.uploader
    if media.info.duration <= 0:
        raise IngestError(f"{media.path} has no readable duration")
    if not media.info.has_audio:
        raise IngestError(
            f"{media.path} has no audio track; the clipper needs speech to find moments"
        )
    if progress:
        progress("source ready", 1.0)
    return media


def _sidecar_captions(path: Path) -> List[Path]:
    """Caption files sitting next to a local video, e.g. ``talk.en.vtt``."""
    found: List[Path] = []
    for sibling in path.parent.glob(f"{glob_escape(path.stem)}*"):
        if sibling.suffix.lower() in _CAPTION_SUFFIXES:
            found.append(sibling)
    return sorted(found)


def glob_escape(text: str) -> str:
    """Escape glob metacharacters so a literal stem matches itself."""
    return re.sub(r"([\[\]*?])", r"[\1]", text)


# ---------------------------------------------------------------------------
# yt-dlp
# ---------------------------------------------------------------------------


def _download(
    url: str,
    workspace: Path,
    *,
    progress: Optional[ProgressFn],
    format_selector: str,
    cookies_file: Optional[Path],
) -> SourceMedia:
    downloads = ensure_dir(workspace / "downloads")
    stem = short_hash(url)
    target_template = str(downloads / f"{stem}.%(ext)s")

    existing = _existing_download(downloads, stem)
    if existing:
        log.info("reusing cached download %s", existing)
        if progress:
            progress("using cached download", 1.0)
        return SourceMedia(
            path=existing,
            info=MediaInfo(path=str(existing), duration=0.0),
            title=_read_cached_title(downloads, stem) or existing.stem,
            source_url=url,
            caption_files=_sidecar_captions(existing),
        )

    options: Dict[str, Any] = {
        "format": format_selector,
        "outtmpl": target_template,
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en.*", "en", "-live_chat"],
        "subtitlesformat": "vtt",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    if cookies_file:
        options["cookiefile"] = str(cookies_file)

    info = _download_with_module(url, options, progress)
    if info is None:
        info = _download_with_cli(url, options, progress)

    path = _existing_download(downloads, stem)
    if not path:
        raise IngestError(f"yt-dlp reported success but no media file appeared for {url}")

    title = str(info.get("title") or path.stem)
    (downloads / f"{stem}.title.txt").write_text(title, encoding="utf-8")
    return SourceMedia(
        path=path,
        info=MediaInfo(path=str(path), duration=float(info.get("duration") or 0.0)),
        title=title,
        uploader=str(info.get("uploader") or info.get("channel") or ""),
        source_url=url,
        caption_files=_sidecar_captions(path),
        metadata={
            "id": info.get("id"),
            "extractor": info.get("extractor_key"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "upload_date": info.get("upload_date"),
            "webpage_url": info.get("webpage_url") or url,
        },
    )


def _existing_download(downloads: Path, stem: str) -> Optional[Path]:
    for candidate in sorted(downloads.glob(f"{glob_escape(stem)}.*")):
        if candidate.suffix.lower() in _CAPTION_SUFFIXES:
            continue
        if candidate.name.endswith(".title.txt") or candidate.suffix == ".part":
            continue
        return candidate
    return None


def _read_cached_title(downloads: Path, stem: str) -> str:
    marker = downloads / f"{stem}.title.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return ""


def _download_with_module(
    url: str, options: Dict[str, Any], progress: Optional[ProgressFn]
) -> Optional[Dict[str, Any]]:
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        return None

    if progress:

        def hook(status: Dict[str, Any]) -> None:
            if status.get("status") != "downloading":
                return
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            fraction = (done / total) if total else 0.0
            progress("downloading source", min(0.99, fraction))

        options = dict(options, progress_hooks=[hook])

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:  # yt_dlp raises a wide range of custom errors
        raise IngestError(f"yt-dlp could not download {url}: {exc}") from exc
    if isinstance(info, dict) and "entries" in info:
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise IngestError(f"no downloadable entries at {url}")
        info = entries[0]
    return info or {}


def _download_with_cli(
    url: str, options: Dict[str, Any], progress: Optional[ProgressFn]
) -> Dict[str, Any]:
    binary = os.environ.get("YTDLP_BINARY") or shutil.which("yt-dlp")
    if not binary:
        raise DependencyMissing("yt-dlp (needed to download from a URL)", YTDLP_HINT)

    args = [
        binary,
        "--no-playlist",
        "--no-warnings",
        "--retries",
        "3",
        "-f",
        options["format"],
        "-o",
        options["outtmpl"],
        "--merge-output-format",
        "mp4",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "en.*,en",
        "--sub-format",
        "vtt",
        "--print-json",
        "--no-simulate",
    ]
    if options.get("cookiefile"):
        args += ["--cookies", str(options["cookiefile"])]
    args.append(url)

    if progress:
        progress("downloading source", 0.05)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise IngestError(f"yt-dlp failed for {url}:\n" + "\n".join(tail))

    import json

    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}
