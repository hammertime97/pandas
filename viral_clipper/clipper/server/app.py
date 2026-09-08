"""FastAPI app: JSON API plus the single page UI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from clipper.config import LAYOUTS, PRESETS, ClipperConfig
from clipper.errors import DependencyMissing
from clipper.server.jobs import JobManager
from clipper.subtitles import STYLE_PRESETS
from clipper.utils import log, slugify

MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
UPLOAD_CHUNK = 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".flv",
    ".m4a", ".mp3", ".wav",
}

STATIC_DIR = Path(__file__).parent / "static"

# FastAPI resolves endpoint annotations against this module's globals, so the
# request model has to live here rather than inside create_app(). The guard
# keeps `import clipper.server.app` working without the server extras so the
# CLI can raise a helpful DependencyMissing instead of an ImportError.
try:  # pragma: no cover - trivial import guard
    from fastapi import File, UploadFile

    from clipper.server.schemas import ClipRequest
except ImportError:  # pragma: no cover
    ClipRequest = None  # type: ignore[assignment]
    File = UploadFile = None  # type: ignore[assignment]


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise DependencyMissing(
            "fastapi and uvicorn (needed for `clipper serve`)",
            "pip install 'viral-clipper[server]'",
        ) from exc


def create_app(workspace: Path = Path("workspace"), workers: int = 1):
    """Build the FastAPI application."""
    _require_fastapi()
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    from contextlib import asynccontextmanager

    manager = JobManager(Path(workspace), workers=workers)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.shutdown()

    app = FastAPI(title="Viral Clipper", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager

    # -- pages ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.exists():  # pragma: no cover - only if the package is broken
            return HTMLResponse("<h1>UI assets are missing</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # -- metadata ---------------------------------------------------------

    @app.get("/api/options")
    def options() -> Dict[str, Any]:
        """Everything the UI needs to build its form."""
        return {
            "platforms": [
                {
                    "name": name,
                    "width": preset.width,
                    "height": preset.height,
                    "min_duration": preset.min_duration,
                    "max_duration": preset.max_duration,
                    "target_duration": preset.target_duration,
                }
                for name, preset in sorted(PRESETS.items())
            ],
            "layouts": list(LAYOUTS),
            "caption_styles": sorted(STYLE_PRESETS),
        }

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        from clipper.cli import _module_available
        from clipper.ffmpeg import find_ffmpeg

        return {
            "ok": bool(find_ffmpeg()),
            "ffmpeg": find_ffmpeg(),
            "yt_dlp": _module_available("yt_dlp") or bool(shutil.which("yt-dlp")),
            "whisper": _module_available("faster_whisper") or _module_available("whisper"),
            "opencv": _module_available("cv2"),
            "anthropic": _module_available("anthropic"),
        }

    # -- jobs -------------------------------------------------------------

    @app.post("/api/jobs", status_code=201)
    def create_job(request: ClipRequest) -> Dict[str, Any]:
        source = request.source.strip()
        if not source.lower().startswith(("http://", "https://")):
            path = Path(source).expanduser()
            if not path.exists():
                raise HTTPException(400, f"no such file: {source}")
            source = str(path.resolve())
        try:
            config = request.to_config()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        job = manager.submit(source, config)
        return job.to_dict()

    @app.get("/api/jobs")
    def list_jobs() -> Dict[str, Any]:
        return {"jobs": [job.to_dict(include_clips=False) for job in manager.list()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, Any]:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        manager.cancel(job_id)
        if not manager.forget(job_id):
            raise HTTPException(409, "job is still running")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/clips/{index}/{kind}")
    def clip_asset(job_id: str, index: int, kind: str) -> Any:
        if kind not in ("video", "thumbnail", "subtitles"):
            raise HTTPException(404, "unknown asset")
        path = manager.clip_file(job_id, index, kind)
        if not path:
            raise HTTPException(404, "asset not available")
        media_type = {
            "video": "video/mp4",
            "thumbnail": "image/jpeg",
            "subtitles": "text/plain; charset=utf-8",
        }[kind]
        return FileResponse(path, media_type=media_type, filename=path.name)

    # -- uploads ----------------------------------------------------------

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                400,
                f"unsupported file type {suffix or '(none)'}; "
                f"expected one of {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}",
            )
        target = manager.uploads / f"{slugify(Path(name).stem)}{suffix}"
        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(UPLOAD_CHUNK):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file is too large")
                    handle.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        log.info("uploaded %s (%.1f MB)", target.name, written / 1e6)
        return {"path": str(target), "name": target.name, "bytes": written}

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    workspace: Path = Path("workspace"),
    workers: int = 1,
) -> None:  # pragma: no cover - runs a server
    """Run the UI and API with uvicorn."""
    _require_fastapi()
    import uvicorn

    app = create_app(workspace=workspace, workers=workers)
    print(f"Viral Clipper running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
