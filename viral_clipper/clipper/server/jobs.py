"""In-process job queue for clipping runs.

Deliberately simple: a bounded thread pool, jobs held in memory, and each
finished job's manifest written to disk so a restart can list past runs.
Rendering is ffmpeg-bound, so threads (not processes) are the right tool.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from clipper.config import ClipperConfig
from clipper.errors import ClipperError
from clipper.pipeline import JobResult, run_pipeline
from clipper.utils import ensure_dir, log

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

MAX_LOG_LINES = 60


@dataclass
class Job:
    """One clipping run, from submission to result."""

    job_id: str
    source: str
    config: ClipperConfig
    status: str = QUEUED
    progress: float = 0.0
    message: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[JobResult] = None
    log_lines: List[str] = field(default_factory=list)
    _cancelled: bool = False

    @property
    def display_name(self) -> str:
        if self.result:
            return self.result.source.title or Path(self.result.source.path).name
        if self.source.startswith("http"):
            return self.source
        return Path(self.source).name

    def to_dict(self, *, include_clips: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "job_id": self.job_id,
            "source": self.source,
            "name": self.display_name,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "platform": self.config.platform,
            "log": self.log_lines[-12:],
        }
        if include_clips and self.result:
            data["stats"] = self.result.stats
            data["output_dir"] = str(self.result.output_dir)
            data["clips"] = [clip.to_dict() for clip in self.result.clips]
        else:
            data["clips"] = []
        return data


class JobManager:
    """Submits, tracks and serves clipping jobs."""

    def __init__(self, workspace: Path, workers: int = 1) -> None:
        self.workspace = ensure_dir(Path(workspace))
        self.uploads = ensure_dir(self.workspace / "uploads")
        self._jobs: Dict[str, Job] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="clipper"
        )

    # -- submission -------------------------------------------------------

    def submit(self, source: str, config: ClipperConfig) -> Job:
        config.workspace = self.workspace
        if config.output_dir is None:
            config.output_dir = self.workspace / "clips"
        config.validate()

        job = Job(job_id=uuid.uuid4().hex[:12], source=source, config=config)
        with self._lock:
            self._jobs[job.job_id] = job
            self._futures[job.job_id] = self._pool.submit(self._run, job)
        log.info("queued job %s for %s", job.job_id, source)
        return job

    def _run(self, job: Job) -> None:
        if job._cancelled:
            job.status = CANCELLED
            job.message = "cancelled before it started"
            return
        job.status = RUNNING
        job.started_at = time.time()
        job.message = "starting"

        def progress(message: str, fraction: float) -> None:
            job.progress = fraction
            job.message = message
            if not job.log_lines or job.log_lines[-1] != message:
                job.log_lines.append(message)
                del job.log_lines[:-MAX_LOG_LINES]

        try:
            job.result = run_pipeline(job.source, job.config, progress=progress)
            job.status = DONE
            job.progress = 1.0
            job.message = f"{len(job.result.clips)} clips ready"
        except ClipperError as exc:
            job.status = ERROR
            job.error = str(exc)
            job.message = "failed"
            log.warning("job %s failed: %s", job.job_id, exc)
        except Exception as exc:  # unexpected: keep the traceback for the UI
            job.status = ERROR
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            log.error("job %s crashed:\n%s", job.job_id, traceback.format_exc())
        finally:
            job.finished_at = time.time()

    # -- lookup -----------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started yet; running jobs run to completion."""
        with self._lock:
            job = self._jobs.get(job_id)
            future = self._futures.get(job_id)
        if not job:
            return False
        job._cancelled = True
        if future and future.cancel():
            job.status = CANCELLED
            job.message = "cancelled"
            job.finished_at = time.time()
            return True
        return job.status in (DONE, ERROR, CANCELLED)

    def forget(self, job_id: str) -> bool:
        """Drop a finished job from the list (files on disk are kept)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == RUNNING:
                return False
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
        return True

    def clip_file(self, job_id: str, clip_index: int, kind: str = "video") -> Optional[Path]:
        """Resolve a clip artefact, refusing anything outside the job's folder."""
        job = self.get(job_id)
        if not job or not job.result:
            return None
        clip = next((c for c in job.result.clips if c.index == clip_index), None)
        if not clip:
            return None
        raw = {
            "video": clip.video_path,
            "thumbnail": clip.thumbnail_path,
            "subtitles": clip.subtitle_path,
        }.get(kind)
        if not raw:
            return None
        path = Path(raw).resolve()
        root = Path(job.result.output_dir).resolve()
        # Paths come from our own manifest, but check anyway: this endpoint is
        # reachable from the browser and must never serve outside the job dir.
        if root not in path.parents or not path.exists():
            log.warning("refusing to serve %s outside %s", path, root)
            return None
        return path

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def load_past_manifests(clips_root: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """Read manifests written by previous runs so the UI can show history."""
    root = Path(clips_root)
    if not root.exists():
        return []
    manifests = sorted(
        root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    out: List[Dict[str, Any]] = []
    for path in manifests[:limit]:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return out
