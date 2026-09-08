"""Pytest configuration: make the package importable from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clipper.ffmpeg import find_ffmpeg  # noqa: E402


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


needs_ffmpeg = pytest.mark.skipif(
    find_ffmpeg() is None, reason="ffmpeg is not installed"
)
needs_fastapi = pytest.mark.skipif(
    not (_module_available("fastapi") and _module_available("httpx")),
    reason="fastapi/httpx are not installed",
)


@pytest.fixture(scope="session")
def ffmpeg_binary() -> str:
    binary = find_ffmpeg()
    if binary is None:
        pytest.skip("ffmpeg is not installed")
    return binary


@pytest.fixture(scope="session")
def sample_source(tmp_path_factory, ffmpeg_binary):
    """A landscape video plus a matching SRT sidecar, built once per session."""
    from tests.fixtures import make_video, script_duration, write_srt

    directory = tmp_path_factory.mktemp("source")
    video = make_video(directory / "talk.mp4", script_duration(), ffmpeg=ffmpeg_binary)
    write_srt(directory / "talk.en.srt")
    return video
