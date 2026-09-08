"""Small helpers shared across the pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

log = logging.getLogger("clipper")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_length: int = 60) -> str:
    """Turn arbitrary text into a filesystem and URL safe slug."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_text).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0] or slug[:max_length]
    return slug or "clip"


def short_hash(*parts: object, length: int = 10) -> str:
    """Stable short hash of the string form of ``parts``."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return digest.hexdigest()[:length]


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def clamp(value: float, low: float, high: float) -> float:
    if low > high:  # defensive: an empty range collapses to its single point
        return low
    return max(low, min(high, value))


def timecode(seconds: float, decimals: int = 2) -> str:
    """``123.4`` -> ``"00:02:03.40"``.

    Rounding is done before the split so that, say, 119.7s at zero decimals
    renders as ``00:02:00`` rather than ``00:01:60``.
    """
    seconds = round(max(0.0, float(seconds)), decimals)
    hours, remainder = divmod(seconds, 3600.0)
    minutes, secs = divmod(remainder, 60.0)
    width = 2 if decimals == 0 else decimals + 3
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:0{width}.{decimals}f}"


def ass_timecode(seconds: float) -> str:
    """ASS uses ``H:MM:SS.cc`` with centisecond precision."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600.0)
    minutes, secs = divmod(remainder, 60.0)
    centis = int(round(secs * 100.0))
    secs, centis = divmod(centis, 100)
    return f"{int(hours):d}:{int(minutes):02d}:{int(secs):02d}.{centis:02d}"


def srt_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600.0)
    minutes, secs = divmod(remainder, 60.0)
    millis = int(round((secs - int(secs)) * 1000.0))
    secs = int(secs)
    if millis == 1000:
        millis, secs = 0, secs + 1
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:02d},{millis:03d}"


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals (0 when disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Centered moving average that keeps the input length."""
    if window <= 1 or not values:
        return list(values)
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in ``[0, 1]``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = clamp(q, 0.0, 1.0) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def run_command(
    args: Sequence[str],
    *,
    capture: bool = True,
    check: bool = True,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, logging the command and surfacing stderr on failure."""
    printable = " ".join(str(a) for a in args)
    log.debug("running: %s", printable)
    started = time.monotonic()
    proc = subprocess.run(
        [str(a) for a in args],
        capture_output=capture,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    log.debug("finished in %.1fs (rc=%s)", time.monotonic() - started, proc.returncode)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise subprocess.CalledProcessError(
            proc.returncode, args, output=proc.stdout, stderr="\n".join(tail)
        )
    return proc


def unique(items: Iterable[str]) -> List[str]:
    """Order preserving de-duplication."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
