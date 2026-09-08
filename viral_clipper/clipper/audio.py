"""Audio excitement analysis.

Speech alone cannot tell you where a room laughed, where a voice cracked or
where the music dropped.  A cheap RMS envelope can, so every candidate window
is scored against the loudness distribution of the whole video.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from clipper.ffmpeg import FFmpeg
from clipper.utils import clamp, moving_average, percentile


@dataclass
class WindowEnergy:
    """Energy statistics for one candidate window, all in ``[0, 1]``."""

    peak: float = 0.5
    variation: float = 0.5
    dynamics: float = 0.5
    silence_ratio: float = 0.0
    samples: int = 0

    @property
    def excitement(self) -> float:
        """A single number: loud, varied and mostly-not-silent scores high."""
        base = 0.45 * self.peak + 0.3 * self.variation + 0.25 * self.dynamics
        return clamp(base * (1.0 - 0.6 * self.silence_ratio), 0.0, 1.0)


@dataclass
class EnergyProfile:
    """The loudness envelope of a whole video, plus its own distribution."""

    times: List[float]
    levels: List[float]
    loud_reference: float = 0.0
    quiet_reference: float = -60.0
    typical_delta: float = 1.0
    typical_spread: float = 1.0
    silence_threshold: float = -55.0

    @classmethod
    def empty(cls) -> "EnergyProfile":
        return cls(times=[], levels=[])

    @property
    def available(self) -> bool:
        return len(self.levels) >= 4

    @classmethod
    def from_samples(cls, samples: Sequence[Tuple[float, float]]) -> "EnergyProfile":
        if not samples:
            return cls.empty()
        times = [t for t, _ in samples]
        # A light smoothing pass removes single-frame spikes that are codec
        # artefacts rather than real emphasis.
        levels = moving_average([lvl for _, lvl in samples], 3)

        loud = percentile(levels, 0.95)
        quiet = percentile(levels, 0.10)
        deltas = [abs(levels[i] - levels[i - 1]) for i in range(1, len(levels))]
        typical_delta = max(0.25, percentile(deltas, 0.75)) if deltas else 1.0

        mean = sum(levels) / len(levels)
        spread = (sum((lvl - mean) ** 2 for lvl in levels) / len(levels)) ** 0.5
        return cls(
            times=times,
            levels=levels,
            loud_reference=loud,
            quiet_reference=quiet,
            typical_delta=typical_delta,
            typical_spread=max(0.5, spread),
            silence_threshold=min(quiet + 3.0, -35.0),
        )

    def slice(self, start: float, end: float) -> List[float]:
        if not self.available or end <= start:
            return []
        left = bisect.bisect_left(self.times, start)
        right = bisect.bisect_right(self.times, end)
        return self.levels[left:right]

    def window(self, start: float, end: float) -> WindowEnergy:
        """Score one window against the distribution of the whole video."""
        chunk = self.slice(start, end)
        if len(chunk) < 2:
            return WindowEnergy(samples=len(chunk))

        span = max(1e-6, self.loud_reference - self.quiet_reference)
        peak = clamp((max(chunk) - self.quiet_reference) / span, 0.0, 1.0)

        mean = sum(chunk) / len(chunk)
        spread = (sum((lvl - mean) ** 2 for lvl in chunk) / len(chunk)) ** 0.5
        variation = clamp(spread / (self.typical_spread * 1.5), 0.0, 1.0)

        deltas = [abs(chunk[i] - chunk[i - 1]) for i in range(1, len(chunk))]
        mean_delta = sum(deltas) / len(deltas)
        dynamics = clamp(mean_delta / (self.typical_delta * 2.0), 0.0, 1.0)

        silent = sum(1 for lvl in chunk if lvl <= self.silence_threshold)
        return WindowEnergy(
            peak=peak,
            variation=variation,
            dynamics=dynamics,
            silence_ratio=silent / len(chunk),
            samples=len(chunk),
        )


def analyze(source: Path, ffmpeg: Optional[FFmpeg] = None, *, window: float = 0.1) -> EnergyProfile:
    """Build an :class:`EnergyProfile` for a media file.

    Returns an empty profile (rather than raising) when the file has no
    analysable audio, so scoring can fall back to text-only signals.
    """
    ffmpeg = ffmpeg or FFmpeg()
    return EnergyProfile.from_samples(ffmpeg.energy_envelope(Path(source), window=window))
