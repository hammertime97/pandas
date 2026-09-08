"""Configuration objects and per-platform presets.

A ``ClipperConfig`` is the single knob bag threaded through the pipeline.  It
is built from a :class:`PlatformPreset` and then overridden by whatever the
CLI or the web request asked for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
# Every signal produced by clipper.scoring is in [0, 1]; weights decide how
# much each one moves the final 0-100 virality score.  They do not have to sum
# to one, the total is normalised by the sum of the weights actually used.

DEFAULT_WEIGHTS: Dict[str, float] = {
    "hook": 2.4,  # does the first line stop the scroll
    "curiosity": 1.4,  # open loops, questions, "here's the thing"
    "payoff": 1.3,  # does the window actually resolve what it opened
    "emotion": 1.2,  # high-arousal language
    "quotability": 1.1,  # short declarative punchlines
    "completeness": 2.0,  # starts and ends on a whole thought
    "audio_energy": 1.0,  # loudness dynamics, laughter, emphasis
    "pace": 0.8,  # words per second inside the sweet spot
    "information_density": 0.8,  # content words vs filler
    "list_structure": 0.6,  # "three reasons", "first... second..."
    "laughter": 0.5,  # explicit reaction markers
    "duration_fit": 0.9,  # length vs the platform sweet spot
    "topic_start": 1.1,  # begins where a new thought begins
}


@dataclass
class PlatformPreset:
    """Everything that differs between TikTok, Reels and Shorts."""

    name: str
    width: int = 1080
    height: int = 1920
    fps: int = 30
    min_duration: float = 15.0
    max_duration: float = 60.0
    target_duration: float = 30.0
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    crf: int = 20
    caption_max_words: int = 4
    caption_font_size: int = 78
    caption_margin_v: int = 430  # distance from the bottom edge in pixels
    loudness_target: float = -14.0  # LUFS, what the platforms normalise to
    max_hashtags: int = 8
    weight_overrides: Dict[str, float] = field(default_factory=dict)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def weights(self) -> Dict[str, float]:
        merged = dict(DEFAULT_WEIGHTS)
        merged.update(self.weight_overrides)
        return merged


PRESETS: Dict[str, PlatformPreset] = {
    "tiktok": PlatformPreset(
        name="tiktok",
        min_duration=15.0,
        max_duration=60.0,
        target_duration=27.0,
        # TikTok punishes a slow open harder than anything else.
        weight_overrides={"hook": 2.8, "pace": 1.0},
    ),
    "reels": PlatformPreset(
        name="reels",
        min_duration=15.0,
        max_duration=75.0,
        target_duration=32.0,
        caption_margin_v=470,  # keep clear of the Reels UI overlay
        weight_overrides={"hook": 2.5, "emotion": 1.4},
    ),
    "shorts": PlatformPreset(
        name="shorts",
        min_duration=20.0,
        max_duration=59.0,
        target_duration=38.0,
        # Shorts viewers tolerate a longer setup for a bigger payoff.
        weight_overrides={"payoff": 1.6, "information_density": 1.0},
    ),
    "square": PlatformPreset(
        name="square",
        width=1080,
        height=1080,
        target_duration=30.0,
        caption_margin_v=120,
        caption_font_size=64,
    ),
}

DEFAULT_PLATFORM = "tiktok"

#: Crop strategies for turning a landscape source into a vertical clip.
LAYOUTS = ("auto", "center", "blur", "fit")


@dataclass
class ClipperConfig:
    """Runtime configuration for a single clipping job."""

    platform: str = DEFAULT_PLATFORM
    workspace: Path = Path("workspace")
    output_dir: Optional[Path] = None

    # selection
    max_clips: int = 5
    min_duration: Optional[float] = None  # falls back to the preset
    max_duration: Optional[float] = None
    max_overlap: float = 0.25  # fraction of the shorter clip
    min_score: float = 0.0
    content_similarity_limit: float = 0.65

    # transcription
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    language: Optional[str] = None
    transcript_path: Optional[Path] = None  # reuse an existing srt/vtt/json

    # framing / styling
    layout: str = "auto"
    burn_subtitles: bool = True
    uppercase_captions: bool = True
    caption_style: str = "punch"  # punch | clean | minimal
    highlight_color: str = "#FFE14D"
    tracking_fps: float = 4.0
    tracking_smoothing: float = 0.22  # EMA alpha, lower is calmer
    tracking_deadzone: float = 0.06  # fraction of frame width

    # rendering
    ffmpeg_preset: str = "veryfast"
    normalize_audio: bool = True
    keep_intermediates: bool = False
    thumbnails: bool = True

    # optional LLM re-ranking
    use_llm: bool = False
    llm_model: str = "claude-opus-5"
    llm_weight: float = 0.5  # blend factor between heuristic and LLM score
    llm_candidates: int = 12

    # misc
    dry_run: bool = False  # score and plan, but do not render
    verbose: bool = False

    def preset(self) -> PlatformPreset:
        try:
            base = PRESETS[self.platform]
        except KeyError:
            raise ValueError(
                f"unknown platform {self.platform!r}; "
                f"expected one of {', '.join(sorted(PRESETS))}"
            ) from None
        overrides: Dict[str, Any] = {}
        if self.min_duration is not None:
            overrides["min_duration"] = float(self.min_duration)
        if self.max_duration is not None:
            overrides["max_duration"] = float(self.max_duration)
        if overrides:
            merged = replace(base, **overrides)
            # Keep the target inside the (possibly narrowed) duration window.
            target = min(max(merged.target_duration, merged.min_duration), merged.max_duration)
            return replace(merged, target_duration=target)
        return base

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir else Path(self.workspace) / "clips"

    def validate(self) -> "ClipperConfig":
        """Raise ``ValueError`` on impossible combinations."""
        preset = self.preset()  # raises on an unknown platform
        if self.layout not in LAYOUTS:
            raise ValueError(
                f"unknown layout {self.layout!r}; expected one of {', '.join(LAYOUTS)}"
            )
        if preset.min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if preset.min_duration > preset.max_duration:
            raise ValueError(
                f"min_duration ({preset.min_duration}) exceeds "
                f"max_duration ({preset.max_duration})"
            )
        if self.max_clips < 1:
            raise ValueError("max_clips must be at least 1")
        if not 0.0 <= self.max_overlap < 1.0:
            raise ValueError("max_overlap must be in [0, 1)")
        if not 0.0 <= self.llm_weight <= 1.0:
            raise ValueError("llm_weight must be in [0, 1]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("workspace", "output_dir", "transcript_path"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipperConfig":
        """Build a config from a (possibly partial, possibly untrusted) dict."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k in known}
        for key in ("workspace", "output_dir", "transcript_path"):
            if kwargs.get(key) is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)
