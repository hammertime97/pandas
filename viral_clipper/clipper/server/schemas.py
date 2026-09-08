"""Request models for the web API.

Kept in their own module so the Pydantic import stays out of the package's
import path until the server is actually being used, while still living at
module scope where FastAPI can resolve the annotation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from clipper.config import LAYOUTS, PRESETS, ClipperConfig
from clipper.subtitles import STYLE_PRESETS


class ClipRequest(BaseModel):
    """Options accepted when submitting a clipping job."""

    source: str = Field(..., min_length=1, description="video URL or local file path")
    platform: str = "tiktok"
    max_clips: int = Field(5, ge=1, le=25)
    min_duration: Optional[float] = Field(None, gt=0, le=600)
    max_duration: Optional[float] = Field(None, gt=0, le=600)
    min_score: float = Field(0.0, ge=0, le=100)
    layout: str = "auto"
    burn_subtitles: bool = True
    caption_style: str = "punch"
    uppercase_captions: bool = True
    highlight_color: str = "#FFE14D"
    whisper_model: str = "small"
    language: Optional[str] = None
    use_llm: bool = False
    dry_run: bool = False

    def to_config(self) -> ClipperConfig:
        """Build a validated :class:`ClipperConfig`; raises ``ValueError``."""
        if self.platform not in PRESETS:
            raise ValueError(
                f"unknown platform {self.platform!r}; "
                f"expected one of {', '.join(sorted(PRESETS))}"
            )
        if self.layout not in LAYOUTS:
            raise ValueError(
                f"unknown layout {self.layout!r}; expected one of {', '.join(LAYOUTS)}"
            )
        if self.caption_style not in STYLE_PRESETS:
            raise ValueError(
                f"unknown caption style {self.caption_style!r}; "
                f"expected one of {', '.join(sorted(STYLE_PRESETS))}"
            )
        return ClipperConfig(
            platform=self.platform,
            max_clips=self.max_clips,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            min_score=self.min_score,
            layout=self.layout,
            burn_subtitles=self.burn_subtitles,
            caption_style=self.caption_style,
            uppercase_captions=self.uppercase_captions,
            highlight_color=self.highlight_color,
            whisper_model=self.whisper_model,
            language=self.language,
            use_llm=self.use_llm,
            dry_run=self.dry_run,
        ).validate()
