"""Data structures that travel through the pipeline.

Everything here is a plain dataclass with a ``to_dict`` so the whole state of
a job can be serialised to JSON for the web UI and for reproducible reruns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from clipper.utils import normalize_whitespace


@dataclass
class Word:
    """A single spoken word with its timing."""

    text: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word":
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class Sentence:
    """A group of words that reads as one complete thought."""

    words: List[Word]
    terminal: bool = False  # ends on . ? ! rather than on a pause

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text(self) -> str:
        return normalize_whitespace(" ".join(w.text for w in self.words))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "terminal": self.terminal,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class Transcript:
    """All words in a source video, plus how we got them."""

    words: List[Word]
    language: str = "en"
    source: str = "unknown"  # which backend produced this

    @property
    def text(self) -> str:
        return normalize_whitespace(" ".join(w.text for w in self.words))

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def words_between(self, start: float, end: float) -> List[Word]:
        """Words whose midpoint falls inside ``[start, end]``."""
        return [w for w in self.words if start <= (w.start + w.end) / 2.0 <= end]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        return cls(
            words=[Word.from_dict(w) for w in data.get("words", [])],
            language=data.get("language", "en"),
            source=data.get("source", "unknown"),
        )


@dataclass
class ScoreBreakdown:
    """Per-signal scores plus the weighted total, kept for explainability."""

    signals: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    @property
    def top_signals(self) -> List[str]:
        """Signal names ordered by their contribution to the total."""
        contributions = {
            name: value * self.weights.get(name, 0.0)
            for name, value in self.signals.items()
        }
        return sorted(contributions, key=contributions.get, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 2),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "weights": dict(self.weights),
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
        }


@dataclass
class Candidate:
    """A window of the source that might make a good short."""

    start: float
    end: float
    sentences: List[Sentence] = field(default_factory=list)
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    #: Silence immediately before/after the window. A long lead-in silence is
    #: strong evidence that this is where a new thought actually begins.
    lead_silence: float = 0.0
    trail_silence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text(self) -> str:
        return normalize_whitespace(" ".join(s.text for s in self.sentences))

    @property
    def words(self) -> List[Word]:
        return [w for s in self.sentences for w in s.words]

    @property
    def hook(self) -> str:
        """The opening sentence, which is what decides the swipe."""
        return self.sentences[0].text if self.sentences else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
            "hook": self.hook,
            "lead_silence": round(self.lead_silence, 3),
            "trail_silence": round(self.trail_silence, 3),
            "score": self.score.to_dict(),
        }


@dataclass
class SocialCopy:
    """Ready-to-paste metadata for a platform upload."""

    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    on_screen_hook: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Clip:
    """A rendered short, plus everything needed to explain and publish it."""

    clip_id: str
    index: int
    start: float
    end: float
    score: float
    breakdown: ScoreBreakdown
    transcript_text: str
    hook: str
    copy: SocialCopy = field(default_factory=SocialCopy)
    words: List[Word] = field(default_factory=list)
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    platform: str = "tiktok"
    width: int = 1080
    height: int = 1920

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 2),
            "breakdown": self.breakdown.to_dict(),
            "hook": self.hook,
            "transcript": self.transcript_text,
            "copy": self.copy.to_dict(),
            "platform": self.platform,
            "width": self.width,
            "height": self.height,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
            "subtitle_path": self.subtitle_path,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class MediaInfo:
    """What ffprobe (or a parsed ``ffmpeg -i``) tells us about a file."""

    path: str
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_audio: bool = True
    has_video: bool = True
    title: str = ""
    source_url: str = ""
    uploader: str = ""

    @property
    def aspect_ratio(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        return 0.0 < self.aspect_ratio < 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
