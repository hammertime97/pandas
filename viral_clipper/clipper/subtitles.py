"""Word-highlight ("karaoke") captions in ASS format.

Burned-in captions are not optional for short form: most of the feed is
watched muted.  The style that works is a small group of words on screen at
once, with the word currently being spoken highlighted.

Timings are emitted relative to the start of the clip, so the resulting file
can be burned straight onto a trimmed segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from clipper.models import Word
from clipper.utils import ass_timecode, clamp

#: Style presets: (font, bold, outline, shadow, border_style, scale_pop)
STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "punch": {
        "font": "Arial Black",
        "bold": True,
        "outline": 5.0,
        "shadow": 2.0,
        "border_style": 1,  # outline + drop shadow
        "scale_pop": 112,  # % size of the active word
        "uppercase": True,
    },
    "clean": {
        "font": "Arial",
        "bold": True,
        "outline": 3.0,
        "shadow": 1.0,
        "border_style": 1,
        "scale_pop": 105,
        "uppercase": False,
    },
    "minimal": {
        "font": "Arial",
        "bold": False,
        "outline": 2.0,
        "shadow": 0.0,
        "border_style": 1,
        "scale_pop": 100,
        "uppercase": False,
        "highlight": False,
    },
}

DEFAULT_STYLE = "punch"


def hex_to_ass_colour(value: str, alpha: int = 0) -> str:
    """``"#FFE14D"`` -> ``"&H004DE1FF"`` (ASS is AABBGGRR)."""
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"expected a 6 digit hex colour, got {value!r}")
    red, green, blue = text[0:2], text[2:4], text[4:6]
    return f"&H{alpha:02X}{blue}{green}{red}".upper()


def escape_ass(text: str) -> str:
    """Escape the characters that would otherwise be read as ASS markup."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
        .replace("\r", " ")
    )


@dataclass
class CaptionChunk:
    """The group of words on screen at one time."""

    words: List[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def group_words(
    words: Sequence[Word],
    *,
    max_words: int = 4,
    max_chars: int = 26,
    max_gap: float = 0.55,
    max_duration: float = 3.0,
) -> List[CaptionChunk]:
    """Split a word stream into on-screen groups.

    A group ends on any of: word count, character count, a pause, sentence
    punctuation, or a duration cap.  Sentence punctuation wins because
    breaking a caption mid-sentence reads worse than a short caption.
    """
    chunks: List[CaptionChunk] = []
    current: List[Word] = []
    chars = 0

    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if current:
            gap = word.start - current[-1].end
            too_long = word.end - current[0].start > max_duration
            if (
                len(current) >= max_words
                or chars + len(token) + 1 > max_chars
                or gap > max_gap
                or too_long
            ):
                chunks.append(CaptionChunk(current))
                current, chars = [], 0
        current.append(word)
        chars += len(token) + 1
        if token.endswith((".", "!", "?", "…", ":")):
            chunks.append(CaptionChunk(current))
            current, chars = [], 0

    if current:
        chunks.append(CaptionChunk(current))
    return chunks


def _style_block(
    *,
    width: int,
    height: int,
    font: str,
    font_size: int,
    bold: bool,
    outline: float,
    shadow: float,
    border_style: int,
    margin_v: int,
    primary: str,
    outline_colour: str,
    back_colour: str,
) -> str:
    fields = [
        "Default",
        font,
        str(font_size),
        primary,
        primary,  # SecondaryColour, unused since we do our own highlighting
        outline_colour,
        back_colour,
        "-1" if bold else "0",
        "0",  # italic
        "0",  # underline
        "0",  # strikeout
        "100",  # ScaleX
        "100",  # ScaleY
        "0",  # Spacing
        "0",  # Angle
        str(border_style),
        f"{outline:g}",
        f"{shadow:g}",
        "2",  # Alignment: bottom centre
        str(int(width * 0.06)),  # MarginL
        str(int(width * 0.06)),  # MarginR
        str(margin_v),
        "1",  # Encoding
    ]
    return "Style: " + ",".join(fields)


def build_ass(
    words: Sequence[Word],
    *,
    width: int = 1080,
    height: int = 1920,
    font_size: int = 78,
    margin_v: int = 430,
    style: str = DEFAULT_STYLE,
    font: Optional[str] = None,
    uppercase: Optional[bool] = None,
    highlight_colour: str = "#FFE14D",
    text_colour: str = "#FFFFFF",
    outline_colour: str = "#000000",
    max_words: int = 4,
    offset: float = 0.0,
) -> str:
    """Render ``words`` as an ASS subtitle document.

    ``offset`` is subtracted from every timestamp, which is how absolute
    source-relative word timings become clip-relative caption timings.
    """
    preset = dict(STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE]))
    font_name = font or str(preset["font"])
    is_upper = preset.get("uppercase", False) if uppercase is None else uppercase
    do_highlight = bool(preset.get("highlight", True))
    scale_pop = int(preset.get("scale_pop", 110))

    primary = hex_to_ass_colour(text_colour)
    active = hex_to_ass_colour(highlight_colour)
    outline_col = hex_to_ass_colour(outline_colour)
    back_col = hex_to_ass_colour(outline_colour, alpha=0x60)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        _style_block(
            width=width,
            height=height,
            font=font_name,
            font_size=font_size,
            bold=bool(preset["bold"]),
            outline=float(preset["outline"]),
            shadow=float(preset["shadow"]),
            border_style=int(preset["border_style"]),
            margin_v=margin_v,
            primary=primary,
            outline_colour=outline_col,
            back_colour=back_col,
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text"
        ),
    ]

    events: List[str] = []
    for chunk in group_words(words, max_words=max_words):
        tokens = [w.text.strip() for w in chunk.words]
        if is_upper:
            tokens = [t.upper() for t in tokens]
        tokens = [escape_ass(t) for t in tokens]

        if not do_highlight:
            events.append(
                _dialogue(chunk.start - offset, chunk.end - offset, " ".join(tokens))
            )
            continue

        for index, word in enumerate(chunk.words):
            start = word.start - offset
            # Tile events edge to edge so the caption never blinks between words.
            if index + 1 < len(chunk.words):
                end = chunk.words[index + 1].start - offset
            else:
                end = chunk.end - offset
            if end <= start:
                end = start + 0.08
            rendered = " ".join(
                f"{{\\c{active}\\fscx{scale_pop}\\fscy{scale_pop}}}{token}{{\\r}}"
                if position == index
                else token
                for position, token in enumerate(tokens)
            )
            events.append(_dialogue(start, end, rendered))

    return "\n".join(header + events) + "\n"


def _dialogue(start: float, end: float, text: str) -> str:
    start = max(0.0, start)
    end = max(start + 0.02, end)
    return (
        f"Dialogue: 0,{ass_timecode(start)},{ass_timecode(end)},Default,,0,0,0,,{text}"
    )


def write_ass(content: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def scale_font_size(base: int, width: int, reference_width: int = 1080) -> int:
    """Keep captions the same relative size at any output resolution."""
    return max(18, int(round(base * clamp(width / reference_width, 0.4, 3.0))))
