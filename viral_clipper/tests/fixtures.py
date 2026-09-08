"""Shared test fixtures: a synthetic source video and a scripted transcript.

The transcript deliberately mixes three strong, self-contained moments with
rambling filler so tests can assert that scoring puts the strong ones on top.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

from clipper.models import Word

# (text, is_strong) - roughly 2.9 words/second when timed by make_words().
SCRIPT: List[Tuple[str, bool]] = [
    ("Okay so, um, welcome back everyone, and thanks for joining today, "
     "we have a lot to get through so let us just sort of dive in I guess.", False),
    ("You know it is like the usual thing where we just kind of go over "
     "the numbers and stuff and yeah anyway that is basically it.", False),

    ("Nobody tells you the biggest mistake founders make when they raise money. "
     "You think a higher valuation is always better for the company. "
     "It is not, and here is the thing nobody says out loud. "
     "A high valuation sets a bar your next round has to clear. "
     "That is why so many companies raise once and then quietly die. "
     "The terms matter more than the headline number, every single time.", True),

    ("Right, so, um, moving on, we had a few, you know, other items, "
     "and I think we sort of covered most of them already anyway.", False),

    ("Here is why most people never get good at anything hard. "
     "They quit in the exact window where progress stops being visible. "
     "I spent four brutal years stuck in that window myself. "
     "The work was identical on day one and day eight hundred. "
     "Turns out the only difference was that I stopped checking for proof. "
     "That is the whole secret, and it is genuinely that stupid.", True),

    ("And then, like, we talked about the other thing, and, um, "
     "yeah it was fine, it was totally fine, nothing really to report.", False),

    ("Stop optimising your morning routine. It is not the problem. "
     "I tracked every hour of my day for six months straight. "
     "The mornings were never where the time went. "
     "Three afternoon meetings were eating nineteen hours a week. "
     "Nobody talks about that because fixing meetings is boring. "
     "The answer is always in the boring part of your calendar.", True),

    ("So yeah, um, that is about it for today, thanks again everybody, "
     "and we will, you know, catch up on all of this next week I think.", False),
]


def make_words(words_per_second: float = 2.9, gap_between_blocks: float = 0.9) -> List[Word]:
    """Time the script out into a flat word stream."""
    out: List[Word] = []
    cursor = 2.0  # a little silence before anyone speaks
    step = 1.0 / words_per_second
    for text, _ in SCRIPT:
        for token in text.split():
            out.append(Word(text=token, start=cursor, end=cursor + step * 0.92))
            cursor += step
        cursor += gap_between_blocks
    return out


def script_duration() -> float:
    words = make_words()
    return words[-1].end + 2.0 if words else 0.0


def write_srt(path: Path) -> Path:
    """Write the script out as an SRT sidecar."""
    from clipper.subtitle_io import words_to_srt

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(words_to_srt(make_words()), encoding="utf-8")
    return path


def make_video(path: Path, duration: float, ffmpeg: str = "ffmpeg") -> Path:
    """Render a landscape test video with a moving subject and varying audio.

    The subject moves so reframing has something to track; the audio level
    swings so the energy analysis has a real distribution to work against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg, "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=#101820:s=1280x720:r=25:d={duration:.2f}",
        "-f", "lavfi", "-i", f"testsrc2=s=260x260:r=25:d={duration:.2f}",
        "-f", "lavfi", "-i",
        # Amplitude swings on a slow cycle so some windows are "louder".
        f"sine=frequency=220:duration={duration:.2f},"
        f"volume='0.15+0.85*abs(sin(t/7))':eval=frame",
        "-filter_complex",
        # A slow sinusoidal pan keeps the subject off-centre most of the time.
        "[0:v][1:v]overlay=x='510+480*sin(t/9)':y=230[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)
    return path
