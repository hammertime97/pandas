#!/usr/bin/env python3
"""Generate a self-contained Google Colab notebook.

The notebook embeds the whole ``clipper`` package as one gzipped blob, so it
is a single file you can drop into Colab with nothing else to download and no
repository to clone. Run this script after changing the package to regenerate
it:

    python build_colab_notebook.py
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "clipper"
OUTPUT = ROOT / "Viral_Clipper_Colab.ipynb"

#: The web server is not useful inside Colab, so it is left out of the bundle.
EXCLUDE = {"clipper/server", "clipper/cli.py", "clipper/__main__.py"}

#: Marks the notebook cell that carries the embedded package.
PACKAGE_MARKER = "PACKAGE_BLOB"


def should_include(path: Path, exclude=None) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    excluded = EXCLUDE if exclude is None else exclude
    return not any(relative == e or relative.startswith(e + "/") for e in excluded)


def build_blob(exclude=None) -> str:
    """Pack the package into a deterministic base64 tar.gz string.

    ``exclude`` defaults to :data:`EXCLUDE` (what the notebook leaves out); the
    local runner passes a smaller set because it needs the CLI.
    """
    files = sorted(p for p in PACKAGE.rglob("*.py") if should_include(p, exclude))
    if not files:
        raise SystemExit("no package files found")

    raw = io.BytesIO()
    # mtime=0 and a fixed sort order keep the blob stable across rebuilds.
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for path in files:
                info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                data = path.read_bytes()
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    encoded = base64.b64encode(raw.getvalue()).decode("ascii")
    print(f"packed {len(files)} modules -> {len(encoded) / 1024:.0f} KB of base64")
    return encoded


def wrap(text: str, width: int = 96) -> List[str]:
    """Split a long string into notebook-friendly source lines."""
    return [text[i : i + width] + "\n" for i in range(0, len(text), width)]


def markdown(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str, **metadata: Any) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

INTRO = """
# 🎬 Viral Clipper

Paste a YouTube link, press play on each cell, get **10 vertical clips** ready for
TikTok / Reels / Shorts — each one 1080×1920 MP4 with the speaker kept in frame and
word-by-word captions burned in.

**How to use it**

1. `Runtime → Change runtime type → T4 GPU` (optional, but transcription is ~10× faster)
2. Run **Step 1** and **Step 2** once — they take a couple of minutes
3. Put your link in **Step 3** and run it
4. Run **Step 4** to watch the clips, **Step 5** to download them

Nothing else to install and no repository to clone — the whole tool is embedded in
this notebook.
"""

SETUP = r"""
#@title Step 1 · Install (run once, ~2 minutes) { display-mode: "form" }
import subprocess, sys, shutil

def sh(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:]); print(result.stderr[-2000:])
    return result.returncode == 0

print("Installing yt-dlp (downloader)…")
sh(f"{sys.executable} -m pip install -q --upgrade yt-dlp")

print("Installing faster-whisper (transcription)…")
sh(f"{sys.executable} -m pip install -q faster-whisper")

if shutil.which("ffmpeg") is None:
    print("Installing ffmpeg…")
    sh("apt-get -qq update && apt-get -qq install -y ffmpeg")

# OpenCV gives face-aware reframing. Importing it is not proof it works —
# Colab sometimes ships a cv2 whose native extension never loaded — so check
# for the attribute we actually call.
try:
    import cv2
    faces = hasattr(cv2, "CascadeClassifier")
except Exception:
    faces = False

try:
    import torch
    gpu = torch.cuda.is_available()
except Exception:
    gpu = False

print()
print("ffmpeg          :", shutil.which("ffmpeg") or "MISSING")
print("GPU             :", "yes — transcription will be fast" if gpu else "no  — CPU, slower but fine")
print("Face tracking   :", "yes" if faces else "no — using motion tracking (works fine)")
print("\nDone. Run Step 2.")
"""

UNPACK_TEMPLATE = r'''
#@title Step 2 · Load the clipper (run once) {{ display-mode: "form" }}
import base64, gzip, io, sys, tarfile
from pathlib import Path

# The whole tool, packed into this notebook. Nothing is downloaded.
PACKAGE_BLOB = (
{blob}
)

target = Path("/content/viral_clipper")
target.mkdir(parents=True, exist_ok=True)
with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(PACKAGE_BLOB))) as gz:
    with tarfile.open(fileobj=gz, mode="r") as tar:
        tar.extractall(target)

if str(target) not in sys.path:
    sys.path.insert(0, str(target))

for module in [m for m in list(sys.modules) if m.split(".")[0] == "clipper"]:
    del sys.modules[module]          # allow re-running this cell cleanly

import clipper
from clipper.config import PRESETS
print(f"Viral Clipper {{clipper.__version__}} loaded.")
print("Platforms:", ", ".join(sorted(PRESETS)))
print("\nRun Step 3.")
'''

UPLOADER = r'''
#@title Upload a cookies.txt or a video file (only if YouTube blocks you) { display-mode: "form" }
#@markdown Click **Choose Files** below. Two kinds of file are understood:
#@markdown
#@markdown * **`cookies.txt`** — saved to `/content/cookies.txt`, and Step 3 picks it up
#@markdown   on its own. Get one with the *Get cookies.txt LOCALLY* Chrome extension:
#@markdown   install it, open youtube.com while signed in, click the extension, Export.
#@markdown * **a video** (`.mp4`, `.mov`, `.mkv`, `.webm`) — the path is printed; paste it
#@markdown   into `UPLOADED_FILE` in Step 3.
#@markdown
#@markdown Large videos upload slowly through the browser. If yours is over ~200 MB,
#@markdown the cookies route is much quicker.

import shutil
from pathlib import Path

try:
    from google.colab import files
except ImportError:
    raise SystemExit("This cell only works inside Google Colab.")

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a"}

for name in files.upload():
    source = Path(name)
    if source.suffix.lower() == ".txt" or "cookie" in source.stem.lower():
        shutil.move(str(source), "/content/cookies.txt")
        size = Path("/content/cookies.txt").stat().st_size
        if size < 100:
            print(f"⚠️  {name} is only {size} bytes — that looks empty. Re-export it.")
        else:
            print(f"✅ Cookies saved ({size / 1024:.0f} KB). "
                  "Just run Step 3 — it will find them automatically.")
    elif source.suffix.lower() in VIDEO_SUFFIXES:
        target = Path("/content") / source.name
        if source.resolve() != target.resolve():
            shutil.move(str(source), target)
        print(f"✅ Video saved. Paste this into UPLOADED_FILE in Step 3:\n   {target}")
    else:
        print(f"⚠️  Not sure what to do with {name} — expected cookies.txt or a video.")
'''

RUN = r'''
#@title Step 3 · Your video → clips { display-mode: "form", run: "auto" }

#@markdown ### Paste your link
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  #@param {type:"string"}

#@markdown ### Settings
HOW_MANY_CLIPS = 10  #@param {type:"slider", min:1, max:20, step:1}
PLATFORM = "tiktok"  #@param ["tiktok", "reels", "shorts", "square"]
FRAMES_PER_SECOND = "30"  #@param ["30", "60"]
SHORTEST_CLIP_SECONDS = 15  #@param {type:"slider", min:5, max:90, step:5}
LONGEST_CLIP_SECONDS = 60  #@param {type:"slider", min:15, max:180, step:5}
FRAMING = "auto"  #@param ["auto", "center", "blur", "fit"]
CAPTION_STYLE = "punch"  #@param ["punch", "clean", "minimal"]
BURN_CAPTIONS = True  #@param {type:"boolean"}
TRANSCRIPTION_QUALITY = "small"  #@param ["tiny", "base", "small", "medium", "large-v3"]

#@markdown ---
#@markdown ### If YouTube blocks the download
#@markdown Colab runs on Google data-centre IPs, which YouTube often challenges with
#@markdown *"Sign in to confirm you're not a bot"* — even for a video that downloads
#@markdown fine on your own machine. The clipper retries several player clients
#@markdown automatically. If it still fails, use **either** of these:
#@markdown
#@markdown **A · Upload the video** (always works). Download it yourself, drag it into
#@markdown the file browser on the left, and put its path here:
UPLOADED_FILE = ""  #@param {type:"string"}
#@markdown **B · Use your cookies.** Export them with a *Get cookies.txt* browser
#@markdown extension while logged into YouTube, upload the file, and put its path here:
COOKIES_FILE = ""  #@param {type:"string"}

# ---------------------------------------------------------------------------
import logging, time
from pathlib import Path
from clipper.config import ClipperConfig
from clipper.errors import ClipperError
from clipper.pipeline import run_pipeline

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)

source = UPLOADED_FILE.strip() or YOUTUBE_URL.strip()
if not source:
    raise SystemExit("Paste a YouTube link (or an uploaded file path) first.")
if source.startswith("http") and "dQw4w9WgXcQ" in source:
    print("⚠️  That is still the placeholder link — replace it with your own video.\n")

cookies = COOKIES_FILE.strip()
if not cookies and Path("/content/cookies.txt").exists():
    cookies = "/content/cookies.txt"      # dropped in by the uploader cell
if cookies and not Path(cookies).exists():
    raise SystemExit(f"No cookies file at {cookies}. Upload it, or clear the field.")
if cookies:
    # resolve_source takes cookies_file as a keyword, so bind it for this run.
    import functools
    import clipper.pipeline as _pipeline
    _pipeline.resolve_source = functools.partial(
        _pipeline.resolve_source, cookies_file=Path(cookies)
    )
    print(f"Using cookies from {cookies}\n")

config = ClipperConfig(
    platform=PLATFORM,
    workspace=Path("/content/workspace"),
    output_dir=Path("/content/clips"),
    max_clips=HOW_MANY_CLIPS,
    min_duration=float(SHORTEST_CLIP_SECONDS),
    max_duration=float(LONGEST_CLIP_SECONDS),
    fps=int(FRAMES_PER_SECOND),
    layout=FRAMING,
    caption_style=CAPTION_STYLE,
    burn_subtitles=BURN_CAPTIONS,
    whisper_model=TRANSCRIPTION_QUALITY,
).validate()

started = time.time()
state = {"line": ""}

def show_progress(message, fraction):
    filled = int(30 * fraction)
    line = f"\r[{'█' * filled}{'░' * (30 - filled)}] {fraction * 100:3.0f}%  {message[:42]:<42}"
    if line != state["line"]:
        print(line, end="", flush=True)
        state["line"] = line

try:
    RESULT = run_pipeline(source, config, progress=show_progress)
except ClipperError as exc:
    print("\n\n❌", exc)
    raise SystemExit(str(exc)) from None
except KeyboardInterrupt:
    print("\n\nStopped.")
    raise SystemExit("interrupted") from None

print(f"\n\n✅ {len(RESULT.clips)} clips in {time.time() - started:.0f}s → {RESULT.output_dir}\n")
print(f"Scanned {RESULT.stats['candidates']} possible moments "
      f"from {RESULT.stats['source_duration'] / 60:.0f} minutes of video.\n")
for clip in RESULT.clips:
    minutes, seconds = divmod(int(clip.start), 60)
    print(f"  {clip.index:>2}. {minutes:>3}:{seconds:02d}  {clip.duration:>4.0f}s  "
          f"score {clip.score:>3.0f}/100   {clip.copy.title[:54]}")
'''

DOWNLOAD_NOTE = """
### A note on YouTube downloads

YouTube challenges requests coming from Google's own data-centre IPs — which is what
Colab runs on — with *"Sign in to confirm you're not a bot"*. A link that downloads
fine on your laptop can therefore fail here.

The clipper retries nine YouTube player clients automatically, which clears the
challenge much of the time. When it does not — you will see every retry fail with the
same "not a bot" message — **cookies are the fix**, and the cell below makes that one
click:

1. Install the *Get cookies.txt LOCALLY* extension in Chrome
2. Open youtube.com while signed in, click the extension, **Export**
3. Run the cell below and upload the file it saved
4. Run Step 3 again — it finds the cookies on its own

Uploading the video itself always works too, and the same cell accepts one.

This is a YouTube restriction rather than something the clipper can fix outright.
"""

PREVIEW = r'''
#@title Step 4 · Watch the clips { display-mode: "form", run: "auto" }
#@markdown The grid below is instant. Videos are heavy — a 30s clip is several MB, and
#@markdown embedding ten of them at once puts ~85 MB into this page and makes the tab
#@markdown crawl. So pick one number at a time to play full size.
PLAY_CLIP = 1  #@param {type:"slider", min:1, max:20, step:1}

import base64
from pathlib import Path
from IPython.display import HTML, display

clips = [c for c in RESULT.clips if c.video_path]
if not clips:
    print("No rendered clips to show — run Step 3 first.")
else:
    def data_uri(path, mime):
        return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()

    # Thumbnails are ~65 KB each, so the whole grid costs well under a megabyte.
    cards = []
    for clip in clips:
        minutes, seconds = divmod(int(clip.start), 60)
        poster = (
            f'<img src="{data_uri(clip.thumbnail_path, "image/jpeg")}" '
            'style="width:100%;aspect-ratio:9/16;object-fit:cover;display:block">'
            if clip.thumbnail_path
            else '<div style="width:100%;aspect-ratio:9/16;background:#000"></div>'
        )
        highlight = "#ffe14d" if clip.index == PLAY_CLIP else "#262c3d"
        cards.append(f"""
          <div style="width:170px;background:#141824;border:2px solid {highlight};
                      border-radius:10px;overflow:hidden;color:#e8ecf5;
                      font-family:system-ui,sans-serif">
            {poster}
            <div style="padding:9px">
              <div style="font-size:17px;font-weight:700;color:#ffe14d">
                {clip.index}. {clip.score:.0f}<span style="font-size:10px;color:#8b93a7;
                     font-weight:400">/100</span>
                <span style="float:right;font-size:10px;color:#8b93a7;line-height:22px">
                  {minutes}:{seconds:02d}·{clip.duration:.0f}s</span></div>
              <div style="font-size:11px;line-height:1.35;margin-top:4px">
                {clip.copy.title[:70]}</div>
            </div>
          </div>""")

    display(HTML(
        "<div style='display:flex;flex-wrap:wrap;gap:11px;background:#0b0d12;padding:14px'>"
        + "".join(cards) + "</div>"
    ))

    chosen = next((c for c in clips if c.index == PLAY_CLIP), None)
    if chosen is None:
        print(f"\nNo clip {PLAY_CLIP} — this run produced {len(clips)}. "
              "Move the slider into range.")
    else:
        size = Path(chosen.video_path).stat().st_size / 1e6
        print(f"\nPlaying clip {chosen.index} ({size:.1f} MB) — "
              "move the slider to watch another.")
        display(HTML(f"""
          <div style="max-width:290px;font-family:system-ui,sans-serif;color:#e8ecf5">
            <video src="{data_uri(chosen.video_path, "video/mp4")}" controls playsinline
                   style="width:100%;aspect-ratio:9/16;background:#000;border-radius:10px"></video>
            <div style="font-weight:600;margin-top:8px">{chosen.copy.title}</div>
            <div style="font-size:12px;color:#6c8cff;word-break:break-word;margin-top:4px">
              {" ".join(chosen.copy.hashtags)}</div>
          </div>"""))
'''

CAPTIONS = r'''
#@title Step 5 · Copy the captions { display-mode: "form" }
for clip in RESULT.clips:
    print("=" * 70)
    print(f"CLIP {clip.index}  ·  score {clip.score:.0f}/100  ·  {clip.duration:.0f}s")
    print("=" * 70)
    print(clip.copy.caption)
    print()
'''

DOWNLOAD = r'''
#@title Step 6 · Download every clip as a zip { display-mode: "form" }
import shutil
from pathlib import Path

archive = shutil.make_archive("/content/viral_clips", "zip", RESULT.output_dir)
size = Path(archive).stat().st_size / 1e6
print(f"{archive}  ({size:.1f} MB)")

try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print("Not running in Colab — the zip is at the path above.")
'''

OUTRO = """
---

### What it picked, and why

Every candidate window is scored on 13 signals — how hard the opening line stops a
scroll, whether the clip starts and ends on a whole thought, whether it begins at a
real topic boundary, whether it pays off what it opened, loudness dynamics, pace,
filler density and more — then overlapping and near-duplicate moments are suppressed
so you get ten *different* moments rather than ten cuts of the same one.

`clip.breakdown.signals` on any clip holds the full per-signal breakdown if you want
to see the reasoning:

```python
for name, value in RESULT.clips[0].breakdown.signals.items():
    print(f"{name:22} {value:.2f}")
```

### Tuning it

- **Clips feel like they start mid-thought** → raise `SHORTEST_CLIP_SECONDS`
- **Speaker drifts out of frame** → try `FRAMING = "blur"`, which keeps the whole frame
- **Captions sit under the platform UI** → `PLATFORM = "reels"` places them higher
- **Transcript is inaccurate** → raise `TRANSCRIPTION_QUALITY` to `medium` or `large-v3`
- **Want different picks** → widen the duration range, or raise `HOW_MANY_CLIPS` and
  keep the best by eye

### Running it outside Colab

The same code works locally with Python 3.9+ and ffmpeg installed — the notebook just
unpacks it to `/content/viral_clipper`. In VS Code, open that folder and:

```python
from clipper.config import ClipperConfig
from clipper.pipeline import run_pipeline

result = run_pipeline("https://youtube.com/watch?v=...",
                      ClipperConfig(platform="tiktok", max_clips=10))
```
"""


def build_notebook() -> Dict[str, Any]:
    blob = build_blob()
    blob_literal = "".join(f'    "{line.rstrip()}"\n' for line in wrap(blob))
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            markdown(INTRO),
            code(SETUP, cellView="form"),
            code(UNPACK_TEMPLATE.format(blob=blob_literal), cellView="form"),
            markdown(DOWNLOAD_NOTE),
            code(UPLOADER, cellView="form"),
            code(RUN, cellView="form"),
            code(PREVIEW, cellView="form"),
            code(CAPTIONS, cellView="form"),
            code(DOWNLOAD, cellView="form"),
            markdown(OUTRO),
        ],
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
