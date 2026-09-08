# Viral Clipper

Point it at a long video — a podcast, a talk, a stream VOD — and it finds the
moments worth cutting, then renders each one as a ready-to-post vertical short
for TikTok, Instagram Reels or YouTube Shorts.

Every clip comes out as a 1080×1920 MP4 with the subject kept in frame,
word-by-word captions burned in, loudness matched to what the platforms expect,
plus a suggested title, caption and hashtags.

```bash
clipper clip https://www.youtube.com/watch?v=... --platform tiktok --max-clips 10
clipper serve                      # same thing, with a web UI on :8000
```

**No install? Use the notebook.** `Viral_Clipper_Colab.ipynb` is a single self-contained
file — open it in Google Colab, paste a link, run the cells. The whole tool is embedded
in it, so there is nothing to clone and nothing else to download. Regenerate it after
changing the package with `python build_colab_notebook.py`.

---

## What it actually does

```
 URL or file
     │
     ├─ ingest ......... yt-dlp download (captions too), or a local file
     ├─ transcribe ..... faster-whisper → word-level timestamps
     │                   (falls back to the source's own captions)
     ├─ segment ........ words → sentences → every window that fits the platform
     ├─ analyse ........ per-window loudness dynamics from the audio
     ├─ score .......... 13 weighted signals → a 0–100 virality score
     ├─ select ......... suppress overlapping and near-duplicate moments
     ├─ reframe ........ track the subject, plan a smoothed 9:16 crop path
     ├─ caption ........ word-highlight ASS subtitles
     └─ render ......... one ffmpeg pass → MP4 + thumbnail + SRT + metadata
```

### Finding the moment

Clip boundaries are the whole game. A short that starts mid-sentence reads as
broken no matter how good the content is, so candidates are only ever cut on
sentence boundaries, and every candidate window is scored on 13 signals:

| Signal | What it measures |
| --- | --- |
| `hook` | does the first line stop the scroll (pattern-matched openers, questions, numbers) |
| `completeness` | starts on a whole thought, ends on a finished one |
| `topic_start` | begins after a real pause, i.e. where a new thought actually begins |
| `payoff` | resolves what it opened, rather than trailing off |
| `curiosity` | open loops — "here's the thing", teases, questions |
| `emotion` | density of high-arousal vocabulary |
| `quotability` | short, absolute, second-person lines that screenshot well |
| `audio_energy` | loudness peaks and dynamics vs. the rest of the video |
| `pace` | words per second inside the energetic-but-clear band |
| `information_density` | content words vs. filler |
| `list_structure` | "three reasons", "first… second…" |
| `laughter` | `[laughter]`, applause and reaction markers |
| `duration_fit` | length vs. the platform's sweet spot |

Then multiplicative penalties for the things that sink a clip regardless of how
well it otherwise scores: a dangling `"And so…"` opener, an unfinished ending,
filler density, verbal repetition, sparse speech, and the housekeeping at the
top of most videos.

Every score keeps its full breakdown, so the CLI and the UI can both tell you
*why* a moment was picked:

```
[1] 00:00:22 - 00:00:47  (25s)   score 70/100
    Nobody tells you the biggest mistake founders make when raising money
    strongest: hook, completeness, topic_start
```

### Keeping the subject in frame

A static centre crop throws away half of most landscape footage. The reframer
samples the clip at 4 fps, estimates where the subject is on each sample (faces
via OpenCV when it's installed, motion-energy centroid otherwise), and plans a
crop path that ffmpeg follows through `sendcmd`.

The path is smoothed forward *and* backward. A plain exponential filter always
trails a moving subject and leaves them drifting toward the edge of frame;
because reframing happens offline the whole path is already known, so filtering
in both directions removes that lag entirely. A deadzone stops small wobbles
from moving the camera at all.

Four framing modes: `auto` (track the subject), `center` (static crop),
`blur` (source letterboxed over a blurred fill of itself), `fit` (black bars).

### Captions

Most of the feed is watched muted, so captions are not optional. Output is ASS
with one word highlighted at a time, tiled edge-to-edge so the text never blinks
between words, positioned clear of each platform's UI overlay. Three styles
(`punch`, `clean`, `minimal`), and an `.srt` is written alongside every clip for
uploading separately.

---

## Install

The only hard requirement is **ffmpeg**.

```bash
# 1. ffmpeg
apt install ffmpeg          # or: brew install ffmpeg
                            # or: pip install imageio-ffmpeg  (bundles a static build)

# 2. the clipper
pip install -e '.[full]'    # everything
pip install -e .            # core only — clips local files that already have captions
```

Optional extras, each of which degrades gracefully if absent:

| Extra | Unlocks |
| --- | --- |
| `yt-dlp` | clipping straight from a URL |
| `faster-whisper` | transcription (otherwise: bring an `.srt`/`.vtt`) |
| `opencv-python-headless` | face-aware reframing (otherwise: motion tracking) |
| `numpy` | ~20× faster subject tracking (otherwise: a pure-Python path with identical output) |
| `fastapi` + `uvicorn` | `clipper serve` |
| `anthropic` | `--llm` re-ranking |

`clipper doctor` reports exactly what you have and what each missing piece
would buy you.

---

## Usage

### CLI

```bash
# The basics
clipper clip talk.mp4
clipper talk.mp4                              # `clip` is the default command

# Platform presets change size, duration, caption placement and scoring weights
clipper clip talk.mp4 --platform reels
clipper clip talk.mp4 --platform shorts --max-clips 8

# See what it would pick, without spending render time
clipper clip talk.mp4 --dry-run

# Bring your own transcript (no speech model needed)
clipper clip talk.mp4 --transcript talk.srt

# Framing and caption styling
clipper clip talk.mp4 --layout blur --caption-style clean --no-uppercase
clipper clip talk.mp4 --highlight-color '#00E5FF'

# Machine-readable output
clipper clip talk.mp4 --json > clips.json
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--platform` | `tiktok` | `tiktok`, `reels`, `shorts`, `square` |
| `--max-clips` | `5` | how many to keep |
| `--min-duration` / `--max-duration` | preset | override the platform window |
| `--min-score` | `0` | drop anything weaker |
| `--layout` | `auto` | `auto`, `center`, `blur`, `fit` |
| `--caption-style` | `punch` | `punch`, `clean`, `minimal` |
| `--fps` | preset (30) | output frame rate, 15-60 |
| `--model` | `small` | whisper size: `tiny` … `large-v3` |
| `--transcript` | — | reuse an existing `.srt`/`.vtt`/`.json` |
| `--dry-run` | off | score and plan only |
| `--llm` | off | re-rank the shortlist with Claude |

### Web UI

```bash
clipper serve --port 8000
```

Submit a URL or upload a file, watch progress live, then preview each clip in
the browser with its score breakdown, copy the suggested caption, and download.
Bound to `127.0.0.1` by default — it runs ffmpeg on whatever you point it at, so
don't expose it to a network you don't control.

### Python

```python
from clipper.config import ClipperConfig
from clipper.pipeline import run_pipeline

result = run_pipeline("talk.mp4", ClipperConfig(platform="reels", max_clips=3))
for clip in result.clips:
    print(clip.score, clip.copy.title, clip.video_path)
```

Each stage is usable on its own — `clipper.segment`, `clipper.scoring`,
`clipper.reframe`, `clipper.subtitles` and `clipper.render` have no hidden
dependencies on each other beyond the dataclasses they pass around.

### HTTP API

| Method | Path | |
| --- | --- | --- |
| `POST` | `/api/jobs` | start a job |
| `GET` | `/api/jobs` | list jobs |
| `GET` | `/api/jobs/{id}` | status, progress and clips |
| `DELETE` | `/api/jobs/{id}` | cancel / forget |
| `GET` | `/api/jobs/{id}/clips/{n}/{video\|thumbnail\|subtitles}` | download an artefact |
| `POST` | `/api/upload` | upload a source file |
| `GET` | `/api/options`, `/api/health` | form options, dependency status |

---

## Output

```
clips/talk-a1b2c3/
├── clip-01-nobody-tells-you-the-biggest-mistake.mp4   1080×1920, captions burned in
├── clip-01-nobody-tells-you-the-biggest-mistake.jpg   thumbnail
├── clip-01-nobody-tells-you-the-biggest-mistake.srt   captions, for uploading separately
├── clip-01-nobody-tells-you-the-biggest-mistake.ass   the styled captions that were burned in
├── clip-02-…
└── manifest.json                                      every clip's timing, score breakdown and copy
```

## Tuning

Scoring weights live in `clipper/config.py` (`DEFAULT_WEIGHTS`, plus
per-platform `weight_overrides`); the patterns and word lists they run on are
all in `clipper/lexicon.py`, kept as plain data so they can be tuned for a niche
or swapped for another language without touching the scoring logic.

`--dry-run` is the fast loop for this: it scores everything and prints the
picks without spending render time.

### Optional: LLM re-ranking

`--llm` sends the top candidates to Claude for a second opinion on how each
would land as a standalone short, and blends that with the heuristic score
(`--llm-weight`, default 0.5). Needs `ANTHROPIC_API_KEY`. If the key or the SDK
is missing it logs a note and carries on with heuristic scores — it never fails
a job.

## Tests

```bash
pip install -e '.[dev]'
pytest                              # the whole suite
pytest tests/test_scoring.py -q     # just the scoring logic, no ffmpeg needed
```

Tests that need ffmpeg, fastapi or numpy skip themselves when those are absent.
The end-to-end tests build a synthetic video whose transcript hides three
strong moments among filler blocks, then assert the pipeline finds all three
and renders playable 1080×1920 MP4s.

## Notes

- Download only what you have the rights to use, and follow the terms of the
  site you are downloading from.
- Render time is dominated by transcription on first run; transcripts are cached
  in the workspace, so re-running with different scoring or framing is fast.
- A source with no speech produces no clips: every signal here is built on what
  is said.

## Licence

MIT.
