"""A small desktop window for people who would rather not use a terminal.

Built on tkinter, which ships with Python on Windows and macOS, so running the
app needs nothing installed beyond the clipper's own dependencies.

The Tk-specific part is deliberately thin: every decision the window makes —
validating the form, turning it into a :class:`ClipperConfig`, choosing where
clips land — lives in plain functions above the widget code so it can be
tested without a display.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from clipper.config import PRESETS, ClipperConfig

#: Every clip is rendered at this size, whatever the source was.
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

#: Pulled by default so the 9:16 crop is a true 1080p rather than an upscale.
#: A 1080p source only leaves a 607px-wide crop; 4K leaves 1215px.
SHARP_SOURCE_HEIGHT = 2160
NORMAL_SOURCE_HEIGHT = 1080

PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "reels": "Instagram Reels",
    "shorts": "YouTube Shorts",
    "square": "Square (1:1)",
}


def default_output_dir() -> Path:
    """Somewhere obvious to put finished clips."""
    home = Path.home()
    for candidate in ("Videos", "Movies", "Downloads"):
        folder = home / candidate
        if folder.is_dir():
            return folder / "ViralClips"
    return home / "ViralClips"


@dataclass
class FormValues:
    """Exactly what the window collects, before any interpretation."""

    url: str = ""
    clips: int = 10
    min_seconds: int = 15
    max_seconds: int = 60
    output_dir: str = ""
    platform: str = "tiktok"
    burn_captions: bool = True
    sharp_source: bool = True


def validate(form: FormValues) -> List[str]:
    """Human-readable problems with the form, empty when it is good to go."""
    problems: List[str] = []
    url = form.url.strip()
    if not url:
        problems.append("Paste a YouTube link (or the path to a video file).")
    elif url.startswith(("http://", "https://")):
        pass
    elif not Path(url).expanduser().exists():
        problems.append(f"No file at {url}")

    if not 1 <= form.clips <= 20:
        problems.append("Number of clips must be between 1 and 20.")
    if form.min_seconds < 5:
        problems.append("The shortest clip must be at least 5 seconds.")
    if form.max_seconds > 180:
        problems.append("The longest clip must be 180 seconds or less.")
    if form.min_seconds >= form.max_seconds:
        problems.append("The shortest clip has to be shorter than the longest.")
    if not form.output_dir.strip():
        problems.append("Choose a folder to save the clips into.")
    if form.platform not in PRESETS:
        problems.append(f"Unknown platform {form.platform!r}.")
    return problems


def build_config(form: FormValues, workspace: Optional[Path] = None) -> ClipperConfig:
    """Turn the form into a validated :class:`ClipperConfig`.

    The output size is not taken from the form: 1080x1920 is fixed, which is
    what every one of these presets already renders.
    """
    output = Path(form.output_dir.strip()).expanduser()
    return ClipperConfig(
        platform=form.platform,
        workspace=workspace or (output / ".work"),
        output_dir=output,
        max_clips=int(form.clips),
        min_duration=float(form.min_seconds),
        max_duration=float(form.max_seconds),
        burn_subtitles=bool(form.burn_captions),
        source_max_height=(
            SHARP_SOURCE_HEIGHT if form.sharp_source else NORMAL_SOURCE_HEIGHT
        ),
    ).validate()


def source_from(form: FormValues) -> str:
    return form.url.strip()


def open_folder(path: Path) -> bool:
    """Reveal a folder in the platform's file manager."""
    path = Path(path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def launch(on_run: Optional[Callable[..., Any]] = None) -> int:
    """Open the window. Returns a process exit code.

    ``on_run`` is the pipeline entry point, injectable so tests can drive the
    window without rendering video.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print(
            "The desktop window needs tkinter, which is missing from this Python.\n"
            "  Windows/macOS : reinstall Python from python.org (it is included)\n"
            "  Debian/Ubuntu : sudo apt install python3-tk\n"
            "\nYou can still use the command line:\n"
            '  python viral_clipper.py "<video url>" -n 10',
            file=sys.stderr,
        )
        return 1

    if on_run is None:
        from clipper.pipeline import run_pipeline as on_run  # type: ignore[assignment]

    BG = "#12141c"
    PANEL = "#1b1f2b"
    TEXT = "#e8ecf5"
    MUTED = "#8b93a7"
    ACCENT = "#ffe14d"

    root = tk.Tk()
    root.title("Viral Clipper")
    root.configure(bg=BG)
    root.minsize(560, 640)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - depends on the Tk build
        pass
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
    style.configure("Title.TLabel", foreground=TEXT, font=("Segoe UI", 17, "bold"))
    style.configure("TCheckbutton", background=BG, foreground=TEXT)
    style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT)
    style.configure("TSpinbox", fieldbackground=PANEL, foreground=TEXT)
    style.configure("TCombobox", fieldbackground=PANEL, foreground=TEXT)
    style.configure("Go.TButton", font=("Segoe UI", 11, "bold"), padding=10)
    style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL, borderwidth=0)

    outer = ttk.Frame(root, padding=22)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(0, weight=1)

    row = 0
    ttk.Label(outer, text="Viral Clipper", style="Title.TLabel").grid(
        row=row, column=0, sticky="w"
    )
    row += 1
    ttk.Label(
        outer,
        text=f"Paste a link, press Make clips. Every clip comes out "
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.",
        style="Muted.TLabel",
    ).grid(row=row, column=0, sticky="w", pady=(2, 16))

    # -- link ------------------------------------------------------------
    row += 1
    ttk.Label(outer, text="YouTube link").grid(row=row, column=0, sticky="w")
    row += 1
    url_var = tk.StringVar()
    url_entry = ttk.Entry(outer, textvariable=url_var, font=("Segoe UI", 10))
    url_entry.grid(row=row, column=0, sticky="ew", pady=(3, 14))
    url_entry.focus_set()

    # -- numbers ---------------------------------------------------------
    row += 1
    numbers = ttk.Frame(outer)
    numbers.grid(row=row, column=0, sticky="ew", pady=(0, 14))

    clips_var = tk.IntVar(value=10)
    ttk.Label(numbers, text="Clips to make").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(numbers, from_=1, to=20, textvariable=clips_var, width=6).grid(
        row=1, column=0, sticky="w", pady=(3, 0)
    )

    ttk.Label(numbers, text="").grid(row=0, column=1, padx=14)

    min_var = tk.IntVar(value=15)
    max_var = tk.IntVar(value=60)
    ttk.Label(numbers, text="Clip length (seconds)").grid(row=0, column=2, sticky="w")
    length = ttk.Frame(numbers)
    length.grid(row=1, column=2, sticky="w", pady=(3, 0))
    ttk.Spinbox(length, from_=5, to=180, textvariable=min_var, width=6).pack(side="left")
    ttk.Label(length, text="  to  ", style="Muted.TLabel").pack(side="left")
    ttk.Spinbox(length, from_=5, to=180, textvariable=max_var, width=6).pack(side="left")

    # -- destination -----------------------------------------------------
    row += 1
    ttk.Label(outer, text="Save clips to").grid(row=row, column=0, sticky="w")
    row += 1
    destination = ttk.Frame(outer)
    destination.grid(row=row, column=0, sticky="ew", pady=(3, 14))
    destination.columnconfigure(0, weight=1)
    out_var = tk.StringVar(value=str(default_output_dir()))
    ttk.Entry(destination, textvariable=out_var).grid(row=0, column=0, sticky="ew")

    def browse() -> None:
        chosen = filedialog.askdirectory(initialdir=str(Path(out_var.get()).parent))
        if chosen:
            out_var.set(chosen)

    ttk.Button(destination, text="Browse…", command=browse).grid(
        row=0, column=1, padx=(8, 0)
    )

    # -- options ---------------------------------------------------------
    row += 1
    options = ttk.Frame(outer)
    options.grid(row=row, column=0, sticky="ew", pady=(0, 6))
    ttk.Label(options, text="Made for").pack(side="left")
    platform_var = tk.StringVar(value=PLATFORM_LABELS["tiktok"])
    ttk.Combobox(
        options,
        textvariable=platform_var,
        values=[PLATFORM_LABELS[p] for p in ("tiktok", "reels", "shorts", "square")],
        state="readonly",
        width=17,
    ).pack(side="left", padx=(8, 18))
    captions_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(options, text="Burn in captions", variable=captions_var).pack(
        side="left"
    )

    row += 1
    sharp_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        outer,
        text="Download the sharpest source available (slower, but a true 1080p crop)",
        variable=sharp_var,
    ).grid(row=row, column=0, sticky="w", pady=(0, 2))
    row += 1
    ttk.Label(
        outer,
        text="A 1080p source only leaves a 607px-wide vertical crop, which has to be "
        "upscaled.\nPulling 4K when it exists is what makes the clips actually look 1080p.",
        style="Muted.TLabel",
        justify="left",
    ).grid(row=row, column=0, sticky="w", pady=(0, 16))

    # -- action ----------------------------------------------------------
    row += 1
    go_button = ttk.Button(outer, text="Make clips", style="Go.TButton")
    go_button.grid(row=row, column=0, sticky="ew")

    row += 1
    progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
    progress.grid(row=row, column=0, sticky="ew", pady=(16, 6))

    row += 1
    status_var = tk.StringVar(value="Ready.")
    ttk.Label(outer, textvariable=status_var, style="Muted.TLabel").grid(
        row=row, column=0, sticky="w"
    )

    row += 1
    log = tk.Text(
        outer, height=11, bg=PANEL, fg=MUTED, relief="flat",
        font=("Consolas", 9), wrap="word", padx=10, pady=8,
    )
    log.grid(row=row, column=0, sticky="nsew", pady=(10, 10))
    log.configure(state="disabled")
    outer.rowconfigure(row, weight=1)

    row += 1
    open_button = ttk.Button(
        outer, text="Open the clips folder",
        command=lambda: open_folder(Path(out_var.get())),
        state="disabled",
    )
    open_button.grid(row=row, column=0, sticky="ew")

    # -- wiring ----------------------------------------------------------
    events: "queue.Queue[tuple]" = queue.Queue()
    running = {"active": False}

    def say(line: str) -> None:
        log.configure(state="normal")
        log.insert("end", line + "\n")
        log.see("end")
        log.configure(state="disabled")

    def current_form() -> FormValues:
        labels = {v: k for k, v in PLATFORM_LABELS.items()}
        return FormValues(
            url=url_var.get(),
            clips=clips_var.get(),
            min_seconds=min_var.get(),
            max_seconds=max_var.get(),
            output_dir=out_var.get(),
            platform=labels.get(platform_var.get(), "tiktok"),
            burn_captions=captions_var.get(),
            sharp_source=sharp_var.get(),
        )

    def worker(form: FormValues, config: ClipperConfig) -> None:
        try:
            result = on_run(
                source_from(form),
                config,
                progress=lambda message, fraction: events.put(
                    ("progress", message, fraction)
                ),
            )
            events.put(("done", result))
        except Exception as exc:  # every failure has to reach the window
            events.put(("error", f"{type(exc).__name__}: {exc}"))

    def pump() -> None:
        try:
            while True:
                event = events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, message, fraction = event
                    progress["value"] = fraction * 100
                    status_var.set(message)
                elif kind == "done":
                    finish(event[1])
                elif kind == "error":
                    fail(event[1])
        except queue.Empty:
            pass
        root.after(120, pump)

    def finish(result: Any) -> None:
        running["active"] = False
        go_button.configure(state="normal", text="Make clips")
        open_button.configure(state="normal")
        progress["value"] = 100
        clips = getattr(result, "clips", [])
        status_var.set(f"Done — {len(clips)} clips in {result.output_dir}")
        say("")
        for clip in clips:
            minutes, seconds = divmod(int(clip.start), 60)
            say(
                f"{clip.index:>2}. {minutes:>3}:{seconds:02d}  {clip.duration:>4.0f}s  "
                f"score {clip.score:>3.0f}/100  {clip.copy.title[:44]}"
            )
        say("")
        say(f"Saved to {result.output_dir}")

    def fail(message: str) -> None:
        running["active"] = False
        go_button.configure(state="normal", text="Make clips")
        status_var.set("Failed.")
        say("")
        say(message)
        messagebox.showerror("Viral Clipper", message)

    def start() -> None:
        if running["active"]:
            return
        form = current_form()
        problems = validate(form)
        if problems:
            messagebox.showwarning("Check the form", "\n".join(problems))
            return
        try:
            config = build_config(form)
        except ValueError as exc:
            messagebox.showwarning("Check the form", str(exc))
            return

        Path(form.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
        running["active"] = True
        go_button.configure(state="disabled", text="Working…")
        open_button.configure(state="disabled")
        progress["value"] = 0
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")
        say(f"Source : {source_from(form)}")
        say(f"Making : {form.clips} clips, {form.min_seconds}-{form.max_seconds}s each")
        say(f"Output : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} -> {form.output_dir}")
        say("")
        threading.Thread(target=worker, args=(form, config), daemon=True).start()

    go_button.configure(command=start)
    root.bind("<Return>", lambda _event: start())
    root.after(120, pump)
    root.mainloop()
    return 0
