"""The desktop window's logic, tested without a display.

Everything the window decides — validation, config building, where clips go —
lives in plain functions so it can be checked here. The widget code itself
needs a real Tk display and is not exercised.
"""

from pathlib import Path

import pytest

from clipper.config import PRESETS
from clipper.gui import (
    NORMAL_SOURCE_HEIGHT,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PLATFORM_LABELS,
    SHARP_SOURCE_HEIGHT,
    FormValues,
    build_config,
    default_output_dir,
    source_from,
    validate,
)


def good_form(tmp_path, **overrides):
    values = dict(
        url="https://www.youtube.com/watch?v=abc",
        clips=10,
        min_seconds=15,
        max_seconds=60,
        output_dir=str(tmp_path / "clips"),
    )
    values.update(overrides)
    return FormValues(**values)


def test_a_well_filled_form_has_no_complaints(tmp_path):
    assert validate(good_form(tmp_path)) == []


def test_a_local_file_path_is_accepted(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    assert validate(good_form(tmp_path, url=str(video))) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"url": ""}, "Paste a YouTube link"),
        ({"url": "/nope/missing.mp4"}, "No file at"),
        ({"clips": 0}, "between 1 and 20"),
        ({"clips": 99}, "between 1 and 20"),
        ({"min_seconds": 2}, "at least 5 seconds"),
        ({"max_seconds": 900}, "180 seconds or less"),
        ({"min_seconds": 60, "max_seconds": 30}, "shorter than the longest"),
        ({"output_dir": "  "}, "Choose a folder"),
        ({"platform": "myspace"}, "Unknown platform"),
    ],
)
def test_bad_input_is_explained_in_plain_words(tmp_path, overrides, expected):
    problems = validate(good_form(tmp_path, **overrides))
    assert any(expected in p for p in problems), problems


def test_the_form_becomes_a_valid_config(tmp_path):
    config = build_config(good_form(tmp_path, clips=7, min_seconds=20, max_seconds=45))
    assert config.max_clips == 7
    assert config.min_duration == 20.0 and config.max_duration == 45.0
    assert config.output_dir == tmp_path / "clips"
    # The working files stay out of the user's clips folder proper.
    assert config.workspace != config.output_dir


def test_output_is_always_1080_by_1920(tmp_path):
    """The size is fixed, whatever else the form says."""
    for platform in ("tiktok", "reels", "shorts"):
        preset = build_config(good_form(tmp_path, platform=platform)).preset()
        assert (preset.width, preset.height) == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        assert (preset.width, preset.height) == (1080, 1920)


def test_sharp_source_pulls_a_taller_download(tmp_path):
    sharp = build_config(good_form(tmp_path, sharp_source=True))
    normal = build_config(good_form(tmp_path, sharp_source=False))
    assert sharp.source_max_height == SHARP_SOURCE_HEIGHT == 2160
    assert normal.source_max_height == NORMAL_SOURCE_HEIGHT == 1080
    assert sharp.source_max_height > normal.source_max_height


def test_captions_toggle_reaches_the_config(tmp_path):
    assert build_config(good_form(tmp_path, burn_captions=False)).burn_subtitles is False
    assert build_config(good_form(tmp_path, burn_captions=True)).burn_subtitles is True


def test_every_offered_platform_is_a_real_preset():
    assert set(PLATFORM_LABELS) == set(PRESETS)
    for name in PLATFORM_LABELS:
        assert PLATFORM_LABELS[name], "every platform needs a human-readable label"


def test_source_is_trimmed():
    assert source_from(FormValues(url="  https://x.test/v  ")) == "https://x.test/v"


def test_default_output_dir_is_absolute_and_named():
    folder = default_output_dir()
    assert folder.is_absolute()
    assert folder.name == "ViralClips"


def test_launch_without_tkinter_explains_itself(monkeypatch, capsys):
    """A Python without tkinter must say so, not traceback."""
    import builtins

    real_import = builtins.__import__

    def no_tkinter(name, *args, **kwargs):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("No module named 'tkinter'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tkinter)
    from clipper.gui import launch

    assert launch() == 1
    message = capsys.readouterr().err
    assert "tkinter" in message
    assert "viral_clipper.py" in message, "it should point at the CLI fallback"


def _fake_tk(monkeypatch):
    """A stand-in for tkinter good enough to execute the window's setup.

    It cannot check that the layout looks right, but it does run every line of
    widget code, so typos, name errors and ordering mistakes surface here
    rather than on someone's desktop.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    tk = MagicMock()
    tk.TclError = type("TclError", (Exception,), {})  # must be a real class

    ttk = MagicMock()
    filedialog = MagicMock()
    messagebox = MagicMock()

    module = types.ModuleType("tkinter")
    module.__dict__.update(
        {k: getattr(tk, k) for k in ("Tk", "StringVar", "IntVar", "BooleanVar", "Text")}
    )
    module.TclError = tk.TclError
    module.ttk = ttk
    module.filedialog = filedialog
    module.messagebox = messagebox

    monkeypatch.setitem(sys.modules, "tkinter", module)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", ttk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", messagebox)
    return module


def test_launch_builds_the_whole_window_without_erroring(monkeypatch):
    fake = _fake_tk(monkeypatch)
    from clipper.gui import launch

    assert launch(on_run=lambda *a, **k: None) == 0
    # The window was created and the event loop entered.
    assert fake.Tk.called
    root = fake.Tk.return_value
    assert root.title.called
    assert root.mainloop.called


def test_launch_survives_a_tk_theme_it_does_not_have(monkeypatch):
    """`clam` is missing on some Tk builds; that must not stop the window."""
    fake = _fake_tk(monkeypatch)
    import tkinter.ttk as ttk

    ttk.Style.return_value.theme_use.side_effect = fake.TclError("no such theme")
    from clipper.gui import launch

    assert launch(on_run=lambda *a, **k: None) == 0
