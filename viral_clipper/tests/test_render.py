from pathlib import Path

import pytest

from clipper.config import PRESETS
from clipper.reframe import CropPlan
from clipper.render import (
    RenderRequest,
    build_command,
    build_filter_complex,
    build_video_chain,
    escape_ffmpeg_path,
)


def make_request(**kwargs):
    defaults = dict(
        source=Path("/in.mp4"),
        output=Path("/out.mp4"),
        start=5.0,
        end=35.0,
        preset=PRESETS["tiktok"],
        plan=CropPlan(
            crop_w=608, crop_h=1080, y=0,
            keyframes=[(0.0, 100), (1.0, 140)],
            source_w=1920, source_h=1080,
        ),
        layout="auto",
    )
    defaults.update(kwargs)
    return RenderRequest(**defaults)


def test_escape_ffmpeg_path_escapes_filter_metacharacters():
    escaped = escape_ffmpeg_path(Path("/a b/c:d'e[f].ass"))
    for char in (":", "'", "[", "]"):
        assert "\\" + char in escaped


def test_auto_layout_uses_sendcmd_for_a_moving_crop():
    chain = build_video_chain(make_request(sendcmd_path=Path("/cmds.txt")))
    graph = ",".join(chain)
    assert "sendcmd=f=/cmds.txt" in graph
    assert graph.index("sendcmd") < graph.index("crop="), "sendcmd must precede crop"
    assert "scale=1080:1920" in graph
    assert graph.startswith("setpts=PTS-STARTPTS")


def test_static_crop_skips_sendcmd():
    plan = CropPlan(crop_w=608, crop_h=1080, y=0, keyframes=[(0.0, 100)],
                    source_w=1920, source_h=1080)
    graph = ",".join(build_video_chain(make_request(plan=plan, sendcmd_path=Path("/c.txt"))))
    assert "sendcmd" not in graph and "crop=" in graph


def test_center_layout_never_pans():
    graph = ",".join(build_video_chain(make_request(layout="center", sendcmd_path=Path("/c.txt"))))
    assert "sendcmd" not in graph


def test_vertical_source_is_not_cropped():
    plan = CropPlan(crop_w=1080, crop_h=1920, y=0, keyframes=[(0.0, 0)],
                    source_w=1080, source_h=1920)
    graph = ",".join(build_video_chain(make_request(plan=plan)))
    assert "crop=" not in graph
    assert "scale=1080:1920" in graph


def test_subtitles_are_burned_after_scaling():
    graph = ",".join(build_video_chain(make_request(subtitle_path=Path("/s.ass"))))
    assert graph.index("scale=") < graph.index("subtitles=")
    assert graph.endswith("format=yuv420p")


def test_blur_layout_builds_a_split_graph():
    graph = build_filter_complex(make_request(layout="blur", subtitle_path=Path("/s.ass")))
    assert "split=2[bgsrc][fgsrc]" in graph
    assert "gblur" in graph and "overlay=(W-w)/2:(H-h)/2" in graph
    assert graph.count("subtitles=") == 1
    assert "[v]" in graph and "[a]" in graph


def test_fit_layout_pads():
    graph = build_filter_complex(make_request(layout="fit"))
    assert "pad=1080:1920" in graph and "gblur" not in graph


def test_audio_chain_normalises_by_default():
    assert "loudnorm=I=-14" in build_filter_complex(make_request())
    assert "loudnorm" not in build_filter_complex(make_request(normalize_audio=False))


def test_silent_source_produces_no_audio_branch():
    graph = build_filter_complex(make_request(), has_audio=False)
    assert "[0:a]" not in graph


def test_command_seeks_and_encodes_correctly():
    args = build_command(None, make_request())
    assert args[:2] == ["-ss", "5.000"]
    assert "-t" in args and args[args.index("-t") + 1] == "30.000"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[-2:] == ["+faststart", "/out.mp4"]
    assert "[a]" in args


def test_command_drops_audio_when_the_source_is_silent():
    args = build_command(None, make_request(), has_audio=False)
    assert "-an" in args and "-c:a" not in args


@pytest.mark.parametrize("platform", sorted(PRESETS))
def test_every_preset_builds_a_graph(platform):
    graph = build_filter_complex(make_request(preset=PRESETS[platform]))
    preset = PRESETS[platform]
    assert f"scale={preset.width}:{preset.height}" in graph
