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
    # Referenced by bare name: ffmpeg runs from the output's folder.
    assert "sendcmd=f=cmds.txt" in graph
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


# ---------------------------------------------------------------------------
# Windows paths in a filtergraph
# ---------------------------------------------------------------------------


def windows_request(**kwargs):
    """A request whose paths look like a real Windows install."""
    output = Path("C:/Users/Pouria/Desktop/clipper/job/clip-01.mp4")
    defaults = dict(
        source=Path("C:/Users/Pouria/Desktop/clipper/downloads/video.mp4"),
        output=output,
        subtitle_path=output.with_suffix(".ass"),
        sendcmd_path=output.with_suffix(".cmds.txt"),
    )
    defaults.update(kwargs)
    return make_request(**defaults)


def test_filtergraph_carries_no_drive_letter_or_backslash():
    """`C:\\Users\\...` in a filtergraph breaks ffmpeg's option parser.

    ffmpeg splits filter options on ':', so a drive colon ends the option and
    the rest of the path is read as a new option name. Referring to files that
    sit beside the output by bare name avoids the whole problem.
    """
    for layout in ("auto", "blur"):
        graph = build_filter_complex(windows_request(layout=layout))
        assert "C:" not in graph
        assert "\\" not in graph
        assert "subtitles=filename=clip-01.ass" in graph


def test_sendcmd_is_referenced_by_bare_name():
    graph = build_filter_complex(windows_request())
    assert "sendcmd=f=clip-01.cmds.txt" in graph


def test_ffmpeg_runs_from_the_folder_holding_those_files():
    request = windows_request()
    assert request.working_dir == Path("C:/Users/Pouria/Desktop/clipper/job")
    assert request.filter_arg(request.subtitle_path) == "clip-01.ass"


def test_a_file_outside_the_output_folder_is_still_escaped():
    """The bare-name shortcut only applies next to the output."""
    request = windows_request(subtitle_path=Path("C:/elsewhere/subs.ass"))
    argument = request.filter_arg(request.subtitle_path)
    # Double-escaped so one level survives the graph parser and reaches the
    # option parser, and forward slashes rather than backslashes.
    assert argument == "C\\\\:/elsewhere/subs.ass"
    assert "\\\\:" in argument


def test_filter_arg_of_nothing_is_empty():
    assert windows_request().filter_arg(None) == ""


def test_input_and_output_stay_absolute():
    """Only filter arguments go relative; ffmpeg's own arguments do not."""
    args = build_command(None, windows_request())
    assert any("video.mp4" in str(a) and ("/" in str(a) or "\\" in str(a)) for a in args)
    assert any(str(a).endswith("clip-01.mp4") for a in args)


def test_posix_paths_are_unaffected():
    graph = build_filter_complex(make_request(subtitle_path=Path("/x/subs.ass")))
    assert "subtitles=filename=" in graph
