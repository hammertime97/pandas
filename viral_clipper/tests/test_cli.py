"""CLI argument handling and command dispatch."""

import pytest

from clipper.cli import build_parser, config_from_args, main
from tests.conftest import needs_ffmpeg


def parse(argv):
    return build_parser().parse_args(argv)


def test_bare_source_defaults_to_the_clip_command(capsys, tmp_path, monkeypatch):
    calls = {}

    def fake_clip(args):
        calls["source"] = args.source
        return 0

    monkeypatch.setitem(__import__("clipper.cli", fromlist=["COMMANDS"]).COMMANDS, "clip", fake_clip)
    assert main(["video.mp4"]) == 0
    assert calls["source"] == "video.mp4"


def test_config_from_args_maps_the_inverted_flags():
    args = parse(["clip", "in.mp4", "--no-subtitles", "--no-uppercase", "--no-normalize"])
    config = config_from_args(args)
    assert config.burn_subtitles is False
    assert config.uppercase_captions is False
    assert config.normalize_audio is False


def test_config_from_args_defaults():
    config = config_from_args(parse(["clip", "in.mp4"]))
    assert config.platform == "tiktok" and config.max_clips == 5
    assert config.layout == "auto" and config.burn_subtitles is True
    assert config.dry_run is False


def test_config_from_args_passes_durations_and_llm():
    config = config_from_args(
        parse(["clip", "in.mp4", "--min-duration", "20", "--max-duration", "45",
               "--llm", "--llm-weight", "0.25"])
    )
    assert (config.min_duration, config.max_duration) == (20.0, 45.0)
    assert config.use_llm is True and config.llm_weight == 0.25


def test_fps_flag():
    assert config_from_args(parse(["clip", "in.mp4", "--fps", "60"])).fps == 60
    assert config_from_args(parse(["clip", "in.mp4"])).fps is None
    with pytest.raises(SystemExit):
        parse(["clip", "in.mp4", "--fps", "500"])


def test_invalid_choices_are_rejected_by_the_parser():
    for argv in (["clip", "in.mp4", "--platform", "myspace"],
                 ["clip", "in.mp4", "--layout", "sideways"],
                 ["clip", "in.mp4", "--caption-style", "graffiti"]):
        with pytest.raises(SystemExit):
            parse(argv)


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out


def test_doctor_runs(capsys):
    code = main(["doctor"])
    output = capsys.readouterr().out
    assert "clipper" in output and "ffmpeg" in output
    assert code in (0, 1)


def test_clipper_errors_become_exit_code_2(capsys):
    assert main(["clip", "/definitely/not/here.mp4", "--workspace", "/tmp/x"]) == 2
    assert "error:" in capsys.readouterr().err


@needs_ffmpeg
def test_dry_run_prints_a_summary(capsys, sample_source, tmp_path):
    code = main([
        "clip", str(sample_source), "--dry-run", "--max-clips", "2",
        "--workspace", str(tmp_path / "ws"), "-o", str(tmp_path / "out"),
    ])
    output = capsys.readouterr().out
    assert code == 0
    assert "dry run" in output and "score" in output
    assert "[1]" in output and "[2]" in output


@needs_ffmpeg
def test_json_output_is_parseable(capsys, sample_source, tmp_path):
    import json

    code = main([
        "clip", str(sample_source), "--dry-run", "--max-clips", "1", "--json",
        "--workspace", str(tmp_path / "ws"), "-o", str(tmp_path / "out"),
    ])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["clips"]) == 1 and data["stats"]["selected"] == 1


def test_closed_pipe_exits_cleanly():
    """`clipper doctor | head -1` must not dump a traceback.

    Run out of process: the handler replaces the stdout file descriptor, which
    would otherwise pull pytest's capture out from under it.
    """
    import subprocess
    import sys

    producer = subprocess.Popen(
        [sys.executable, "-m", "clipper", "doctor"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    consumer = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=producer.stdout,
    )
    producer.stdout.close()
    consumer.wait(timeout=60)
    stderr = producer.communicate(timeout=60)[1].decode()

    assert "Traceback" not in stderr, stderr
    assert "BrokenPipeError" not in stderr, stderr
