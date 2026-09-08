"""End-to-end tests: a real video in, real MP4 shorts out."""

import json

import pytest

from clipper.config import ClipperConfig
from clipper.errors import NoMomentsFound
from clipper.ffmpeg import FFmpeg
from clipper.ingest import is_url, resolve_source
from clipper.pipeline import Progress, run_pipeline
from tests.conftest import needs_ffmpeg


def test_is_url():
    assert is_url("https://youtube.com/watch?v=x")
    assert is_url("HTTP://example.com/a.mp4")
    assert not is_url("/tmp/a.mp4")
    assert not is_url("relative/path.mp4")


def test_progress_is_monotonic_and_bounded():
    seen = []
    tracker = Progress(lambda message, fraction: seen.append(fraction))
    for stage in ("source", "transcribe", "analyze", "select", "render", "finish"):
        tracker.stage(stage)
        tracker.update(stage, 0.5)
        tracker.update(stage, 1.0)
    assert seen == sorted(seen)
    assert 0.0 <= seen[0] and seen[-1] <= 1.0
    assert seen[-1] == pytest.approx(1.0, abs=0.02)


@needs_ffmpeg
def test_resolve_source_picks_up_a_caption_sidecar(sample_source, tmp_path):
    media = resolve_source(str(sample_source), tmp_path)
    assert media.info.duration > 60
    assert [p.name for p in media.caption_files] == ["talk.en.srt"]


@needs_ffmpeg
def test_dry_run_scores_without_rendering(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=3, dry_run=True,
    )
    result = run_pipeline(str(sample_source), config)
    assert len(result.clips) == 3
    assert all(clip.video_path is None for clip in result.clips)
    assert result.stats["transcript_source"].startswith("captions:")


@needs_ffmpeg
def test_scoring_finds_the_scripted_moments(sample_source, tmp_path):
    """The fixture hides three strong blocks among filler; all three must win."""
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=3, dry_run=True,
    )
    result = run_pipeline(str(sample_source), config)
    hooks = " ".join(clip.hook.lower() for clip in result.clips)
    assert "nobody tells you the biggest mistake" in hooks
    assert "most people never get good" in hooks
    assert "morning routine" in hooks
    # None of the filler blocks should have made it in.
    assert "welcome back everyone" not in hooks
    assert all(clip.score > 40 for clip in result.clips)


@needs_ffmpeg
@pytest.mark.parametrize("layout", ["auto", "blur"])
def test_renders_playable_vertical_clips(sample_source, tmp_path, layout):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=1, layout=layout, ffmpeg_preset="ultrafast",
    )
    result = run_pipeline(str(sample_source), config)
    clip = result.clips[0]

    info = FFmpeg().probe(clip.video_path)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio and info.has_video
    assert info.duration == pytest.approx(clip.duration, abs=0.6)

    assert clip.thumbnail_path and clip.subtitle_path
    subtitles = open(clip.subtitle_path, encoding="utf-8").read()
    assert "[Events]" in subtitles and "Dialogue:" in subtitles
    # Captions are burned relative to the clip, so they start at zero.
    assert "0:00:00" in subtitles


@needs_ffmpeg
def test_manifest_is_written_and_reloadable(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=2, dry_run=True,
    )
    result = run_pipeline(str(sample_source), config)
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(data["clips"]) == 2
    assert data["source"]["info"]["width"] == 1280
    assert data["config"]["platform"] == "tiktok"
    for clip in data["clips"]:
        assert clip["copy"]["title"] and clip["copy"]["hashtags"]
        assert clip["breakdown"]["signals"]


@needs_ffmpeg
def test_clips_do_not_overlap(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=3, dry_run=True,
    )
    clips = run_pipeline(str(sample_source), config).clips
    for earlier, later in zip(clips, clips[1:]):
        assert later.start >= earlier.start
        shared = min(earlier.end, later.end) - later.start
        assert shared < 0.3 * min(earlier.duration, later.duration)


@needs_ffmpeg
def test_transcript_is_cached_between_runs(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        max_clips=1, dry_run=True,
    )
    run_pipeline(str(sample_source), config)
    cached = list((tmp_path / "ws" / "transcripts").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text())["words"]


@needs_ffmpeg
def test_impossible_duration_window_reports_clearly(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        min_duration=400, max_duration=500, dry_run=True,
    )
    with pytest.raises(NoMomentsFound, match="min-duration"):
        run_pipeline(str(sample_source), config)


@needs_ffmpeg
def test_min_score_above_everything_reports_clearly(sample_source, tmp_path):
    config = ClipperConfig(
        workspace=tmp_path / "ws", output_dir=tmp_path / "out",
        min_score=99.9, dry_run=True,
    )
    with pytest.raises(NoMomentsFound, match="min-score"):
        run_pipeline(str(sample_source), config)
