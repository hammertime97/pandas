"""Source resolution, and the YouTube bot-check retry path."""

from pathlib import Path

import pytest

from clipper import ingest
from clipper.errors import IngestError


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://www.youtube-nocookie.com/embed/abc", True),
        ("https://vimeo.com/123", False),
        ("/tmp/local.mp4", False),
    ],
)
def test_is_youtube(url, expected):
    assert ingest.is_youtube(url) is expected


@pytest.mark.parametrize(
    "message",
    [
        "ERROR: [youtube] 4zVFht1KbnY: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication.",
        "Please sign in to view this video",
        "failed to extract any player response",
    ],
)
def test_bot_check_is_recognised(message):
    assert ingest.looks_like_bot_check(message)


@pytest.mark.parametrize(
    "message",
    ["HTTP Error 404: Not Found", "Video unavailable", "Unsupported URL", ""],
)
def test_other_failures_are_not_mistaken_for_a_bot_check(message):
    assert not ingest.looks_like_bot_check(message)


def _fail_with(message):
    def attempt(url, options, progress):
        raise IngestError(message)

    return attempt


def test_bot_check_rotates_through_player_clients(monkeypatch):
    """Each retry must actually ask for a different client."""
    seen = []

    def attempt(url, options, progress):
        seen.append(
            options.get("extractor_args", {}).get("youtube", {}).get("player_client")
        )
        if len(seen) < 3:
            raise IngestError("Sign in to confirm you're not a bot")
        return {"title": "ok"}

    monkeypatch.setattr(ingest, "_download_with_module", attempt)
    info = ingest._download_with_fallbacks(
        "https://youtu.be/abc", {"format": "best"}, None
    )
    assert info == {"title": "ok"}
    assert seen[0] is None, "the first attempt uses yt-dlp's own default"
    assert [c for c in seen[1:]] == [["android_vr"], ["tv"]]


def test_exhausted_retries_explain_the_workarounds(monkeypatch):
    monkeypatch.setattr(
        ingest, "_download_with_module", _fail_with("Sign in to confirm you're not a bot")
    )
    with pytest.raises(IngestError) as caught:
        ingest._download_with_fallbacks("https://youtu.be/abc", {}, None)
    message = str(caught.value)
    assert "cookies" in message.lower()
    assert "local file" in message.lower()


def test_a_normal_failure_is_not_retried(monkeypatch):
    """Rotating clients cannot fix a 404, so do not waste five attempts on it."""
    calls = []

    def attempt(url, options, progress):
        calls.append(options)
        raise IngestError("HTTP Error 404: Not Found")

    monkeypatch.setattr(ingest, "_download_with_module", attempt)
    with pytest.raises(IngestError, match="404"):
        ingest._download_with_fallbacks("https://youtu.be/abc", {}, None)
    assert len(calls) == 1


def test_cookies_are_authoritative(monkeypatch):
    """With cookies supplied there is nothing to rotate through — fail fast."""
    calls = []

    def attempt(url, options, progress):
        calls.append(options)
        raise IngestError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(ingest, "_download_with_module", attempt)
    with pytest.raises(IngestError):
        ingest._download_with_fallbacks(
            "https://youtu.be/abc", {"cookiefile": "/tmp/c.txt"}, None
        )
    assert len(calls) == 1


def test_non_youtube_urls_are_not_rotated(monkeypatch):
    calls = []

    def attempt(url, options, progress):
        calls.append(options)
        raise IngestError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(ingest, "_download_with_module", attempt)
    with pytest.raises(IngestError):
        ingest._download_with_fallbacks("https://vimeo.com/1", {}, None)
    assert len(calls) == 1


def test_missing_local_file_reports_clearly(tmp_path):
    with pytest.raises(IngestError, match="no such file"):
        ingest.resolve_source(str(tmp_path / "nope.mp4"), tmp_path)


def test_ytdlp_accepts_a_pip_installed_ffmpeg_filename():
    """imageio-ffmpeg names its binary oddly; yt-dlp still has to recognise it.

    yt-dlp maps a path to a program by looking for "ffmpeg"/"ffprobe" *inside*
    the filename, so `ffmpeg-win-x86_64-v7.0.2.exe` resolves correctly. If that
    ever stops being true, passing the path would silently do nothing.
    """
    import os

    for filename in (
        "ffmpeg-win-x86_64-v7.0.2.exe",
        "ffmpeg-linux-x86_64-v7.0.2",
        "ffmpeg.exe",
        "ffmpeg",
    ):
        basename = next((p for p in ("ffmpeg", "ffprobe") if p in filename), "ffmpeg")
        assert basename == "ffmpeg"
        assert basename in os.path.basename(filename)


def test_download_hands_ytdlp_the_ffmpeg_we_resolved(monkeypatch, tmp_path):
    """Without this, a pip-installed ffmpeg is invisible to yt-dlp."""
    captured = {}

    def fake_fallbacks(url, options, progress):
        captured.update(options)
        raise IngestError("stop here, the options are what we are checking")

    monkeypatch.setattr(ingest, "find_ffmpeg", lambda: "/opt/ff/ffmpeg-v7.exe")
    monkeypatch.setattr(ingest, "_download_with_fallbacks", fake_fallbacks)
    with pytest.raises(IngestError):
        ingest._download(
            "https://youtu.be/abc", tmp_path,
            progress=None, format_selector="best", cookies_file=None,
        )
    assert captured["ffmpeg_location"] == "/opt/ff/ffmpeg-v7.exe"


def test_download_omits_the_option_when_there_is_no_ffmpeg(monkeypatch, tmp_path):
    """Passing None would make yt-dlp warn about a location that does not exist."""
    captured = {}

    def fake_fallbacks(url, options, progress):
        captured.update({"keys": set(options)})
        raise IngestError("stop")

    monkeypatch.setattr(ingest, "find_ffmpeg", lambda: None)
    monkeypatch.setattr(ingest, "_download_with_fallbacks", fake_fallbacks)
    with pytest.raises(IngestError):
        ingest._download(
            "https://youtu.be/abc", tmp_path,
            progress=None, format_selector="best", cookies_file=None,
        )
    assert "ffmpeg_location" not in captured["keys"]


def test_cli_download_forwards_the_ffmpeg_location(monkeypatch):
    """The subprocess fallback needs the same treatment as the module path."""
    captured = {}

    class Result:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(args, **kwargs):
        captured["args"] = args
        return Result()

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/yt-dlp")
    with pytest.raises(IngestError):
        ingest._download_with_cli(
            "https://youtu.be/abc",
            {"format": "best", "outtmpl": "x", "ffmpeg_location": "/opt/ff/ffmpeg"},
            None,
        )
    args = captured["args"]
    assert "--ffmpeg-location" in args
    assert args[args.index("--ffmpeg-location") + 1] == "/opt/ff/ffmpeg"


# ---------------------------------------------------------------------------
# Unmerged download fragments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("1693eb8db2.f140.m4a", True),    # YouTube audio-only format
        ("1693eb8db2.f616.mp4", True),    # video-only format
        ("video.f251.webm", True),
        ("1693eb8db2.mp4", False),        # the merged result
        ("my.video.mp4", False),
        ("talk.f.mp4", False),            # no format number
    ],
)
def test_format_fragments_are_recognised(name, expected):
    assert ingest.is_format_fragment(Path(name)) is expected


def _touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


def test_a_merged_download_wins_over_its_fragments(tmp_path):
    """`.f140.m4a` sorts before `.mp4`, and is audio only — it must not win."""
    _touch(
        tmp_path,
        "1693eb8db2.f140.m4a",
        "1693eb8db2.f616.mp4",
        "1693eb8db2.mp4",
        "1693eb8db2.title.txt",
        "1693eb8db2.en.vtt",
    )
    found = ingest._existing_download(tmp_path, "1693eb8db2")
    assert found is not None and found.name == "1693eb8db2.mp4"


def test_leftover_fragments_alone_do_not_count_as_a_download(tmp_path):
    """A failed run leaves fragments; reusing them would poison every retry."""
    _touch(tmp_path, "1693eb8db2.f140.m4a", "1693eb8db2.f616.mp4")
    assert ingest._existing_download(tmp_path, "1693eb8db2") is None


def test_partial_and_bookkeeping_files_are_ignored(tmp_path):
    _touch(tmp_path, "abc.part", "abc.ytdl", "abc.title.txt", "abc.en.srt")
    assert ingest._existing_download(tmp_path, "abc") is None


def test_a_video_container_is_preferred_over_audio(tmp_path):
    _touch(tmp_path, "abc.m4a", "abc.mkv")
    found = ingest._existing_download(tmp_path, "abc")
    assert found is not None and found.suffix == ".mkv"


def test_an_audio_only_source_is_rejected_with_a_useful_message(tmp_path, monkeypatch):
    """The pipeline needs a video stream; say so, and say how to recover."""
    from clipper.models import MediaInfo

    video = tmp_path / "audio-only.m4a"
    video.write_bytes(b"")

    class FakeFFmpeg:
        def probe(self, path):
            return MediaInfo(
                path=str(path), duration=120.0, width=0, height=0,
                has_audio=True, has_video=False,
            )

    with pytest.raises(IngestError) as caught:
        ingest.resolve_source(str(video), tmp_path, FakeFFmpeg())
    message = str(caught.value)
    assert "no video track" in message
    assert "downloads folder" in message, "the error should say how to recover"
