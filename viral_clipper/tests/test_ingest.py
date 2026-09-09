"""Source resolution, and the YouTube bot-check retry path."""

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
