"""Web API tests. These need fastapi + httpx and a real ffmpeg."""

import time

import pytest

from tests.conftest import needs_fastapi, needs_ffmpeg

pytestmark = [needs_fastapi]


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from clipper.server.app import create_app

    with TestClient(create_app(workspace=tmp_path / "ws", workers=1)) as test_client:
        yield test_client


def wait_for(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_index_serves_the_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Viral" in response.text and "<script" in response.text


def test_options_describe_every_choice(client):
    data = client.get("/api/options").json()
    assert {p["name"] for p in data["platforms"]} >= {"tiktok", "reels", "shorts"}
    assert "auto" in data["layouts"] and "punch" in data["caption_styles"]
    assert all(p["width"] > 0 for p in data["platforms"])


def test_health_reports_ffmpeg(client):
    data = client.get("/api/health").json()
    assert set(data) >= {"ok", "ffmpeg", "yt_dlp", "whisper"}


def test_rejects_a_missing_source(client):
    response = client.post("/api/jobs", json={"source": "/no/such/file.mp4"})
    assert response.status_code == 400
    assert "no such file" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"source": "x", "platform": "myspace"}, 400),
        ({"source": "x", "layout": "sideways"}, 400),
        ({"source": "x", "caption_style": "graffiti"}, 400),
        ({"source": "x", "max_clips": 99}, 422),
        ({"source": ""}, 422),
        ({}, 422),
    ],
)
def test_rejects_bad_options(client, tmp_path, payload, status):
    if payload.get("source") == "x":
        source = tmp_path / "real.mp4"
        source.write_bytes(b"not really a video")
        payload["source"] = str(source)
    assert client.post("/api/jobs", json=payload).status_code == status


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.delete("/api/jobs/deadbeef").status_code == 404


def test_upload_rejects_unsupported_types(client):
    response = client.post(
        "/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]


def test_upload_stores_the_file(client):
    response = client.post(
        "/api/upload", files={"file": ("My Talk.mp4", b"\x00" * 2048, "video/mp4")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["bytes"] == 2048
    assert body["name"] == "my-talk.mp4", "the filename is slugified"
    assert "uploads" in body["path"]


@needs_ffmpeg
def test_full_job_lifecycle(client, sample_source):
    created = client.post(
        "/api/jobs",
        json={
            "source": str(sample_source),
            "platform": "reels",
            "max_clips": 1,
            "layout": "center",
        },
    )
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    assert client.get("/api/jobs").json()["jobs"][0]["job_id"] == job_id

    job = wait_for(client, job_id)
    assert job["status"] == "done", job.get("error")
    assert len(job["clips"]) == 1
    clip = job["clips"][0]
    assert clip["copy"]["title"] and clip["copy"]["hashtags"]
    assert clip["width"] == 1080 and clip["height"] == 1920

    video = client.get(f"/api/jobs/{job_id}/clips/1/video")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 10_000
    assert client.get(f"/api/jobs/{job_id}/clips/1/thumbnail").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/clips/1/subtitles").status_code == 200

    # Unknown clips and asset kinds must not leak anything.
    assert client.get(f"/api/jobs/{job_id}/clips/99/video").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/clips/1/passwd").status_code == 404

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


@needs_ffmpeg
def test_failed_job_reports_the_error(client, tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video file at all")
    job_id = client.post("/api/jobs", json={"source": str(broken)}).json()["job_id"]
    job = wait_for(client, job_id, timeout=60)
    assert job["status"] == "error"
    assert job["error"]
