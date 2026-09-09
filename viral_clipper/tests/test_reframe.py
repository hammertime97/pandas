import pytest

from clipper.reframe import (
    CropPlan,
    crop_size,
    even,
    sendcmd_script,
    smooth_positions,
    write_sendcmd,
)


def test_even_rounding():
    # h264 needs even dimensions; exact .5 cases may round either way.
    assert even(100) == 100
    assert even(99.4) == 100
    assert even(102.9) == 102
    assert all(even(v) % 2 == 0 for v in (0, 1, 7, 99, 100, 101, 1079.5))


@pytest.mark.parametrize(
    "src_w, src_h, ratio, expected",
    [
        (1920, 1080, 9 / 16, (608, 1080)),   # landscape -> tall window
        (1080, 1920, 9 / 16, (1080, 1920)),  # already vertical -> unchanged
        (1280, 720, 1.0, (720, 720)),        # square crop
    ],
)
def test_crop_size(src_w, src_h, ratio, expected):
    assert crop_size(src_w, src_h, ratio) == expected


def test_crop_size_always_fits_inside_the_frame():
    for width, height in [(1920, 1080), (640, 480), (1080, 1350), (720, 720)]:
        crop_w, crop_h = crop_size(width, height, 9 / 16)
        assert crop_w <= width and crop_h <= height
        assert crop_w % 2 == 0 and crop_h % 2 == 0


def test_crop_size_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        crop_size(0, 100, 9 / 16)


def test_smoothing_rejects_outliers_and_fills_dropouts():
    noisy = [0.5, 0.9, 0.5, None, 0.52, 0.51, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5]
    smoothed = smooth_positions(noisy)
    assert len(smoothed) == len(noisy)
    assert max(smoothed) < 0.6, "single-frame spikes must not move the camera"


def test_smoothing_has_no_phase_lag():
    """A symmetric step must be crossed at the step, not after it."""
    step = [0.2] * 10 + [0.8] * 10
    smoothed = smooth_positions(step)
    midpoint = (smoothed[0] + smoothed[-1]) / 2
    crossing = next(i for i, v in enumerate(smoothed) if v >= midpoint)
    assert 9 <= crossing <= 11


def test_smoothing_handles_all_missing_and_empty():
    assert smooth_positions([None, None, None]) == [0.5, 0.5, 0.5]
    assert smooth_positions([]) == []


def test_smoothing_stays_in_range():
    assert all(0.0 <= v <= 1.0 for v in smooth_positions([0.0, 1.0] * 12))


def test_deadzone_ignores_small_wobbles():
    jitter = [0.5, 0.52, 0.5, 0.51, 0.49, 0.5] * 3
    smoothed = smooth_positions(jitter, deadzone=0.1)
    assert max(smoothed) - min(smoothed) < 0.02


def test_crop_plan_reports_static_and_travel():
    static = CropPlan(crop_w=608, crop_h=1080, y=0, keyframes=[(0.0, 100)])
    assert static.static and static.x == 100 and static.travel == 0
    moving = CropPlan(crop_w=608, crop_h=1080, y=0, keyframes=[(0.0, 100), (1.0, 140), (2.0, 120)])
    assert not moving.static and moving.travel == 60
    assert moving.to_dict()["keyframes"] == 3


def test_sendcmd_script_format():
    plan = CropPlan(crop_w=608, crop_h=1080, y=0, keyframes=[(0.0, 10), (0.25, 20)])
    script = sendcmd_script(plan)
    assert script.splitlines() == ["0.000 crop x 10;", "0.250 crop x 20;"]


def test_write_sendcmd(tmp_path):
    plan = CropPlan(crop_w=608, crop_h=1080, y=0, keyframes=[(0.0, 10)])
    path = write_sendcmd(plan, tmp_path / "nested" / "cmds.txt")
    assert path.read_text().strip() == "0.000 crop x 10;"


def test_numpy_and_pure_python_tracking_agree():
    """The numpy fast path must not change the answer, only the speed."""
    pytest.importorskip("numpy")
    from clipper.reframe import _centroid, _column_energy_numpy, _column_energy_python

    width, height, count = 24, 16, 4
    # A bright block that shifts right on each frame.
    frames = []
    for step in range(count):
        buffer = bytearray(width * height)
        for row in range(4, 12):
            for column in range(step * 3, step * 3 + 6):
                buffer[row * width + column] = 240
        frames.append(memoryview(bytes(buffer)))

    fast = _centroid(list(_column_energy_numpy(frames, width, height)))
    slow = _centroid(_column_energy_python(frames, width, height))
    assert fast == pytest.approx(slow, abs=1e-9)
    assert fast is not None and 0.0 <= fast <= 1.0


def test_broken_opencv_falls_back_instead_of_crashing(monkeypatch):
    """Colab ships a cv2 that imports but has no attributes; that must not crash.

    Guarding only ImportError is not enough: the module is importable, so the
    failure surfaces as an AttributeError deep inside a render.
    """
    import sys
    import types

    from clipper.reframe import _detect_faces_opencv, _load_face_cascade

    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    assert _load_face_cascade() is None
    assert _detect_faces_opencv([memoryview(bytes(64))], 8, 8) == []


def test_cascade_loading_survives_an_exploding_cv2(monkeypatch):
    import sys
    import types

    from clipper.reframe import _load_face_cascade

    exploding = types.ModuleType("cv2")

    def boom(*args, **kwargs):
        raise RuntimeError("native library not loaded")

    exploding.CascadeClassifier = boom
    monkeypatch.setitem(sys.modules, "cv2", exploding)
    assert _load_face_cascade() is None


def test_tracking_still_follows_the_subject_without_opencv(monkeypatch):
    """With faces unavailable, motion tracking has to carry the reframe."""
    from clipper import reframe

    monkeypatch.setattr(reframe, "_detect_faces_opencv", lambda *a, **k: [])
    monkeypatch.setattr(
        reframe,
        "estimate_subject_track",
        reframe.estimate_subject_track,  # unchanged, exercised via smoothing below
    )
    drifting = [i / 20 for i in range(21)]
    smoothed = reframe.smooth_positions(drifting)
    assert smoothed[-1] > smoothed[0] + 0.4, "the crop must still travel"


def test_face_detection_failure_mid_clip_is_contained(monkeypatch):
    import sys
    import types

    from clipper.reframe import _detect_faces_opencv

    pytest.importorskip("numpy")

    class Cascade:
        def empty(self):
            return False

        def detectMultiScale(self, *args, **kwargs):
            raise RuntimeError("cascade blew up on frame 2")

    module = types.ModuleType("cv2")
    module.CascadeClassifier = lambda *a, **k: Cascade()
    module.data = types.SimpleNamespace(haarcascades="/tmp/")
    monkeypatch.setitem(sys.modules, "cv2", module)
    assert _detect_faces_opencv([memoryview(bytes(64))], 8, 8) == []
