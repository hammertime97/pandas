"""Plan the 9:16 crop window for a landscape source.

A static centre crop throws away half of most talking-head footage.  This
module samples the clip, estimates where the subject is on each sample, and
produces a *smoothed* crop path that ffmpeg can follow via ``sendcmd``.

Everything degrades gracefully: OpenCV is used for face detection when it is
installed, numpy accelerates the motion fallback when it is installed, and a
pure-Python path keeps working when neither is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from clipper.ffmpeg import FFmpeg
from clipper.utils import clamp, log, moving_average

#: Grayscale analysis resolution. Small is fine: we only need a centroid.
ANALYSIS_WIDTH = 224


def even(value: float) -> int:
    """Round to the nearest even integer (h264 requires even dimensions)."""
    return int(round(value / 2.0)) * 2


@dataclass
class CropPlan:
    """A crop window, optionally moving over time."""

    crop_w: int
    crop_h: int
    y: int
    keyframes: List[Tuple[float, int]] = field(default_factory=list)
    layout: str = "auto"
    source_w: int = 0
    source_h: int = 0

    @property
    def static(self) -> bool:
        return len(self.keyframes) <= 1

    @property
    def x(self) -> int:
        return self.keyframes[0][1] if self.keyframes else 0

    @property
    def travel(self) -> int:
        """Total horizontal distance the crop window moves, in source pixels."""
        xs = [x for _, x in self.keyframes]
        return sum(abs(xs[i] - xs[i - 1]) for i in range(1, len(xs)))

    def to_dict(self) -> dict:
        return {
            "crop_w": self.crop_w,
            "crop_h": self.crop_h,
            "x": self.x,
            "y": self.y,
            "layout": self.layout,
            "static": self.static,
            "keyframes": len(self.keyframes),
            "travel_px": self.travel,
        }


def crop_size(source_w: int, source_h: int, target_ratio: float) -> Tuple[int, int]:
    """Largest window of ``target_ratio`` that fits inside the source frame."""
    if source_w <= 0 or source_h <= 0:
        raise ValueError("source dimensions must be positive")
    if source_w / source_h > target_ratio:
        # Source is wider than the target: full height, narrower width.
        height = source_h
        width = min(source_w, even(source_h * target_ratio))
    else:
        width = source_w
        height = min(source_h, even(source_w / target_ratio))
    return max(2, even(width)), max(2, even(height))


# ---------------------------------------------------------------------------
# Subject position estimation
# ---------------------------------------------------------------------------


def _frames_from_bytes(
    raw: bytes, width: int, height: int
) -> List[memoryview]:
    """Split a rawvideo gray buffer into per-frame views."""
    frame_size = width * height
    if frame_size <= 0:
        return []
    count = len(raw) // frame_size
    view = memoryview(raw)
    return [view[i * frame_size : (i + 1) * frame_size] for i in range(count)]


def _column_energy_numpy(frames: Sequence[memoryview], width: int, height: int):
    import numpy as np  # type: ignore

    stack = np.frombuffer(b"".join(bytes(f) for f in frames), dtype=np.uint8)
    stack = stack.reshape(len(frames), height, width).astype(np.int16)
    if len(frames) < 2:
        # No motion available: fall back to spatial detail (edges).
        detail = np.abs(np.diff(stack[0].astype(np.int16), axis=1))
        return np.pad(detail.sum(axis=0), (0, 1)).astype(float)
    diff = np.abs(np.diff(stack, axis=0)).sum(axis=(0, 1))
    return diff.astype(float)


def _column_energy_python(
    frames: Sequence[memoryview], width: int, height: int
) -> List[float]:
    """Pure-Python fallback for per-column motion energy."""
    energy = [0.0] * width
    if len(frames) < 2:
        return energy
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        for row in range(height):
            offset = row * width
            prev_row = previous[offset : offset + width]
            cur_row = current[offset : offset + width]
            for column in range(width):
                energy[column] += abs(cur_row[column] - prev_row[column])
    return energy


def _centroid(weights: Sequence[float]) -> Optional[float]:
    """Weighted centre of mass as a fraction of the width, or ``None``."""
    total = float(sum(weights))
    if total <= 0:
        return None
    moment = sum(index * value for index, value in enumerate(weights))
    return (moment / total) / max(1, len(weights) - 1)


def _load_face_cascade():
    """Return a usable Haar cascade, or ``None``.

    Being importable is not the same as being usable. Some environments —
    Colab among them — ship a ``cv2`` that imports fine but whose native
    extension never loaded, so the module exists with almost none of its
    attributes. Everything here is therefore feature-checked rather than
    assumed, and any failure just means we fall back to motion tracking.
    """
    try:
        import cv2  # type: ignore
    except Exception as exc:  # not only ImportError: broken builds raise others
        log.debug("opencv unavailable: %s", exc)
        return None

    if not hasattr(cv2, "CascadeClassifier"):
        log.info(
            "opencv is installed but not working (no CascadeClassifier) — "
            "using motion tracking instead"
        )
        return None

    try:
        haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        cascade = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            log.debug("opencv face cascade not found under %r", haar_dir)
            return None
        return cascade
    except Exception as exc:
        log.debug("could not load the opencv face cascade: %s", exc)
        return None


def _detect_faces_opencv(
    frames: Sequence[memoryview], width: int, height: int
) -> List[Optional[float]]:
    """Per-frame face centre as a fraction of the width (``None`` when absent).

    Returns an empty list if face detection is unavailable for any reason, which
    the caller reads as "fall back to motion tracking".
    """
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return []

    cascade = _load_face_cascade()
    if cascade is None:
        return []

    positions: List[Optional[float]] = []
    for frame in frames:
        image = np.frombuffer(bytes(frame), dtype=np.uint8).reshape(height, width)
        try:
            faces = cascade.detectMultiScale(
                image, scaleFactor=1.15, minNeighbors=5, minSize=(max(12, width // 20),) * 2
            )
        except Exception as exc:
            log.debug("face detection failed mid-clip: %s", exc)
            return []
        if len(faces) == 0:
            positions.append(None)
            continue
        # Weight by area so the speaker (usually largest / closest) wins.
        total_area = sum(int(w) * int(h) for _, _, w, h in faces)
        centre = sum((int(x) + int(w) / 2.0) * int(w) * int(h) for x, _, w, h in faces)
        positions.append((centre / total_area) / max(1, width - 1))
    return positions


def estimate_subject_track(
    ffmpeg: FFmpeg,
    source: Path,
    start: float,
    duration: float,
    *,
    sample_fps: float = 4.0,
    analysis_width: int = ANALYSIS_WIDTH,
    source_ratio: float = 16 / 9,
) -> List[Optional[float]]:
    """Subject position per sample, as a fraction of frame width in ``[0, 1]``.

    Faces win when OpenCV finds them; otherwise motion energy is used, which
    tracks a moving speaker, a gameplay focus or an active whiteboard well
    enough for reframing.
    """
    analysis_height = max(2, even(analysis_width / max(0.1, source_ratio)))
    try:
        raw = ffmpeg.read_gray_frames(
            source, start, duration, sample_fps, analysis_width, analysis_height
        )
    except Exception as exc:
        log.debug("frame sampling failed, using static crop: %s", exc)
        return []

    frames = _frames_from_bytes(raw, analysis_width, analysis_height)
    if not frames:
        return []

    try:
        faces = _detect_faces_opencv(frames, analysis_width, analysis_height)
    except Exception as exc:
        # Reframing is an enhancement; never let it fail a whole render.
        log.warning("face tracking failed (%s) — falling back to motion tracking", exc)
        faces = []
    if faces and sum(1 for f in faces if f is not None) >= max(2, len(faces) // 4):
        log.debug("tracking %d/%d frames by face", sum(f is not None for f in faces), len(faces))
        return faces

    # Motion fallback: energy is computed over a short rolling group of frames
    # so the centroid reflects recent movement rather than the whole clip.
    group = max(2, int(round(sample_fps)))
    positions: List[Optional[float]] = []
    for index in range(len(frames)):
        window = frames[max(0, index - group + 1) : index + 1]
        try:
            energy = _column_energy_numpy(window, analysis_width, analysis_height)
        except ImportError:
            energy = _column_energy_python(window, analysis_width, analysis_height)
        positions.append(_centroid(list(energy)))
    return positions


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_positions(
    positions: Sequence[Optional[float]],
    *,
    alpha: float = 0.22,
    deadzone: float = 0.06,
    default: float = 0.5,
) -> List[float]:
    """Turn noisy per-frame detections into a calm camera move.

    Four stages: hold the last known value through dropouts, drop single
    sample outliers, apply a deadzone so small wobbles do not move the camera,
    then smooth.

    The smoothing runs forward *and* backward over the track.  A plain
    exponential filter always trails a moving subject, which leaves them
    drifting toward the edge of the frame; because reframing happens offline
    we already know the whole path, so filtering in both directions removes
    that lag entirely.
    """
    if not positions:
        return []

    # 1. Hold through dropouts, forwards then backwards.
    held: List[float] = []
    last = None
    for value in positions:
        if value is not None:
            last = value
        held.append(last if last is not None else default)
    if last is None:
        return [default] * len(positions)
    first_known = next((v for v in positions if v is not None), default)
    for index, value in enumerate(positions):
        if value is not None:
            break
        held[index] = first_known

    # 2. A short median pass kills single-sample outliers (a false face hit).
    held = _median3(held)

    # 3. Deadzone: ignore movement too small to be worth a camera move.
    anchored: List[float] = []
    current = held[0]
    for value in held:
        if abs(value - current) > deadzone:
            current = value - deadzone if value > current else value + deadzone
        anchored.append(current)

    # 4. Zero-phase smoothing: exponential pass forward, then backward.
    # Each pass is seeded with padding so neither end of the clip is dragged
    # toward the middle while the filter converges.
    pad = min(len(anchored), max(2, int(round(1.0 / max(0.01, alpha)))))
    padded = [anchored[0]] * pad + list(anchored) + [anchored[-1]] * pad
    forward = _ema(padded, alpha)
    backward = list(reversed(_ema(list(reversed(forward)), alpha)))
    trimmed = backward[pad : pad + len(anchored)]
    return [clamp(v, 0.0, 1.0) for v in moving_average(trimmed, 3)]


def _ema(values: Sequence[float], alpha: float) -> List[float]:
    """Single-pole exponential filter."""
    if not values:
        return []
    alpha = clamp(alpha, 0.01, 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(out[-1] + alpha * (value - out[-1]))
    return out


def _median3(values: Sequence[float]) -> List[float]:
    if len(values) < 3:
        return list(values)
    out = [values[0]]
    for i in range(1, len(values) - 1):
        out.append(sorted(values[i - 1 : i + 2])[1])
    out.append(values[-1])
    return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_crop(
    ffmpeg: FFmpeg,
    source: Path,
    *,
    start: float,
    duration: float,
    source_w: int,
    source_h: int,
    target_ratio: float,
    layout: str = "auto",
    sample_fps: float = 4.0,
    alpha: float = 0.22,
    deadzone: float = 0.06,
) -> CropPlan:
    """Build the :class:`CropPlan` for one clip."""
    crop_w, crop_h = crop_size(source_w, source_h, target_ratio)

    # Faces sit in the upper half of the frame far more often than the centre,
    # so bias a vertical crop upwards instead of centring it.
    y = int(clamp((source_h - crop_h) * 0.35, 0, max(0, source_h - crop_h)))
    max_x = max(0, source_w - crop_w)
    centre_x = even(max_x / 2.0)

    plan = CropPlan(
        crop_w=crop_w,
        crop_h=crop_h,
        y=y,
        keyframes=[(0.0, centre_x)],
        layout=layout,
        source_w=source_w,
        source_h=source_h,
    )

    if layout != "auto" or max_x <= 2:
        return plan

    positions = estimate_subject_track(
        ffmpeg,
        source,
        start,
        duration,
        sample_fps=sample_fps,
        source_ratio=source_w / source_h if source_h else 16 / 9,
    )
    if not positions:
        return plan

    smoothed = smooth_positions(positions, alpha=alpha, deadzone=deadzone)
    keyframes: List[Tuple[float, int]] = []
    for index, fraction in enumerate(smoothed):
        # `fraction` is where the subject is; place the window so the subject
        # lands in the middle of it.
        x = even(clamp(fraction * source_w - crop_w / 2.0, 0, max_x))
        timestamp = index / sample_fps
        if keyframes and keyframes[-1][1] == x:
            continue  # sendcmd only needs a line when the value changes
        keyframes.append((timestamp, x))

    if not keyframes:
        keyframes = [(0.0, centre_x)]
    elif keyframes[0][0] > 0.0:
        keyframes.insert(0, (0.0, keyframes[0][1]))

    plan.keyframes = keyframes
    return plan


def sendcmd_script(plan: CropPlan) -> str:
    """Render a crop path as an ffmpeg ``sendcmd`` script."""
    lines = [f"{timestamp:.3f} crop x {x};" for timestamp, x in plan.keyframes]
    return "\n".join(lines) + "\n"


def write_sendcmd(plan: CropPlan, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(sendcmd_script(plan), encoding="utf-8")
    return destination
