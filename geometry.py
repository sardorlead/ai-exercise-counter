"""Geometric utilities: 2D/3D angles, torso normalization, One-Euro filter, and flexible splines."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class LandmarkPoint:
    """Represents a pose landmark in normalized and isotropic pixel coordinates."""

    x: float
    y: float
    z: float
    visibility: float
    px: float
    py: float
    pz: float

    @property
    def xy_px(self) -> Tuple[int, int]:
        return int(round(self.px)), int(round(self.py))

    @property
    def xy_float(self) -> Tuple[float, float]:
        return self.px, self.py


def midpoint(p1: LandmarkPoint, p2: LandmarkPoint) -> LandmarkPoint:
    """Compute anatomical midpoint between two landmarks."""
    return LandmarkPoint(
        x=0.5 * (p1.x + p2.x),
        y=0.5 * (p1.y + p2.y),
        z=0.5 * (p1.z + p2.z),
        visibility=min(p1.visibility, p2.visibility),
        px=0.5 * (p1.px + p2.px),
        py=0.5 * (p1.py + p2.py),
        pz=0.5 * (p1.pz + p2.pz),
    )


def calculate_angle_2d(a: LandmarkPoint, b: LandmarkPoint, c: LandmarkPoint) -> float:
    """Calculate interior 2D angle (degrees, [0, 180]) at vertex b formed by a-b-c."""
    ba = np.array([a.px - b.px, a.py - b.py], dtype=np.float64)
    bc = np.array([c.px - b.px, c.py - b.py], dtype=np.float64)

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def calculate_angle_3d(a: LandmarkPoint, b: LandmarkPoint, c: LandmarkPoint) -> float:
    """Calculate interior 3D angle (degrees, [0, 180]) at vertex b formed by a-b-c."""
    ba = np.array([a.px - b.px, a.py - b.py, a.pz - b.pz], dtype=np.float64)
    bc = np.array([c.px - b.px, c.py - b.py, c.pz - b.pz], dtype=np.float64)

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def euclidean_distance_2d(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Compute 2D Euclidean distance in pixel space."""
    return math.hypot(p1.px - p2.px, p1.py - p2.py)


def calculate_torso_length(shoulder: LandmarkPoint, hip: LandmarkPoint) -> float:
    """Compute 2D pixel torso length between shoulder and hip."""
    return euclidean_distance_2d(shoulder, hip)


def normalize_vertical_displacement(
    current_y: float, baseline_y: float, torso_length: float
) -> float:
    """Normalize downward vertical displacement by torso length (positive = moving down)."""
    if torso_length < 1e-6:
        return 0.0
    return (current_y - baseline_y) / torso_length


def angle_to_horizontal(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Acute angle [0, 90] between segment p1->p2 and horizontal axis (0 = flat)."""
    dx = abs(p2.px - p1.px)
    dy = abs(p2.py - p1.py)
    if dx < 1e-6 and dy < 1e-6:
        return 90.0
    return math.degrees(math.atan2(dy, dx))


def angle_to_vertical(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Acute angle [0, 90] between segment p1->p2 and vertical axis (0 = upright)."""
    dx = abs(p2.px - p1.px)
    dy = abs(p2.py - p1.py)
    if dx < 1e-6 and dy < 1e-6:
        return 90.0
    return math.degrees(math.atan2(dx, dy))


def clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to [low, high]."""
    return max(low, min(high, value))


def catmull_rom_spline(
    points: Sequence[Tuple[float, float]],
    samples_per_segment: int = 8,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Generate a smooth Centripetal Catmull-Rom spline passing exactly through every control point.
    Centripetal parameterization (alpha=0.5) prevents cusps and overshoot on sharp joint bends.
    Returns an int32 array of shape (N, 1, 2) ready for cv2.polylines.
    """
    n = len(points)
    if n < 2:
        return np.empty((0, 1, 2), dtype=np.int32)
    if n == 2:
        arr = np.array(points, dtype=np.float64)
        return np.round(arr).astype(np.int32).reshape((-1, 1, 2))

    pts = [np.array(p, dtype=np.float64) for p in points]
    # Extrapolate virtual endpoints for natural tangent flow at ends
    p_start = 2.0 * pts[0] - pts[1]
    p_end = 2.0 * pts[-1] - pts[-2]
    ext = [p_start] + pts + [p_end]

    curve_pts: List[np.ndarray] = []

    def _tj(ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
        dist = float(np.linalg.norm(pj - pi))
        return ti + max(dist, 1e-4) ** alpha

    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = _tj(t0, p0, p1)
        t2 = _tj(t1, p1, p2)
        t3 = _tj(t2, p2, p3)

        # Include endpoint only on the very last segment to avoid duplicate vertices
        endpoint = i == len(ext) - 3
        t_vals = np.linspace(t1, t2, samples_per_segment + (1 if endpoint else 0), endpoint=endpoint)

        for t in t_vals:
            a1 = ((t1 - t) / (t1 - t0)) * p0 + ((t - t0) / (t1 - t0)) * p1
            a2 = ((t2 - t) / (t2 - t1)) * p1 + ((t - t1) / (t2 - t1)) * p2
            a3 = ((t3 - t) / (t3 - t2)) * p2 + ((t - t2) / (t3 - t2)) * p3

            b1 = ((t2 - t) / (t2 - t0)) * a1 + ((t - t0) / (t2 - t0)) * a2
            b2 = ((t3 - t) / (t3 - t1)) * a2 + ((t - t1) / (t3 - t1)) * a3

            c = ((t2 - t) / (t2 - t1)) * b1 + ((t - t1) / (t2 - t1)) * b2
            curve_pts.append(c)

    if not curve_pts:
        return np.empty((0, 1, 2), dtype=np.int32)

    out = np.vstack(curve_pts)
    return np.round(out).astype(np.int32).reshape((-1, 1, 2))


class EMAFilter:
    """Scalar Exponential Moving Average filter."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, measurement: float) -> float:
        if self._value is None:
            self._value = measurement
        else:
            self._value = self._alpha * measurement + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


class OneEuroFilter:
    """
    Velocity-adaptive One-Euro Filter.
    Heavy smoothing when stationary (eliminates jitter), low latency during fast motion.
    """

    def __init__(
        self, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    @property
    def value(self) -> Optional[float]:
        return self._x_prev

    @property
    def velocity(self) -> float:
        return self._dx_prev

    def update(self, measurement: float, timestamp: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = measurement
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return measurement

        dt = max(timestamp - self._t_prev, 1e-4)
        self._t_prev = timestamp

        dx = (measurement - self._x_prev) / dt
        a_d = self._alpha(dt, self._d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * measurement + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class LandmarkOneEuroFilter:
    """Applies independent One-Euro filtering to normalized (x, y, z) coordinates of a landmark."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fz = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def update(
        self, x: float, y: float, z: float, timestamp: float
    ) -> Tuple[float, float, float]:
        return (
            self._fx.update(x, timestamp),
            self._fy.update(y, timestamp),
            self._fz.update(z, timestamp),
        )

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()