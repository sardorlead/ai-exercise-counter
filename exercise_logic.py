"""Strict Finite State Machine (FSM), anti-speed-cheat tempo gate, and full-ROM validators."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional, Tuple
import math

from config import CONTROLLER, FILTER, PUSHUP, SQUAT
from geometry import (
    EMAFilter,
    LandmarkPoint,
    OneEuroFilter,
    angle_to_horizontal,
    angle_to_vertical,
    calculate_angle_2d,
    calculate_torso_length,
    clamp,
    normalize_vertical_displacement,
)
from pose_detector import PoseFrameResult


class ExerciseMode(str, Enum):
    PUSHUP = "PUSHUP"
    SQUAT = "SQUAT"
    AUTO = "AUTO"


class ExerciseType(str, Enum):
    NONE = "NONE"
    PUSHUP = "PUSHUP"
    SQUAT = "SQUAT"


class RepState(str, Enum):
    IDLE = "IDLE"
    UP_READY = "UP_READY"
    GOING_DOWN = "GOING_DOWN"
    BOTTOM_REACHED = "BOTTOM_REACHED"
    GOING_UP = "GOING_UP"
    REP_COMPLETE = "REP_COMPLETE"


@dataclass
class AnalysisSnapshot:
    """Immutable per-frame telemetry snapshot for HUD rendering and audio cues."""

    mode: ExerciseMode
    active_exercise: ExerciseType
    state: RepState
    pushup_count: int
    squat_count: int
    rejected_count: int
    last_rep_quality: Optional[int]
    last_rep_duration: Optional[float]
    rep_just_completed: bool
    completed_exercise: Optional[ExerciseType]
    depth_progress: float  # 0.0 to 100.0
    primary_angle: Optional[float]
    secondary_angle: Optional[float]
    normalized_drop: float
    target_depth_y: Optional[float]
    target_anchor_x: Optional[float]
    target_reached: bool
    feedback_message: str
    feedback_level: str  # "info" | "warning" | "error" | "success"
    auto_candidate: Optional[ExerciseType]
    auto_progress: float
    active_side: str
    landmarks_valid: bool


class AnchorStabilityMonitor:
    """Tracks recent 2D positions of planted wrists/ankles to reject walking or hand waving."""

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._history: Deque[Tuple[float, float, float]] = deque()

    def update(self, now: float, px: float, py: float, torso_length: float) -> float:
        self._history.append((now, px, py))
        cutoff = now - self._window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 2 or torso_length < 1e-6:
            return 0.0

        ref_x, ref_y = self._history[0][1], self._history[0][2]
        max_dist = max(math.hypot(hx - ref_x, hy - ref_y) for _, hx, hy in self._history)
        return max_dist / torso_length

    def reset(self) -> None:
        self._history.clear()


class PushupTracker:
    """Strict Push-up FSM enforcing full elbow depth, vertical shoulder travel, alignment, and tempo."""

    def __init__(self) -> None:
        self.count: int = 0
        self.rejected_count: int = 0
        self.last_rep_quality: Optional[int] = None
        self.last_rep_duration: Optional[float] = None
        self.state: RepState = RepState.IDLE

        self._elbow_flt = OneEuroFilter(
            FILTER.angle_min_cutoff, FILTER.angle_beta, FILTER.angle_d_cutoff
        )
        self._align_flt = OneEuroFilter(
            FILTER.angle_min_cutoff, FILTER.angle_beta, FILTER.angle_d_cutoff
        )
        self._shoulder_y_flt = OneEuroFilter(
            FILTER.pos_min_cutoff, FILTER.pos_beta, FILTER.pos_d_cutoff
        )
        self._torso_ema = EMAFilter(FILTER.torso_ema_alpha)
        self._wrist_monitor = AnchorStabilityMonitor(FILTER.anchor_window_sec)

        self._baseline_shoulder_y: Optional[float] = None
        self._min_elbow_seen: float = 180.0
        self._max_drop_seen: float = 0.0
        self._min_align_seen: float = 180.0
        self._max_elbow_during_ascent: float = 0.0
        self._bottom_dwell_start: Optional[float] = None
        self._top_dwell_start: Optional[float] = None
        self._bottom_confirmed_time: Optional[float] = None
        self._rep_start_time: Optional[float] = None
        self._rep_complete_time: Optional[float] = None
        self._bad_alignment_start: Optional[float] = None
        self._form_compromised: bool = False

    def reset_state(self, clear_filters: bool = False) -> None:
        self.state = RepState.IDLE
        self._baseline_shoulder_y = None
        self._min_elbow_seen = 180.0
        self._max_drop_seen = 0.0
        self._min_align_seen = 180.0
        self._max_elbow_during_ascent = 0.0
        self._bottom_dwell_start = None
        self._top_dwell_start = None
        self._bottom_confirmed_time = None
        self._rep_start_time = None
        self._rep_complete_time = None
        self._bad_alignment_start = None
        self._form_compromised = False
        if clear_filters:
            self._elbow_flt.reset()
            self._align_flt.reset()
            self._shoulder_y_flt.reset()
            self._torso_ema.reset()
            self._wrist_monitor.reset()

    def reset_counter(self) -> None:
        self.count = 0
        self.rejected_count = 0
        self.last_rep_quality = None
        self.last_rep_duration = None
        self.reset_state(clear_filters=True)

    @staticmethod
    def evaluate_posture_gate(
        joints: Dict[str, LandmarkPoint],
        torso_len: float,
        wrist_drift: float,
        alignment_angle: float,
    ) -> Tuple[bool, bool, str]:
        shoulder = joints["shoulder"]
        wrist = joints["wrist"]
        hip = joints["hip"]

        if angle_to_horizontal(shoulder, hip) >= PUSHUP.max_torso_horizontal_angle:
            return False, False, "not_horizontal"

        if (wrist.py - shoulder.py) < PUSHUP.min_wrist_below_shoulder_ratio * torso_len:
            return False, False, "wrists_above_shoulders"

        if wrist_drift > PUSHUP.max_wrist_drift_ratio:
            return False, False, "wrists_moving"

        alignment_ok = (
            PUSHUP.min_body_alignment_angle
            <= alignment_angle
            <= PUSHUP.max_body_alignment_angle
        )
        return True, alignment_ok, "ok" if alignment_ok else "bad_alignment"

    def _compute_quality_score(self, duration: float) -> int:
        depth_score = clamp(
            (PUSHUP.down_elbow_angle + 8.0 - self._min_elbow_seen) / 20.0, 0.0, 1.0
        )
        drop_score = clamp(self._max_drop_seen / (PUSHUP.min_shoulder_drop_ratio + 0.08), 0.0, 1.0)
        align_score = clamp((self._min_align_seen - 148.0) / 24.0, 0.0, 1.0)
        tempo_score = clamp((duration - 0.80) / 0.90, 0.0, 1.0)
        total = 100.0 * (0.35 * depth_score + 0.30 * drop_score + 0.20 * align_score + 0.15 * tempo_score)
        return int(round(clamp(total, 50.0, 100.0)))

    def update(
        self, joints: Dict[str, LandmarkPoint], now: float, frame_height: int
    ) -> Tuple[
        bool, float, float, float, float, Optional[float], Optional[float], bool, Optional[Tuple[str, str]]
    ]:
        shoulder = joints["shoulder"]
        elbow = joints["elbow"]
        wrist = joints["wrist"]
        hip = joints["hip"]
        ankle = joints["ankle"]

        raw_torso = calculate_torso_length(shoulder, hip)
        if raw_torso < FILTER.min_torso_height_ratio * frame_height:
            self.reset_state(clear_filters=True)
            return False, 0.0, 0.0, 0.0, 0.0, None, None, False, (
                "No valid exercise posture detected",
                "error",
            )

        torso_len = self._torso_ema.update(raw_torso)
        elbow_angle = self._elbow_flt.update(calculate_angle_2d(shoulder, elbow, wrist), now)
        align_angle = self._align_flt.update(calculate_angle_2d(shoulder, hip, ankle), now)
        shoulder_y = self._shoulder_y_flt.update(shoulder.py, now)
        wrist_drift = self._wrist_monitor.update(now, wrist.px, wrist.py, torso_len)

        is_plank, align_ok, _ = self.evaluate_posture_gate(
            joints, torso_len, wrist_drift, align_angle
        )

        if not is_plank:
            self.reset_state(clear_filters=False)
            return False, 0.0, elbow_angle, align_angle, 0.0, None, None, False, (
                "No valid exercise posture detected",
                "error",
            )

        feedback: Optional[Tuple[str, str]] = None

        if not align_ok:
            feedback = ("Straighten your back", "warning")
            if self._bad_alignment_start is None:
                self._bad_alignment_start = now
            if self.state in (RepState.GOING_DOWN, RepState.BOTTOM_REACHED, RepState.GOING_UP):
                self._form_compromised = True
                if now - self._bad_alignment_start >= PUSHUP.bad_alignment_abort_sec:
                    self.rejected_count += 1
                    self.reset_state(clear_filters=False)
            else:
                self.state = RepState.IDLE
            return False, 0.0, elbow_angle, align_angle, 0.0, None, None, False, feedback
        else:
            self._bad_alignment_start = None

        norm_drop = 0.0
        target_depth_y: Optional[float] = None
        if self._baseline_shoulder_y is not None:
            norm_drop = normalize_vertical_displacement(
                shoulder_y, self._baseline_shoulder_y, torso_len
            )
            target_depth_y = self._baseline_shoulder_y + PUSHUP.min_shoulder_drop_ratio * torso_len

        angle_prog = clamp(
            (PUSHUP.up_elbow_angle - elbow_angle)
            / (PUSHUP.up_elbow_angle - PUSHUP.down_elbow_angle),
            0.0,
            1.0,
        )
        drop_prog = clamp(norm_drop / PUSHUP.min_shoulder_drop_ratio, 0.0, 1.0)
        depth_pct = min(angle_prog, drop_prog) * 100.0 if self._baseline_shoulder_y is not None else 0.0
        target_reached = (
            elbow_angle <= PUSHUP.down_elbow_angle
            and norm_drop >= PUSHUP.min_shoulder_drop_ratio
        )

        rep_completed = False

        # --- Strict 6-State FSM ---
        if self.state == RepState.IDLE:
            if elbow_angle >= PUSHUP.up_elbow_angle and abs(self._elbow_flt.velocity) < 65.0:
                if self._top_dwell_start is None:
                    self._top_dwell_start = now
                elif now - self._top_dwell_start >= PUSHUP.min_top_dwell_sec:
                    self.state = RepState.UP_READY
                    self._baseline_shoulder_y = shoulder_y
                    self._top_dwell_start = None
                    self._form_compromised = False
            else:
                self._top_dwell_start = None

        elif self.state == RepState.UP_READY:
            if elbow_angle >= PUSHUP.up_elbow_angle:
                # Only update baseline gently when truly stationary (prevents bounce inflation)
                assert self._baseline_shoulder_y is not None
                if abs(self._elbow_flt.velocity) < 35.0 and abs(shoulder_y - self._baseline_shoulder_y) < 0.05 * torso_len:
                    self._baseline_shoulder_y = 0.92 * self._baseline_shoulder_y + 0.08 * shoulder_y
            elif elbow_angle < PUSHUP.up_elbow_angle - PUSHUP.hysteresis_deg:
                self.state = RepState.GOING_DOWN
                self._min_elbow_seen = elbow_angle
                self._max_drop_seen = norm_drop
                self._min_align_seen = align_angle
                self._bottom_dwell_start = None
                self._top_dwell_start = None
                self._rep_start_time = now
                self._form_compromised = False

        elif self.state == RepState.GOING_DOWN:
            if self._rep_start_time and (now - self._rep_start_time > PUSHUP.max_rep_duration_sec):
                self.reset_state(clear_filters=False)
                return False, 0.0, elbow_angle, align_angle, norm_drop, target_depth_y, shoulder.px, False, feedback

            self._min_elbow_seen = min(self._min_elbow_seen, elbow_angle)
            self._max_drop_seen = max(self._max_drop_seen, norm_drop)
            self._min_align_seen = min(self._min_align_seen, align_angle)

            if target_reached:
                if self._bottom_dwell_start is None:
                    self._bottom_dwell_start = now
                elif now - self._bottom_dwell_start >= PUSHUP.min_bottom_dwell_sec:
                    self.state = RepState.BOTTOM_REACHED
                    self._bottom_confirmed_time = now
                    self._max_elbow_during_ascent = elbow_angle
            else:
                self._bottom_dwell_start = None
                # Half-rep check: user reversed upward before hitting strict depth + dwell
                reversed_early = (
                    elbow_angle >= self._min_elbow_seen + PUSHUP.reversal_threshold_deg
                    and elbow_angle >= PUSHUP.up_elbow_angle - PUSHUP.hysteresis_deg
                )
                if reversed_early:
                    self.rejected_count += 1
                    feedback = ("Go lower! Full depth required", "warning")
                    self.state = RepState.UP_READY
                    self._min_elbow_seen = 180.0
                    self._max_drop_seen = 0.0
                    self._rep_start_time = None

        elif self.state == RepState.BOTTOM_REACHED:
            self._min_elbow_seen = min(self._min_elbow_seen, elbow_angle)
            self._max_drop_seen = max(self._max_drop_seen, norm_drop)
            self._min_align_seen = min(self._min_align_seen, align_angle)

            if self._bottom_confirmed_time and (
                now - self._bottom_confirmed_time > PUSHUP.max_bottom_hold_sec
            ):
                self.reset_state(clear_filters=False)
                return False, 0.0, elbow_angle, align_angle, norm_drop, target_depth_y, shoulder.px, False, feedback

            if elbow_angle > PUSHUP.down_elbow_angle + PUSHUP.hysteresis_deg:
                self.state = RepState.GOING_UP
                self._max_elbow_during_ascent = elbow_angle
                self._top_dwell_start = None

        elif self.state == RepState.GOING_UP:
            if self._rep_start_time and (now - self._rep_start_time > PUSHUP.max_rep_duration_sec):
                self.reset_state(clear_filters=False)
                return False, 0.0, elbow_angle, align_angle, norm_drop, target_depth_y, shoulder.px, False, feedback

            self._max_elbow_during_ascent = max(self._max_elbow_during_ascent, elbow_angle)
            self._min_align_seen = min(self._min_align_seen, align_angle)

            top_reached = (
                elbow_angle >= PUSHUP.up_elbow_angle
                and norm_drop <= PUSHUP.top_return_ratio
            )
            if top_reached:
                if self._top_dwell_start is None:
                    self._top_dwell_start = now
                elif now - self._top_dwell_start >= PUSHUP.min_top_dwell_sec:
                    duration = (now - self._rep_start_time) if self._rep_start_time else 0.0
                    if self._form_compromised:
                        self.rejected_count += 1
                        self.state = RepState.UP_READY
                        feedback = ("Rep ignored: keep back straight", "warning")
                    elif duration < PUSHUP.min_rep_duration_sec:
                        self.rejected_count += 1
                        self.state = RepState.UP_READY
                        feedback = ("Too fast! Control each rep", "warning")
                    else:
                        self.count += 1
                        rep_completed = True
                        self.last_rep_duration = duration
                        self.last_rep_quality = self._compute_quality_score(duration)
                        self.state = RepState.REP_COMPLETE
                        self._rep_complete_time = now
                        feedback = (f"Clean push-up! ({self.last_rep_quality}%)", "success")

                    self._baseline_shoulder_y = shoulder_y
                    self._rep_start_time = None
                    self._top_dwell_start = None
            else:
                self._top_dwell_start = None
                if elbow_angle < self._max_elbow_during_ascent - PUSHUP.reversal_threshold_deg:
                    self.rejected_count += 1
                    feedback = ("Lock out arms at the top", "warning")
                    self.state = RepState.GOING_DOWN
                    self._min_elbow_seen = elbow_angle
                    self._max_drop_seen = norm_drop
                    self._bottom_dwell_start = None
                    self._rep_start_time = now

        elif self.state == RepState.REP_COMPLETE:
            if elbow_angle < PUSHUP.up_elbow_angle - PUSHUP.hysteresis_deg:
                self.state = RepState.GOING_DOWN
                self._min_elbow_seen = elbow_angle
                self._max_drop_seen = norm_drop
                self._min_align_seen = align_angle
                self._bottom_dwell_start = None
                self._top_dwell_start = None
                self._rep_start_time = now
                self._form_compromised = False
            elif self._rep_complete_time and (
                now - self._rep_complete_time >= CONTROLLER.rep_complete_display_sec
            ):
                self.state = RepState.UP_READY

        return (
            rep_completed,
            depth_pct,
            elbow_angle,
            align_angle,
            norm_drop,
            target_depth_y,
            shoulder.px,
            target_reached,
            feedback,
        )


class SquatTracker:
    """Strict Squat FSM enforcing full knee depth, hip vertical travel, dwell, and controlled tempo."""

    def __init__(self) -> None:
        self.count: int = 0
        self.rejected_count: int = 0
        self.last_rep_quality: Optional[int] = None
        self.last_rep_duration: Optional[float] = None
        self.state: RepState = RepState.IDLE

        self._knee_flt = OneEuroFilter(
            FILTER.angle_min_cutoff, FILTER.angle_beta, FILTER.angle_d_cutoff
        )
        self._torso_vert_flt = OneEuroFilter(
            FILTER.angle_min_cutoff, FILTER.angle_beta, FILTER.angle_d_cutoff
        )
        self._hip_y_flt = OneEuroFilter(
            FILTER.pos_min_cutoff, FILTER.pos_beta, FILTER.pos_d_cutoff
        )
        self._torso_ema = EMAFilter(FILTER.torso_ema_alpha)
        self._ankle_monitor = AnchorStabilityMonitor(FILTER.anchor_window_sec)

        self._baseline_hip_y: Optional[float] = None
        self._min_knee_seen: float = 180.0
        self._max_drop_seen: float = 0.0
        self._max_lean_seen: float = 0.0
        self._max_knee_during_ascent: float = 0.0
        self._bottom_dwell_start: Optional[float] = None
        self._top_dwell_start: Optional[float] = None
        self._bottom_confirmed_time: Optional[float] = None
        self._rep_start_time: Optional[float] = None
        self._rep_complete_time: Optional[float] = None

    def reset_state(self, clear_filters: bool = False) -> None:
        self.state = RepState.IDLE
        self._baseline_hip_y = None
        self._min_knee_seen = 180.0
        self._max_drop_seen = 0.0
        self._max_lean_seen = 0.0
        self._max_knee_during_ascent = 0.0
        self._bottom_dwell_start = None
        self._top_dwell_start = None
        self._bottom_confirmed_time = None
        self._rep_start_time = None
        self._rep_complete_time = None
        if clear_filters:
            self._knee_flt.reset()
            self._torso_vert_flt.reset()
            self._hip_y_flt.reset()
            self._torso_ema.reset()
            self._ankle_monitor.reset()

    def reset_counter(self) -> None:
        self.count = 0
        self.rejected_count = 0
        self.last_rep_quality = None
        self.last_rep_duration = None
        self.reset_state(clear_filters=True)

    @staticmethod
    def evaluate_posture_gate(
        joints: Dict[str, LandmarkPoint],
        torso_len: float,
        ankle_drift: float,
        torso_vert_angle: float,
    ) -> Tuple[bool, str]:
        shoulder = joints["shoulder"]
        hip = joints["hip"]
        ankle = joints["ankle"]

        min_sep = SQUAT.min_vertical_stack_ratio * torso_len

        if (hip.py - shoulder.py) < min_sep:
            return False, "shoulders_not_above_hips"

        if (ankle.py - hip.py) < min_sep:
            return False, "hips_not_above_ankles"

        if torso_vert_angle > SQUAT.max_torso_vertical_angle:
            return False, "torso_too_horizontal"

        if ankle_drift > SQUAT.max_ankle_drift_ratio:
            return False, "ankles_moving"

        return True, "ok"

    def _compute_quality_score(self, duration: float) -> int:
        depth_score = clamp((SQUAT.down_knee_angle + 8.0 - self._min_knee_seen) / 22.0, 0.0, 1.0)
        drop_score = clamp(self._max_drop_seen / (SQUAT.min_hip_drop_ratio + 0.10), 0.0, 1.0)
        lean_score = clamp((SQUAT.max_torso_vertical_angle - self._max_lean_seen) / 28.0, 0.0, 1.0)
        tempo_score = clamp((duration - 0.85) / 1.0, 0.0, 1.0)
        total = 100.0 * (0.35 * depth_score + 0.35 * drop_score + 0.15 * lean_score + 0.15 * tempo_score)
        return int(round(clamp(total, 50.0, 100.0)))

    def update(
        self, joints: Dict[str, LandmarkPoint], now: float, frame_height: int
    ) -> Tuple[
        bool, float, float, float, float, Optional[float], Optional[float], bool, Optional[Tuple[str, str]]
    ]:
        shoulder = joints["shoulder"]
        hip = joints["hip"]
        knee = joints["knee"]
        ankle = joints["ankle"]

        raw_torso = calculate_torso_length(shoulder, hip)
        if raw_torso < FILTER.min_torso_height_ratio * frame_height:
            self.reset_state(clear_filters=True)
            return False, 0.0, 0.0, 0.0, 0.0, None, None, False, (
                "No valid exercise posture detected",
                "error",
            )

        torso_len = self._torso_ema.update(raw_torso)
        knee_angle = self._knee_flt.update(calculate_angle_2d(hip, knee, ankle), now)
        torso_vert = self._torso_vert_flt.update(angle_to_vertical(shoulder, hip), now)
        hip_y = self._hip_y_flt.update(hip.py, now)
        ankle_drift = self._ankle_monitor.update(now, ankle.px, ankle.py, torso_len)

        gate_ok, _ = self.evaluate_posture_gate(joints, torso_len, ankle_drift, torso_vert)
        if not gate_ok:
            self.reset_state(clear_filters=False)
            return False, 0.0, knee_angle, torso_vert, 0.0, None, None, False, (
                "No valid exercise posture detected",
                "error",
            )

        norm_drop = 0.0
        target_depth_y: Optional[float] = None
        if self._baseline_hip_y is not None:
            norm_drop = normalize_vertical_displacement(hip_y, self._baseline_hip_y, torso_len)
            target_depth_y = self._baseline_hip_y + SQUAT.min_hip_drop_ratio * torso_len

        # Strict Dual-AND Depth Progress (removed the OR bypass that allowed shallow squats)
        angle_prog = clamp(
            (SQUAT.up_knee_angle - knee_angle) / (SQUAT.up_knee_angle - SQUAT.down_knee_angle),
            0.0,
            1.0,
        )
        drop_prog = clamp(norm_drop / SQUAT.min_hip_drop_ratio, 0.0, 1.0)
        depth_pct = min(angle_prog, drop_prog) * 100.0 if self._baseline_hip_y is not None else 0.0
        target_reached = (
            knee_angle <= SQUAT.down_knee_angle
            and norm_drop >= SQUAT.min_hip_drop_ratio
        )

        feedback: Optional[Tuple[str, str]] = None
        rep_completed = False

        # --- Strict 6-State FSM ---
        if self.state == RepState.IDLE:
            if knee_angle >= SQUAT.up_knee_angle and abs(self._knee_flt.velocity) < 65.0:
                if self._top_dwell_start is None:
                    self._top_dwell_start = now
                elif now - self._top_dwell_start >= SQUAT.min_top_dwell_sec:
                    self.state = RepState.UP_READY
                    self._baseline_hip_y = hip_y
                    self._top_dwell_start = None
            else:
                self._top_dwell_start = None

        elif self.state == RepState.UP_READY:
            if knee_angle >= SQUAT.up_knee_angle:
                assert self._baseline_hip_y is not None
                if abs(self._knee_flt.velocity) < 35.0 and abs(hip_y - self._baseline_hip_y) < 0.05 * torso_len:
                    self._baseline_hip_y = 0.92 * self._baseline_hip_y + 0.08 * hip_y
            elif knee_angle < SQUAT.up_knee_angle - SQUAT.hysteresis_deg:
                self.state = RepState.GOING_DOWN
                self._min_knee_seen = knee_angle
                self._max_drop_seen = norm_drop
                self._max_lean_seen = torso_vert
                self._bottom_dwell_start = None
                self._top_dwell_start = None
                self._rep_start_time = now

        elif self.state == RepState.GOING_DOWN:
            if self._rep_start_time and (now - self._rep_start_time > SQUAT.max_rep_duration_sec):
                self.reset_state(clear_filters=False)
                return False, 0.0, knee_angle, torso_vert, norm_drop, target_depth_y, hip.px, False, feedback

            self._min_knee_seen = min(self._min_knee_seen, knee_angle)
            self._max_drop_seen = max(self._max_drop_seen, norm_drop)
            self._max_lean_seen = max(self._max_lean_seen, torso_vert)

            if target_reached:
                if self._bottom_dwell_start is None:
                    self._bottom_dwell_start = now
                elif now - self._bottom_dwell_start >= SQUAT.min_bottom_dwell_sec:
                    self.state = RepState.BOTTOM_REACHED
                    self._bottom_confirmed_time = now
                    self._max_knee_during_ascent = knee_angle
            else:
                self._bottom_dwell_start = None
                reversed_early = (
                    knee_angle >= self._min_knee_seen + SQUAT.reversal_threshold_deg
                    and knee_angle >= SQUAT.up_knee_angle - SQUAT.hysteresis_deg
                )
                if reversed_early:
                    self.rejected_count += 1
                    feedback = ("Squat deeper! Reach target line", "warning")
                    self.state = RepState.UP_READY
                    self._min_knee_seen = 180.0
                    self._max_drop_seen = 0.0
                    self._rep_start_time = None

        elif self.state == RepState.BOTTOM_REACHED:
            self._min_knee_seen = min(self._min_knee_seen, knee_angle)
            self._max_drop_seen = max(self._max_drop_seen, norm_drop)
            self._max_lean_seen = max(self._max_lean_seen, torso_vert)

            if self._bottom_confirmed_time and (
                now - self._bottom_confirmed_time > SQUAT.max_bottom_hold_sec
            ):
                self.reset_state(clear_filters=False)
                return False, 0.0, knee_angle, torso_vert, norm_drop, target_depth_y, hip.px, False, feedback

            if knee_angle > SQUAT.down_knee_angle + SQUAT.hysteresis_deg:
                self.state = RepState.GOING_UP
                self._max_knee_during_ascent = knee_angle
                self._top_dwell_start = None

        elif self.state == RepState.GOING_UP:
            if self._rep_start_time and (now - self._rep_start_time > SQUAT.max_rep_duration_sec):
                self.reset_state(clear_filters=False)
                return False, 0.0, knee_angle, torso_vert, norm_drop, target_depth_y, hip.px, False, feedback

            self._max_knee_during_ascent = max(self._max_knee_during_ascent, knee_angle)

            top_reached = (
                knee_angle >= SQUAT.up_knee_angle
                and norm_drop <= SQUAT.top_return_ratio
            )
            if top_reached:
                if self._top_dwell_start is None:
                    self._top_dwell_start = now
                elif now - self._top_dwell_start >= SQUAT.min_top_dwell_sec:
                    duration = (now - self._rep_start_time) if self._rep_start_time else 0.0
                    if duration < SQUAT.min_rep_duration_sec:
                        self.rejected_count += 1
                        self.state = RepState.UP_READY
                        feedback = ("Too fast! Control each squat", "warning")
                    else:
                        self.count += 1
                        rep_completed = True
                        self.last_rep_duration = duration
                        self.last_rep_quality = self._compute_quality_score(duration)
                        self.state = RepState.REP_COMPLETE
                        self._rep_complete_time = now
                        feedback = (f"Clean squat! ({self.last_rep_quality}%)", "success")

                    self._baseline_hip_y = hip_y
                    self._rep_start_time = None
                    self._top_dwell_start = None
            else:
                self._top_dwell_start = None
                if knee_angle < self._max_knee_during_ascent - SQUAT.reversal_threshold_deg:
                    self.rejected_count += 1
                    feedback = ("Stand all the way up", "warning")
                    self.state = RepState.GOING_DOWN
                    self._min_knee_seen = knee_angle
                    self._max_drop_seen = norm_drop
                    self._bottom_dwell_start = None
                    self._rep_start_time = now

        elif self.state == RepState.REP_COMPLETE:
            if knee_angle < SQUAT.up_knee_angle - SQUAT.hysteresis_deg:
                self.state = RepState.GOING_DOWN
                self._min_knee_seen = knee_angle
                self._max_drop_seen = norm_drop
                self._max_lean_seen = torso_vert
                self._bottom_dwell_start = None
                self._top_dwell_start = None
                self._rep_start_time = now
            elif self._rep_complete_time and (
                now - self._rep_complete_time >= CONTROLLER.rep_complete_display_sec
            ):
                self.state = RepState.UP_READY

        return (
            rep_completed,
            depth_pct,
            knee_angle,
            torso_vert,
            norm_drop,
            target_depth_y,
            hip.px,
            target_reached,
            feedback,
        )


class ExerciseController:
    """Top-level coordinator managing Push-up, Squat, and debounced Auto-detect modes."""

    def __init__(self, initial_mode: ExerciseMode = ExerciseMode.AUTO) -> None:
        self._mode: ExerciseMode = initial_mode
        self._active_exercise: ExerciseType = self._default_exercise_for_mode(initial_mode)

        self._pushup = PushupTracker()
        self._squat = SquatTracker()

        self._auto_wrist_monitor = AnchorStabilityMonitor(FILTER.anchor_window_sec)
        self._auto_ankle_monitor = AnchorStabilityMonitor(FILTER.anchor_window_sec)
        self._auto_candidate: Optional[ExerciseType] = None
        self._auto_candidate_since: Optional[float] = None

        self._invalid_since: Optional[float] = None
        self._sticky_feedback: str = ""
        self._sticky_level: str = "info"
        self._sticky_until: float = 0.0

    @staticmethod
    def _default_exercise_for_mode(mode: ExerciseMode) -> ExerciseType:
        if mode == ExerciseMode.PUSHUP:
            return ExerciseType.PUSHUP
        if mode == ExerciseMode.SQUAT:
            return ExerciseType.SQUAT
        return ExerciseType.NONE

    def set_mode(self, mode: ExerciseMode, now: float) -> None:
        self._mode = mode
        self._auto_candidate = None
        self._auto_candidate_since = None
        self._pushup.reset_state(clear_filters=True)
        self._squat.reset_state(clear_filters=True)
        self._active_exercise = self._default_exercise_for_mode(mode)
        self._set_sticky_feedback(f"Mode: {mode.value}", "info", now)

    def reset_counters(self, now: float) -> None:
        self._pushup.reset_counter()
        self._squat.reset_counter()
        self._auto_candidate = None
        self._auto_candidate_since = None
        self._set_sticky_feedback("Counters reset", "info", now)

    def _set_sticky_feedback(self, message: str, level: str, now: float) -> None:
        self._sticky_feedback = message
        self._sticky_level = level
        self._sticky_until = now + CONTROLLER.feedback_display_sec

    def _resolve_feedback(
        self, frame_feedback: Optional[Tuple[str, str]], now: float
    ) -> Tuple[str, str]:
        if frame_feedback is not None:
            msg, lvl = frame_feedback
            if lvl in ("warning", "success"):
                self._set_sticky_feedback(msg, lvl, now)
                return msg, lvl
            if now < self._sticky_until:
                return self._sticky_feedback, self._sticky_level
            return msg, lvl

        if now < self._sticky_until:
            return self._sticky_feedback, self._sticky_level
        return "", "info"

    def _update_auto_detector(self, pose: PoseFrameResult, now: float) -> float:
        active_tracker = (
            self._pushup
            if self._active_exercise == ExerciseType.PUSHUP
            else self._squat
            if self._active_exercise == ExerciseType.SQUAT
            else None
        )
        if active_tracker is not None and active_tracker.state in (
            RepState.GOING_DOWN,
            RepState.BOTTOM_REACHED,
            RepState.GOING_UP,
        ):
            self._auto_candidate = None
            self._auto_candidate_since = None
            return 0.0

        joints = pose.active_joints
        shoulder, elbow, wrist = joints["shoulder"], joints["elbow"], joints["wrist"]
        hip, knee, ankle = joints["hip"], joints["knee"], joints["ankle"]
        torso_len = calculate_torso_length(shoulder, hip)

        if torso_len < FILTER.min_torso_height_ratio * pose.frame_height:
            self._auto_candidate = None
            self._auto_candidate_since = None
            return 0.0

        wrist_drift = self._auto_wrist_monitor.update(now, wrist.px, wrist.py, torso_len)
        ankle_drift = self._auto_ankle_monitor.update(now, ankle.px, ankle.py, torso_len)

        elbow_angle = calculate_angle_2d(shoulder, elbow, wrist)
        align_angle = calculate_angle_2d(shoulder, hip, ankle)
        knee_angle = calculate_angle_2d(hip, knee, ankle)
        torso_vert = angle_to_vertical(shoulder, hip)

        is_plank, align_ok, _ = PushupTracker.evaluate_posture_gate(
            joints, torso_len, wrist_drift, align_angle
        )
        pushup_ready = is_plank and align_ok and elbow_angle >= (PUSHUP.up_elbow_angle - 10.0)

        squat_gate_ok, _ = SquatTracker.evaluate_posture_gate(
            joints, torso_len, ankle_drift, torso_vert
        )
        squat_ready = squat_gate_ok and knee_angle >= (SQUAT.up_knee_angle - 10.0)

        detected: Optional[ExerciseType] = None
        if pushup_ready and not squat_ready:
            detected = ExerciseType.PUSHUP
        elif squat_ready and not pushup_ready:
            detected = ExerciseType.SQUAT

        if detected is None or detected == self._active_exercise:
            self._auto_candidate = None
            self._auto_candidate_since = None
            return 0.0

        if detected != self._auto_candidate:
            self._auto_candidate = detected
            self._auto_candidate_since = now
            return 0.0

        assert self._auto_candidate_since is not None
        elapsed = now - self._auto_candidate_since
        progress = clamp(elapsed / CONTROLLER.auto_detect_debounce_sec, 0.0, 1.0)

        if elapsed >= CONTROLLER.auto_detect_debounce_sec:
            self._active_exercise = detected
            self._pushup.reset_state(clear_filters=True)
            self._squat.reset_state(clear_filters=True)
            self._auto_candidate = None
            self._auto_candidate_since = None
            self._set_sticky_feedback(f"Auto-locked: {detected.value}", "info", now)
            return 0.0

        return progress

    def update(self, pose: PoseFrameResult, now: float) -> AnalysisSnapshot:
        if pose.side_switched:
            self._pushup.reset_state(clear_filters=True)
            self._squat.reset_state(clear_filters=True)
            self._auto_wrist_monitor.reset()
            self._auto_ankle_monitor.reset()

        total_rejected = self._pushup.rejected_count + self._squat.rejected_count

        if not pose.detected or not pose.landmarks_valid:
            if self._invalid_since is None:
                self._invalid_since = now
            if now - self._invalid_since >= CONTROLLER.occlusion_grace_sec:
                self._pushup.reset_state(clear_filters=True)
                self._squat.reset_state(clear_filters=True)
                self._auto_candidate = None
                self._auto_candidate_since = None
                if self._mode == ExerciseMode.AUTO:
                    self._active_exercise = ExerciseType.NONE

            msg, lvl = self._resolve_feedback(
                ("No valid exercise posture detected", "error"), now
            )
            return AnalysisSnapshot(
                mode=self._mode,
                active_exercise=self._active_exercise,
                state=RepState.IDLE,
                pushup_count=self._pushup.count,
                squat_count=self._squat.count,
                rejected_count=total_rejected,
                last_rep_quality=None,
                last_rep_duration=None,
                rep_just_completed=False,
                completed_exercise=None,
                depth_progress=0.0,
                primary_angle=None,
                secondary_angle=None,
                normalized_drop=0.0,
                target_depth_y=None,
                target_anchor_x=None,
                target_reached=False,
                feedback_message=msg,
                feedback_level=lvl,
                auto_candidate=None,
                auto_progress=0.0,
                active_side=pose.active_side,
                landmarks_valid=False,
            )

        self._invalid_since = None
        auto_progress = 0.0

        if self._mode == ExerciseMode.AUTO:
            auto_progress = self._update_auto_detector(pose, now)

        if self._active_exercise == ExerciseType.PUSHUP:
            (
                rep_done,
                depth,
                p_ang,
                s_ang,
                drop,
                target_y,
                anchor_x,
                target_hit,
                fb,
            ) = self._pushup.update(pose.active_joints, now, pose.frame_height)
            state = self._pushup.state
            completed_ex = ExerciseType.PUSHUP if rep_done else None
            quality = self._pushup.last_rep_quality
            duration = self._pushup.last_rep_duration
        elif self._active_exercise == ExerciseType.SQUAT:
            (
                rep_done,
                depth,
                p_ang,
                s_ang,
                drop,
                target_y,
                anchor_x,
                target_hit,
                fb,
            ) = self._squat.update(pose.active_joints, now, pose.frame_height)
            state = self._squat.state
            completed_ex = ExerciseType.SQUAT if rep_done else None
            quality = self._squat.last_rep_quality
            duration = self._squat.last_rep_duration
        else:
            rep_done, depth, p_ang, s_ang, drop = False, 0.0, None, None, 0.0
            target_y, anchor_x, target_hit = None, None, False
            state = RepState.IDLE
            completed_ex = None
            quality, duration = None, None
            if self._auto_candidate is not None:
                fb = (f"Hold {self._auto_candidate.value} posture...", "info")
            else:
                fb = ("No valid exercise posture detected", "error")

        total_rejected = self._pushup.rejected_count + self._squat.rejected_count
        msg, lvl = self._resolve_feedback(fb, now)

        return AnalysisSnapshot(
            mode=self._mode,
            active_exercise=self._active_exercise,
            state=state,
            pushup_count=self._pushup.count,
            squat_count=self._squat.count,
            rejected_count=total_rejected,
            last_rep_quality=quality,
            last_rep_duration=duration,
            rep_just_completed=rep_done,
            completed_exercise=completed_ex,
            depth_progress=depth,
            primary_angle=p_ang,
            secondary_angle=s_ang,
            normalized_drop=drop,
            target_depth_y=target_y,
            target_anchor_x=anchor_x,
            target_reached=target_hit,
            feedback_message=msg,
            feedback_level=lvl,
            auto_candidate=self._auto_candidate,
            auto_progress=auto_progress,
            active_side=pose.active_side,
            landmarks_valid=True,
        )