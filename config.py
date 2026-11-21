"""Centralized configuration and strict anti-cheat thresholds for the AI Exercise Counter."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    mirror_frame: bool = True


@dataclass(frozen=True)
class MediaPipeConfig:
    model_complexity: int = 1  # 0=lite, 1=balanced CPU accuracy/FPS, 2=heavy
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    smooth_landmarks: bool = True


@dataclass(frozen=True)
class FilterConfig:
    min_visibility: float = 0.65
    # One-Euro Filter parameters for normalized landmark coordinates (smooth when still, fast on motion)
    landmark_min_cutoff: float = 1.6
    landmark_beta: float = 2.4
    landmark_d_cutoff: float = 1.0
    # One-Euro Filter parameters for joint angles (degrees)
    angle_min_cutoff: float = 1.3
    angle_beta: float = 0.015
    angle_d_cutoff: float = 1.0
    # One-Euro Filter parameters for vertical positions (pixels)
    pos_min_cutoff: float = 1.4
    pos_beta: float = 0.005
    pos_d_cutoff: float = 1.0
    # Torso length smoothing
    torso_ema_alpha: float = 0.20
    # Hysteresis margin to avoid rapid left/right switching
    side_switch_margin: float = 0.08
    # Reject distant/invalid detections where torso < 8% of frame height
    min_torso_height_ratio: float = 0.08
    # Sliding window duration (seconds) for wrist/ankle stability checks
    anchor_window_sec: float = 0.45


@dataclass(frozen=True)
class PushupConfig:
    # Posture gate
    max_torso_horizontal_angle: float = 38.0
    min_body_alignment_angle: float = 148.0
    max_body_alignment_angle: float = 180.0
    min_wrist_below_shoulder_ratio: float = 0.05
    max_wrist_drift_ratio: float = 0.20

    # Strict dual-condition rep thresholds (blocks half-reps)
    up_elbow_angle: float = 158.0
    down_elbow_angle: float = 78.0
    hysteresis_deg: float = 10.0
    min_shoulder_drop_ratio: float = 0.30
    top_return_ratio: float = 0.10

    # Anti-speed-cheat & dwell timers
    min_bottom_dwell_sec: float = 0.15
    min_top_dwell_sec: float = 0.08
    min_rep_duration_sec: float = 0.85
    max_bottom_hold_sec: float = 4.0
    max_rep_duration_sec: float = 8.0
    reversal_threshold_deg: float = 12.0
    bad_alignment_abort_sec: float = 0.65


@dataclass(frozen=True)
class SquatConfig:
    # Posture gate
    max_torso_vertical_angle: float = 50.0
    min_vertical_stack_ratio: float = 0.12
    max_ankle_drift_ratio: float = 0.18

    # Strict dual-condition rep thresholds (blocks half-squats & bouncing)
    up_knee_angle: float = 162.0
    down_knee_angle: float = 85.0
    hysteresis_deg: float = 10.0
    min_hip_drop_ratio: float = 0.34
    top_return_ratio: float = 0.10

    # Anti-speed-cheat, anti-chair & dwell timers
    min_bottom_dwell_sec: float = 0.15
    min_top_dwell_sec: float = 0.08
    min_rep_duration_sec: float = 0.90
    max_bottom_hold_sec: float = 3.5
    max_rep_duration_sec: float = 7.0
    reversal_threshold_deg: float = 12.0


@dataclass(frozen=True)
class ControllerConfig:
    auto_detect_debounce_sec: float = 0.85
    occlusion_grace_sec: float = 0.25
    feedback_display_sec: float = 1.80
    rep_complete_display_sec: float = 0.35


@dataclass(frozen=True)
class UIConfig:
    # Colors in BGR format
    bg_dark: Tuple[int, int, int] = (20, 22, 26)
    text_primary: Tuple[int, int, int] = (245, 245, 248)
    text_muted: Tuple[int, int, int] = (165, 172, 182)
    accent_cyan: Tuple[int, int, int] = (240, 210, 60)
    accent_green: Tuple[int, int, int] = (85, 225, 110)
    accent_amber: Tuple[int, int, int] = (40, 185, 250)
    accent_red: Tuple[int, int, int] = (75, 75, 245)
    skeleton_secondary: Tuple[int, int, int] = (130, 145, 160)
    panel_alpha: float = 0.72
    glow_alpha: float = 0.32
    spline_samples_per_segment: int = 8
    sound_enabled: bool = True


CAMERA = CameraConfig()
MEDIAPIPE = MediaPipeConfig()
FILTER = FilterConfig()
PUSHUP = PushupConfig()
SQUAT = SquatConfig()
CONTROLLER = ControllerConfig()
UI = UIConfig()