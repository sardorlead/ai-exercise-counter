"""MediaPipe Pose wrapper with per-landmark One-Euro smoothing and dominant-side gating."""

from dataclasses import dataclass
from typing import Dict
import cv2
import mediapipe as mp
import numpy as np

from config import FILTER, MEDIAPIPE
from geometry import LandmarkOneEuroFilter, LandmarkPoint


SIDE_LANDMARK_INDICES: Dict[str, Dict[str, int]] = {
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
    },
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
    },
}


@dataclass
class PoseFrameResult:
    """Structured output for a single processed video frame."""

    detected: bool
    active_side: str
    side_switched: bool
    mean_visibility: float
    landmarks_valid: bool
    active_joints: Dict[str, LandmarkPoint]
    all_landmarks: Dict[int, LandmarkPoint]
    frame_width: int
    frame_height: int


class PoseDetector:
    """Wraps MediaPipe Pose with adaptive One-Euro smoothing across all 33 body landmarks."""

    def __init__(self) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE.model_complexity,
            smooth_landmarks=MEDIAPIPE.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=MEDIAPIPE.min_detection_confidence,
            min_tracking_confidence=MEDIAPIPE.min_tracking_confidence,
        )
        self._active_side: str = "left"
        self._lm_filters: Dict[int, LandmarkOneEuroFilter] = {
            idx: LandmarkOneEuroFilter(
                FILTER.landmark_min_cutoff, FILTER.landmark_beta, FILTER.landmark_d_cutoff
            )
            for idx in range(33)
        }

    def _reset_landmark_filters(self) -> None:
        for flt in self._lm_filters.values():
            flt.reset()

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> PoseFrameResult:
        """Run pose estimation, apply One-Euro smoothing, and validate active landmarks."""
        frame_h, frame_w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.pose_landmarks:
            self._reset_landmark_filters()
            return PoseFrameResult(
                detected=False,
                active_side=self._active_side,
                side_switched=False,
                mean_visibility=0.0,
                landmarks_valid=False,
                active_joints={},
                all_landmarks={},
                frame_width=frame_w,
                frame_height=frame_h,
            )

        raw_lms = results.pose_landmarks.landmark
        all_landmarks: Dict[int, LandmarkPoint] = {}

        for idx, lm in enumerate(raw_lms):
            vis = float(lm.visibility)
            if vis < 0.25:
                # Avoid pulling the filter toward hallucinated off-screen points
                self._lm_filters[idx].reset()
                sx, sy, sz = float(lm.x), float(lm.y), float(lm.z)
            else:
                sx, sy, sz = self._lm_filters[idx].update(
                    float(lm.x), float(lm.y), float(lm.z), timestamp
                )

            all_landmarks[idx] = LandmarkPoint(
                x=sx,
                y=sy,
                z=sz,
                visibility=vis,
                px=sx * frame_w,
                py=sy * frame_h,
                pz=sz * frame_w,
            )

        left_vis = self._compute_side_mean_visibility(all_landmarks, "left")
        right_vis = self._compute_side_mean_visibility(all_landmarks, "right")

        prev_side = self._active_side
        if self._active_side == "left" and right_vis > left_vis + FILTER.side_switch_margin:
            self._active_side = "right"
        elif self._active_side == "right" and left_vis > right_vis + FILTER.side_switch_margin:
            self._active_side = "left"

        side_switched = self._active_side != prev_side
        chosen_map = SIDE_LANDMARK_INDICES[self._active_side]
        active_joints = {name: all_landmarks[idx] for name, idx in chosen_map.items()}
        mean_vis = left_vis if self._active_side == "left" else right_vis

        landmarks_valid = all(
            pt.visibility >= FILTER.min_visibility
            and -0.05 <= pt.x <= 1.05
            and -0.05 <= pt.y <= 1.05
            for pt in active_joints.values()
        )

        return PoseFrameResult(
            detected=True,
            active_side=self._active_side,
            side_switched=side_switched,
            mean_visibility=mean_vis,
            landmarks_valid=landmarks_valid,
            active_joints=active_joints,
            all_landmarks=all_landmarks,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    @staticmethod
    def _compute_side_mean_visibility(
        all_landmarks: Dict[int, LandmarkPoint], side: str
    ) -> float:
        indices = SIDE_LANDMARK_INDICES[side].values()
        scores = [all_landmarks[i].visibility for i in indices]
        return sum(scores) / len(scores)

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        self._pose.close()