"""Flexible organic spline skeleton renderer, joint angle arcs, depth target line, and HUD."""

from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np

from config import FILTER, UI
from exercise_logic import AnalysisSnapshot, ExerciseType, RepState
from geometry import LandmarkPoint, catmull_rom_spline, midpoint
from pose_detector import PoseFrameResult


class UIRenderer:
    """Renders body-hugging flexible Catmull-Rom skeleton curves, live arcs, and coaching HUD."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        pose: PoseFrameResult,
        snapshot: AnalysisSnapshot,
        fps: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_flexible_skeleton(canvas, pose, snapshot)
        self._draw_joint_angle_arc_and_labels(canvas, pose, snapshot)
        self._draw_target_depth_guide(canvas, snapshot)
        self._draw_top_hud(canvas, snapshot, fps)
        self._draw_depth_bar(canvas, snapshot)
        self._draw_auto_lock_bar(canvas, snapshot)
        self._draw_feedback_banner(canvas, snapshot)
        self._draw_controls_footer(canvas)
        return canvas

    def _draw_filled_panel(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = UI.bg_dark,
        alpha: float = UI.panel_alpha,
    ) -> None:
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(canvas.shape[1], pt2[0]), min(canvas.shape[0], pt2[1])
        if x2 <= x1 or y2 <= y1:
            return
        roi = canvas[y1:y2, x1:x2]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def _state_color(self, state: RepState) -> Tuple[int, int, int]:
        if state == RepState.IDLE:
            return UI.accent_red
        if state == RepState.UP_READY:
            return UI.accent_cyan
        if state in (RepState.GOING_DOWN, RepState.GOING_UP):
            return UI.accent_amber
        return UI.accent_green

    @staticmethod
    def _collect_visible_chain(
        lms: Dict[int, LandmarkPoint], indices: List[int], min_vis: float = 0.35
    ) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for idx in indices:
            if idx in lms and lms[idx].visibility >= min_vis:
                pts.append(lms[idx].xy_float)
        return pts

    def _draw_flexible_skeleton(
        self, canvas: np.ndarray, pose: PoseFrameResult, snapshot: AnalysisSnapshot
    ) -> None:
        """
        Draw smooth, body-hugging Centripetal Catmull-Rom curves instead of stiff sticks:
        - Central flexible spine (nose -> neck -> mid-back -> pelvis)
        - Smooth shoulder and hip anatomical contours
        - Continuous flexible arm & leg splines passing exactly through elbows, knees, hands, feet
        """
        if not pose.detected or not pose.all_landmarks:
            return

        lms = pose.all_landmarks
        active_color = self._state_color(snapshot.state) if pose.landmarks_valid else UI.accent_red
        inactive_color = UI.skeleton_secondary

        # Compute anatomical midpoints for flexible spine curve
        has_shoulders = 11 in lms and 12 in lms and lms[11].visibility >= 0.35 and lms[12].visibility >= 0.35
        has_hips = 23 in lms and 24 in lms and lms[23].visibility >= 0.35 and lms[24].visibility >= 0.35

        neck: Optional[LandmarkPoint] = midpoint(lms[11], lms[12]) if has_shoulders else None
        pelvis: Optional[LandmarkPoint] = midpoint(lms[23], lms[24]) if has_hips else None
        mid_spine: Optional[LandmarkPoint] = (
            midpoint(neck, pelvis) if (neck is not None and pelvis is not None) else None
        )

        # Build spline chains: (points, color, thickness, is_active)
        chains: List[Tuple[List[Tuple[float, float]], Tuple[int, int, int], int, bool]] = []

        # 1. Central flexible spine (head -> neck -> mid-back -> pelvis)
        spine_pts: List[Tuple[float, float]] = []
        if 0 in lms and lms[0].visibility >= 0.40:
            spine_pts.append(lms[0].xy_float)
        if neck is not None:
            spine_pts.append(neck.xy_float)
        if mid_spine is not None:
            spine_pts.append(mid_spine.xy_float)
        if pelvis is not None:
            spine_pts.append(pelvis.xy_float)
        if len(spine_pts) >= 2:
            chains.append((spine_pts, active_color, 3, True))

        # 2. Shoulder & hip subtle contours
        if has_shoulders and neck is not None:
            chains.append(
                ([lms[11].xy_float, neck.xy_float, lms[12].xy_float], inactive_color, 2, False)
            )
        if has_hips and pelvis is not None:
            chains.append(
                ([lms[23].xy_float, pelvis.xy_float, lms[24].xy_float], inactive_color, 2, False)
            )

        # 3. Left & Right Arms (neck -> shoulder -> elbow -> wrist -> hand)
        for side, indices in (("left", [11, 13, 15, 19]), ("right", [12, 14, 16, 20])):
            pts = self._collect_visible_chain(lms, indices)
            if neck is not None and len(pts) >= 2:
                pts = [neck.xy_float] + pts
            is_active = side == pose.active_side
            chains.append((pts, active_color if is_active else inactive_color, 4 if is_active else 2, is_active))

        # 4. Left & Right Legs (pelvis -> hip -> knee -> ankle -> foot)
        for side, indices in (("left", [23, 25, 27, 31]), ("right", [24, 26, 28, 32])):
            pts = self._collect_visible_chain(lms, indices)
            if pelvis is not None and len(pts) >= 2:
                pts = [pelvis.xy_float] + pts
            is_active = side == pose.active_side
            chains.append((pts, active_color if is_active else inactive_color, 4 if is_active else 2, is_active))

        # Render soft outer glow in a single blended pass for fast CPU performance
        glow_layer = canvas.copy()
        computed_curves: List[Tuple[np.ndarray, Tuple[int, int, int], int]] = []

        for pts, color, thickness, is_active in chains:
            if len(pts) < 2:
                continue
            curve = catmull_rom_spline(pts, samples_per_segment=UI.spline_samples_per_segment)
            if curve.size == 0:
                continue
            computed_curves.append((curve, color, thickness))
            glow_w = thickness + (8 if is_active else 4)
            cv2.polylines(glow_layer, [curve], False, color, glow_w, cv2.LINE_AA)

        cv2.addWeighted(glow_layer, UI.glow_alpha, canvas, 1.0 - UI.glow_alpha, 0.0, dst=canvas)

        # Draw crisp organic core splines on top
        for curve, color, thickness in computed_curves:
            cv2.polylines(canvas, [curve], False, color, thickness, cv2.LINE_AA)

        # Draw anatomical joints with soft rings
        for joint_name, pt in pose.active_joints.items():
            vis_ok = pt.visibility >= FILTER.min_visibility
            fill = active_color if vis_ok else UI.accent_red
            radius = 7 if joint_name in ("elbow", "knee", "hip") else 5
            cv2.circle(canvas, pt.xy_px, radius + 3, UI.bg_dark, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt.xy_px, radius, fill, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt.xy_px, radius + 2, UI.text_primary, 1, cv2.LINE_AA)

    def _draw_angle_arc(
        self,
        canvas: np.ndarray,
        a: LandmarkPoint,
        b: LandmarkPoint,
        c: LandmarkPoint,
        color: Tuple[int, int, int],
        radius: int = 34,
    ) -> None:
        """Draw a translucent bending arc inside joint b showing live flexion angle."""
        ang_a = math.degrees(math.atan2(a.py - b.py, a.px - b.px)) % 360.0
        ang_c = math.degrees(math.atan2(c.py - b.py, c.px - b.px)) % 360.0
        diff = (ang_c - ang_a) % 360.0
        if diff <= 180.0:
            start_ang, end_ang = ang_a, ang_a + diff
        else:
            start_ang, end_ang = ang_c, ang_c + (360.0 - diff)

        overlay = canvas.copy()
        cv2.ellipse(overlay, b.xy_px, (radius, radius), 0.0, start_ang, end_ang, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0, dst=canvas)
        cv2.ellipse(canvas, b.xy_px, (radius, radius), 0.0, start_ang, end_ang, color, 2, cv2.LINE_AA)

    def _draw_joint_angle_arc_and_labels(
        self, canvas: np.ndarray, pose: PoseFrameResult, snapshot: AnalysisSnapshot
    ) -> None:
        if not pose.detected or not pose.landmarks_valid or snapshot.primary_angle is None:
            return

        joints = pose.active_joints
        arc_color = UI.accent_green if snapshot.target_reached else self._state_color(snapshot.state)

        if snapshot.active_exercise == ExerciseType.PUSHUP:
            self._draw_angle_arc(canvas, joints["shoulder"], joints["elbow"], joints["wrist"], arc_color)
            self._draw_angle_tag(canvas, joints["elbow"].xy_px, f"{int(snapshot.primary_angle)} deg")
            if snapshot.secondary_angle is not None:
                self._draw_angle_tag(
                    canvas, joints["hip"].xy_px, f"Back {int(snapshot.secondary_angle)} deg"
                )
        elif snapshot.active_exercise == ExerciseType.SQUAT:
            self._draw_angle_arc(canvas, joints["hip"], joints["knee"], joints["ankle"], arc_color)
            self._draw_angle_tag(canvas, joints["knee"].xy_px, f"{int(snapshot.primary_angle)} deg")
            if snapshot.secondary_angle is not None:
                self._draw_angle_tag(
                    canvas, joints["hip"].xy_px, f"Lean {int(snapshot.secondary_angle)} deg"
                )

    def _draw_target_depth_guide(
        self, canvas: np.ndarray, snapshot: AnalysisSnapshot
    ) -> None:
        """Draw a dashed horizontal line showing exact depth required for a valid rep."""
        if snapshot.target_depth_y is None or snapshot.target_anchor_x is None:
            return
        if snapshot.state == RepState.IDLE:
            return

        h, w = canvas.shape[:2]
        ty = int(round(snapshot.target_depth_y))
        if ty < 40 or ty > h - 40:
            return

        cx = int(round(snapshot.target_anchor_x))
        x1 = max(20, cx - 85)
        x2 = min(w - 90, cx + 85)
        color = UI.accent_green if snapshot.target_reached else UI.accent_amber

        # Dashed line
        dash_len = 10
        for x in range(x1, x2, dash_len * 2):
            cv2.line(canvas, (x, ty), (min(x + dash_len, x2), ty), color, 2, cv2.LINE_AA)

        label = "DEPTH OK" if snapshot.target_reached else "TARGET DEPTH"
        cv2.putText(canvas, label, (x2 + 6, ty + 4), self.FONT, 0.42, color, 1, cv2.LINE_AA)

    def _draw_angle_tag(
        self, canvas: np.ndarray, anchor: Tuple[int, int], text: str
    ) -> None:
        x, y = anchor[0] + 14, anchor[1] - 12
        (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.54, 2)
        self._draw_filled_panel(canvas, (x - 5, y - th - 5), (x + tw + 6, y + 5), alpha=0.78)
        cv2.putText(canvas, text, (x, y), self.FONT, 0.54, UI.text_primary, 2, cv2.LINE_AA)

    def _draw_top_hud(
        self, canvas: np.ndarray, snapshot: AnalysisSnapshot, fps: float
    ) -> None:
        _, w = canvas.shape[:2]

        # Left card: Rep counters & Quality stats
        self._draw_filled_panel(canvas, (16, 16), (360, 164))
        cv2.putText(
            canvas, "AI EXERCISE COUNTER PRO", (30, 40), self.FONT, 0.55, UI.accent_cyan, 2, cv2.LINE_AA
        )

        pu_color = (
            UI.accent_green
            if snapshot.active_exercise == ExerciseType.PUSHUP
            else UI.text_primary
        )
        sq_color = (
            UI.accent_green
            if snapshot.active_exercise == ExerciseType.SQUAT
            else UI.text_primary
        )
        cv2.putText(
            canvas,
            f"PUSH-UPS: {snapshot.pushup_count}",
            (30, 78),
            self.FONT,
            0.82,
            pu_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"SQUATS:   {snapshot.squat_count}",
            (30, 114),
            self.FONT,
            0.82,
            sq_color,
            2,
            cv2.LINE_AA,
        )

        qual_str = f"{snapshot.last_rep_quality}%" if snapshot.last_rep_quality is not None else "--"
        tempo_str = f"{snapshot.last_rep_duration:.1f}s" if snapshot.last_rep_duration is not None else "--"
        cv2.putText(
            canvas,
            f"QUALITY: {qual_str}  |  TEMPO: {tempo_str}  |  BAD: {snapshot.rejected_count}",
            (30, 148),
            self.FONT,
            0.46,
            UI.text_muted,
            1,
            cv2.LINE_AA,
        )

        # Right card: Mode, State, Tracking, FPS
        rx1 = w - 345
        self._draw_filled_panel(canvas, (rx1, 16), (w - 16, 164))
        cv2.putText(
            canvas,
            f"MODE: {snapshot.mode.value}   |   FPS: {fps:4.1f}",
            (rx1 + 14, 44),
            self.FONT,
            0.55,
            UI.text_muted,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"ACTIVE: {snapshot.active_exercise.value}",
            (rx1 + 14, 80),
            self.FONT,
            0.65,
            UI.text_primary,
            2,
            cv2.LINE_AA,
        )
        state_col = self._state_color(snapshot.state)
        cv2.putText(
            canvas,
            f"STATE: {snapshot.state.value}",
            (rx1 + 14, 114),
            self.FONT,
            0.60,
            state_col,
            2,
            cv2.LINE_AA,
        )
        vis_text = "LOCKED" if snapshot.landmarks_valid else "LOW VIS"
        cv2.putText(
            canvas,
            f"SIDE: {snapshot.active_side.upper()}  |  POSE: {vis_text}",
            (rx1 + 14, 148),
            self.FONT,
            0.48,
            UI.text_muted,
            1,
            cv2.LINE_AA,
        )

    def _draw_depth_bar(self, canvas: np.ndarray, snapshot: AnalysisSnapshot) -> None:
        h, w = canvas.shape[:2]
        bar_w = 28
        bar_x2 = w - 28
        bar_x1 = bar_x2 - bar_w
        bar_y1 = 194
        bar_y2 = h - 130
        if bar_y2 <= bar_y1 + 60:
            return

        self._draw_filled_panel(canvas, (bar_x1 - 12, bar_y1 - 36), (bar_x2 + 12, bar_y2 + 34))
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), UI.text_muted, 1)

        pct = max(0.0, min(100.0, snapshot.depth_progress))
        fill_h = int((bar_y2 - bar_y1) * (pct / 100.0))
        fill_y1 = bar_y2 - fill_h

        if pct >= 99.0:
            bar_color = UI.accent_green
        elif pct >= 45.0:
            bar_color = UI.accent_amber
        else:
            bar_color = UI.accent_cyan

        if fill_h > 0:
            cv2.rectangle(canvas, (bar_x1 + 2, fill_y1), (bar_x2 - 2, bar_y2 - 2), bar_color, -1)

        cv2.putText(
            canvas, "DEPTH", (bar_x1 - 10, bar_y1 - 12), self.FONT, 0.45, UI.text_muted, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"{int(round(pct))}%",
            (bar_x1 - 6, bar_y2 + 22),
            self.FONT,
            0.55,
            bar_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_auto_lock_bar(self, canvas: np.ndarray, snapshot: AnalysisSnapshot) -> None:
        if snapshot.auto_candidate is None or snapshot.auto_progress <= 0.0:
            return

        w = canvas.shape[1]
        bar_w = 300
        x1 = (w - bar_w) // 2
        y1 = 22
        self._draw_filled_panel(canvas, (x1 - 16, y1 - 6), (x1 + bar_w + 16, y1 + 52))
        cv2.putText(
            canvas,
            f"Locking {snapshot.auto_candidate.value}...",
            (x1, y1 + 16),
            self.FONT,
            0.58,
            UI.accent_cyan,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (x1, y1 + 26), (x1 + bar_w, y1 + 40), UI.text_muted, 1)
        fill_w = int(bar_w * snapshot.auto_progress)
        cv2.rectangle(canvas, (x1, y1 + 26), (x1 + fill_w, y1 + 40), UI.accent_green, -1)

    def _draw_feedback_banner(self, canvas: np.ndarray, snapshot: AnalysisSnapshot) -> None:
        if not snapshot.feedback_message:
            return

        h, w = canvas.shape[:2]
        level_colors = {
            "info": UI.accent_cyan,
            "warning": UI.accent_amber,
            "error": UI.accent_red,
            "success": UI.accent_green,
        }
        color = level_colors.get(snapshot.feedback_level, UI.text_primary)

        text = snapshot.feedback_message
        (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.78, 2)
        cx = w // 2
        y_base = h - 54
        self._draw_filled_panel(
            canvas,
            (cx - tw // 2 - 22, y_base - th - 14),
            (cx + tw // 2 + 22, y_base + 14),
            alpha=0.84,
        )
        cv2.rectangle(
            canvas,
            (cx - tw // 2 - 22, y_base - th - 14),
            (cx + tw // 2 + 22, y_base + 14),
            color,
            2,
        )
        cv2.putText(
            canvas, text, (cx - tw // 2, y_base), self.FONT, 0.78, color, 2, cv2.LINE_AA
        )

    def _draw_controls_footer(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        controls = "[1] Push-up   [2] Squat   [3] Auto   [R] Reset   [M] Mute   [Q] Quit"
        self._draw_filled_panel(canvas, (0, h - 28), (w, h), alpha=0.80)
        cv2.putText(
            canvas, controls, (18, h - 9), self.FONT, 0.48, UI.text_muted, 1, cv2.LINE_AA
        )