"""Main entry point: webcam loop, keyboard controls, and non-blocking rep audio feedback."""

import argparse
import platform
import shutil
import subprocess
import sys
import threading
import time
import cv2

from config import CAMERA, UI
from exercise_logic import ExerciseController, ExerciseMode
from pose_detector import PoseDetector
from ui_renderer import UIRenderer


class SoundFeedback:
    """Cross-platform, non-blocking audio chime triggered upon valid rep completion."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._system = platform.system()

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def play_rep_chime(self) -> None:
        if not self.enabled:
            return
        thread = threading.Thread(target=self._beep_worker, daemon=True)
        thread.start()

    def _beep_worker(self) -> None:
        try:
            if self._system == "Windows":
                import winsound

                winsound.Beep(920, 110)
            elif self._system == "Darwin" and shutil.which("afplay"):
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Tink.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            elif self._system == "Linux" and shutil.which("paplay"):
                subprocess.run(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time AI Exercise Counter Pro")
    parser.add_argument("--camera", type=int, default=CAMERA.camera_index, help="Webcam device index")
    parser.add_argument(
        "--mode",
        choices=["pushup", "squat", "auto"],
        default="auto",
        help="Initial exercise detection mode",
    )
    parser.add_argument("--no-sound", action="store_true", help="Disable rep completion beep")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal frame mirroring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode_map = {
        "pushup": ExerciseMode.PUSHUP,
        "squat": ExerciseMode.SQUAT,
        "auto": ExerciseMode.AUTO,
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam index {args.camera}.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.frame_height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA.target_fps)

    detector = PoseDetector()
    controller = ExerciseController(initial_mode=mode_map[args.mode])
    renderer = UIRenderer()
    sound = SoundFeedback(enabled=(UI.sound_enabled and not args.no_sound))

    mirror = CAMERA.mirror_frame and not args.no_mirror
    window_name = "AI Exercise Counter Pro"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps_ema = float(CAMERA.target_fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Dropped frame from webcam; exiting loop.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            prev_time = now
            fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt)

            pose_result = detector.process_frame(frame, now)
            snapshot = controller.update(pose_result, now)

            if snapshot.rep_just_completed:
                sound.play_rep_chime()

            output_frame = renderer.render(frame, pose_result, snapshot, fps_ema)
            cv2.imshow(window_name, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                controller.set_mode(ExerciseMode.PUSHUP, now)
            elif key == ord("2"):
                controller.set_mode(ExerciseMode.SQUAT, now)
            elif key == ord("3"):
                controller.set_mode(ExerciseMode.AUTO, now)
            elif key == ord("r"):
                controller.reset_counters(now)
            elif key == ord("m"):
                sound.toggle()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()bu 