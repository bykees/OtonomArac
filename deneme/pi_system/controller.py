import os

import cv2

from .camera import CameraStream
from .config import AppConfig
from .lane_follow import LaneFollower
from .motor_driver import MotorDriver
from .parking import ParkingController
from .sign_detection import SignDetector
from .state_machine import VehicleStateMachine
from .traffic_light import TrafficLightDetector
from .types import PerceptionSnapshot


class VehicleController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.camera = CameraStream(config.camera)
        self.motor = MotorDriver()
        self.lane = LaneFollower(config.detection)
        self.traffic_light = TrafficLightDetector(config.detection)
        template_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        template_dir = self._resolve_template_dir(template_root)
        self.sign = SignDetector(config.detection, os.path.abspath(template_dir))
        self.parking = ParkingController(config.vehicle)
        self.state_machine = VehicleStateMachine()
        self.frame_index = 0
        self.last_snapshot = PerceptionSnapshot()

    def _resolve_template_dir(self, template_root):
        for folder_name in ("templates", "Template"):
            candidate = os.path.join(template_root, folder_name)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(template_root, "templates")

    def setup(self) -> None:
        self.sign.load_templates()
        self.motor.setup()
        self.camera.open()

    def loop(self) -> None:
        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    self.motor.stop()
                    break

                self.frame_index += 1
                snapshot = self._run_pipeline(frame)
                command = self.state_machine.update(snapshot)
                self._apply_motor_command(snapshot, command)
                display = frame.copy()
                self._draw_overlay(display, snapshot, command)

                cv2.imshow("Pi System", display)
                if self.config.debug.show_mask and "sign_mask" in snapshot.debug_images:
                    cv2.imshow("Sign Mask", snapshot.debug_images["sign_mask"])
                if self.config.debug.show_mask and snapshot.lane and snapshot.lane.debug_mask is not None:
                    cv2.imshow("Lane Mask", snapshot.lane.debug_mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
        except KeyboardInterrupt:
            self.motor.stop()
        except Exception:
            self.motor.stop()
            raise
        finally:
            self.motor.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()

    def _run_pipeline(self, frame) -> PerceptionSnapshot:
        snapshot = PerceptionSnapshot(frame_index=self.frame_index)

        if self.frame_index % self.config.scheduler.lane_every_n_frames == 0:
            snapshot.lane = self.lane.process(frame)
        else:
            snapshot.lane = self.last_snapshot.lane

        if self.frame_index % self.config.scheduler.light_every_n_frames == 0:
            snapshot.traffic_light = self.traffic_light.process(frame)
        else:
            snapshot.traffic_light = self.last_snapshot.traffic_light

        if self.frame_index % self.config.scheduler.sign_every_n_frames == 0:
            _, sign_debug = self.sign.detect(frame)
            snapshot.sign = self.sign.last_result
            if sign_debug is not None:
                snapshot.debug_images["sign_mask"] = sign_debug
        else:
            snapshot.sign = self.last_snapshot.sign

        if self.frame_index % self.config.scheduler.parking_every_n_frames == 0:
            snapshot.parking = self.parking.process(snapshot.sign)
        else:
            snapshot.parking = self.last_snapshot.parking

        if snapshot.sign and snapshot.sign.bbox is not None:
            sign_bbox = snapshot.sign.bbox
            if snapshot.sign.symbol_mask is not None:
                snapshot.debug_images["sign_symbol"] = snapshot.sign.symbol_mask
            if "sign_mask" not in snapshot.debug_images and self.sign.last_debug_frame is not None:
                snapshot.debug_images["sign_mask"] = self.sign.last_debug_frame

        self.last_snapshot = snapshot
        return snapshot

    def _apply_motor_command(self, snapshot: PerceptionSnapshot, command) -> None:
        sign_label = snapshot.sign.label if snapshot.sign else "YOK"
        sign_conf = snapshot.sign.confidence if snapshot.sign else 0.0
        lane_error = snapshot.lane.steering_error if snapshot.lane and snapshot.lane.found else None

        if command.mode == "STOP_AT_RED" or (sign_label == "STOP" and sign_conf >= 0.60):
            self.motor.stop()
            self._print_motor_debug(0, 0, sign_label, lane_error)
            return

        if sign_label == "SOL" and sign_conf >= 0.65:
            self.motor.set_motor(18, 42)
            self._print_motor_debug(18, 42, sign_label, lane_error)
            return

        if sign_label == "SAG" and sign_conf >= 0.65:
            self.motor.set_motor(42, 18)
            self._print_motor_debug(42, 18, sign_label, lane_error)
            return

        if sign_label == "ILERI" and sign_conf >= 0.65:
            self.motor.forward(35)
            self._print_motor_debug(35, 35, sign_label, lane_error)
            return

        if lane_error is None:
            self.motor.stop()
            self._print_motor_debug(0, 0, sign_label, lane_error)
            return

        base_speed = 35
        turn_gain = 0.4
        left_speed = base_speed - lane_error * turn_gain
        right_speed = base_speed + lane_error * turn_gain
        self.motor.set_motor(left_speed, right_speed)
        self._print_motor_debug(left_speed, right_speed, sign_label, lane_error)

    def _print_motor_debug(self, left_speed, right_speed, sign_label, lane_error) -> None:
        if not self.config.debug.enabled or self.frame_index % 10 != 0:
            return

        print(
            f"motor left={left_speed:.1f} right={right_speed:.1f} "
            f"sign={sign_label} lane_error={lane_error}"
        )

    def _draw_overlay(self, frame, snapshot: PerceptionSnapshot, command) -> None:
        y = 24
        lines = [
            f"Mode: {command.mode}",
            f"Speed: {command.speed:.2f}",
            f"Steer: {command.steering:.2f}",
            f"Reason: {command.reason}",
        ]

        if snapshot.sign:
            lines.append(f"Sign: {snapshot.sign.label} ({snapshot.sign.confidence:.2f})")
        if snapshot.traffic_light:
            lines.append(
                f"Light: {snapshot.traffic_light.state} ({snapshot.traffic_light.confidence:.2f})"
            )
        if snapshot.lane:
            lines.append(f"Lane: {snapshot.lane.direction_hint} err={snapshot.lane.steering_error}")

        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            y += 18

        if snapshot.sign and snapshot.sign.bbox is not None:
            x, sy, w, h = snapshot.sign.bbox
            cv2.rectangle(frame, (x, sy), (x + w, sy + h), (0, 255, 255), 2)

            if self.config.debug.overlay_symbol_preview and snapshot.sign.symbol_mask is not None:
                preview = cv2.resize(snapshot.sign.symbol_mask, (72, 72), interpolation=cv2.INTER_NEAREST)
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                px1 = min(frame.shape[1] - 82, x + w + 8)
                py1 = min(max(10, sy), frame.shape[0] - 82)
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = px1 + 72
                py2 = py1 + 72

                if px2 <= frame.shape[1] and py2 <= frame.shape[0]:
                    frame[py1:py2, px1:px2] = preview
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 1)

            debug_text = (
                f"{snapshot.sign.label} score={snapshot.sign.match_score:.2f} "
                f"area={snapshot.sign.area}"
            )
            cv2.putText(
                frame,
                debug_text,
                (x, max(18, sy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        if snapshot.traffic_light and snapshot.traffic_light.bbox is not None:
            x, y0, w, h = snapshot.traffic_light.bbox
            cv2.rectangle(frame, (x, y0), (x + w, y0 + h), (0, 0, 255), 2)
