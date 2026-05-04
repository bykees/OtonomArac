import cv2
import numpy as np

try:
    from .config import CameraConfig
except ImportError:
    from config import CameraConfig

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class CameraStream:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.picam2 = None

    def open(self) -> None:
        if self.config.use_picamera2:
            self._open_picamera2()
            if self.picam2 is not None:
                return

        self._open_opencv_camera()

    def _open_picamera2(self) -> None:
        if Picamera2 is None:
            return

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": self.config.color_format,
            },
            buffer_count=max(1, self.config.buffer_size),
        )
        self.picam2.configure(config)
        try:
            self.picam2.set_controls({"AwbEnable": True})
        except Exception:
            pass
        self.picam2.start()

    def _open_opencv_camera(self) -> None:
        if self.config.use_directshow:
            self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        if not self.cap.isOpened():
            raise RuntimeError("Kamera acilamadi.")

    def read(self):
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None

            if frame.ndim == 3 and frame.shape[2] == 3 and self.config.color_conversion == "RGB2BGR":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3 and self.config.color_conversion == "BGR2RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 3 and self.config.color_conversion == "NONE":
                frame = frame.copy()
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = self._postprocess(frame)
            return True, frame

        if self.cap is None:
            raise RuntimeError("Kamera once open() ile acilmali.")

        ok, frame = self.cap.read()
        if not ok:
            return False, None

        frame = self._postprocess(frame)
        return True, frame

    def _postprocess(self, frame):
        frame = cv2.resize(frame, (self.config.width, self.config.height))

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)

        return frame

    def release(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
