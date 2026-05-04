import cv2
import numpy as np

from .config import DetectionConfig
from .types import TrafficLightResult


class TrafficLightDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config

    def _create_red_mask(self, hsv):
        lower_red1 = np.array([0, self.config.traffic_red_min_saturation, self.config.traffic_red_min_value])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([170, self.config.traffic_red_min_saturation, self.config.traffic_red_min_value])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def process(self, frame) -> TrafficLightResult:
        h, w = frame.shape[:2]
        roi = frame[: int(h * self.config.red_light_roi_end_ratio), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._create_red_mask(hsv)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_red_light_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(max(1, bh))
            if not 0.5 <= aspect <= 1.5:
                continue

            if y + bh > roi.shape[0] * 0.75:
                continue

            center_x = x + bw // 2
            center_offset_ratio = abs(center_x - (w // 2)) / float(w)
            if center_offset_ratio > self.config.max_red_light_center_offset_ratio:
                continue

            rect_area = bw * bh
            if rect_area <= 0:
                continue

            fill_ratio = area / float(rect_area)
            if fill_ratio < self.config.min_red_light_fill_ratio:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4.0 * np.pi * area / float(max(perimeter * perimeter, 1.0))
            if circularity < 0.35:
                continue

            if area > best_area:
                best_area = area
                best = (x, y, bw, bh)

        if best is None:
            return TrafficLightResult(state="YOK", confidence=0.0, bbox=None)

        confidence = min(1.0, best_area / 1000.0)
        return TrafficLightResult(state="KIRMIZI", confidence=confidence, bbox=best)
