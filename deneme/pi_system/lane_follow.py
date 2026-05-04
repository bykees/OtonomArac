import cv2
import numpy as np

try:
    from .config import DetectionConfig
    from .types import LaneResult
except ImportError:
    from config import DetectionConfig
    from types import LaneResult


class LaneFollower:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.last_lane_center = None
        self.lane_memory = {
            "left": {"x": None, "confidence": 0},
            "right": {"x": None, "confidence": 0},
        }
        self.last_lane_width = config.lane_default_width

    def process(self, frame) -> LaneResult:
        h, w = frame.shape[:2]
        y0 = int(h * self.config.lane_roi_start_ratio)
        roi = frame[y0:, :]

        mask, value_channel = self._build_lane_mask(roi)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = self._apply_drive_roi(mask)
        mask = self._filter_lane_components(mask, value_channel)
        frame_center = w // 2
        histogram_center = self._histogram_lane_center(mask)
        left_x, right_x = self._find_lane_points(mask)
        lane_center = self._estimate_lane_center(left_x, right_x, w, histogram_center)

        if lane_center is None:
            lane_center = self.last_lane_center if self.last_lane_center is not None else frame_center
            found = False
        else:
            found = True
            if (
                self.last_lane_center is not None
                and abs(lane_center - self.last_lane_center) > self.config.lane_max_center_jump
            ):
                lane_center = int(0.75 * self.last_lane_center + 0.25 * lane_center)
            lane_center = int(
                self.config.lane_smooth_alpha * lane_center
                + (1.0 - self.config.lane_smooth_alpha)
                * (self.last_lane_center if self.last_lane_center is not None else lane_center)
            )
            self.last_lane_center = lane_center

        error = lane_center - frame_center
        return LaneResult(
            found=found,
            lane_center_x=lane_center,
            frame_center_x=frame_center,
            steering_error=error,
            direction_hint=self._direction_from_error(error),
            debug_mask=mask,
        )

    def _build_lane_mask(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        top_hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        local_light = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, top_hat_kernel)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        h, w = gray.shape
        adaptive = np.zeros_like(gray)

        for y in range(0, h, 4):
            row = enhanced[y:y + 4]
            if row.size == 0:
                continue

            mean_val = np.mean(row)
            std_val = np.std(row)
            threshold = mean_val + self.config.lane_adaptive_std_coef * std_val
            adaptive[y:y + 4] = np.where(row > threshold, 255, 0)

        _, bright_mask = cv2.threshold(
            v_channel, self.config.lane_white_min_value, 255, cv2.THRESH_BINARY
        )
        _, saturation_mask = cv2.threshold(
            s_channel, self.config.lane_white_max_saturation, 255, cv2.THRESH_BINARY
        )
        saturation_mask = cv2.bitwise_not(saturation_mask)
        _, contrast_mask = cv2.threshold(local_light, 18, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(adaptive, contrast_mask)
        mask = cv2.bitwise_and(mask, bright_mask)
        mask = cv2.bitwise_and(mask, saturation_mask)

        # Alt tarafi daha onemli kabul et.
        row_weights = np.linspace(0.6, 1.0, h).reshape(h, 1)
        weighted = (mask.astype(np.float32) * row_weights).astype(np.uint8)
        return weighted, v_channel

    def _apply_drive_roi(self, mask):
        h, w = mask.shape
        top_half_width = int(w * self.config.lane_roi_top_width_ratio * 0.5)
        bottom_half_width = int(w * self.config.lane_roi_bottom_width_ratio * 0.5)
        center = w // 2
        polygon = np.array(
            [
                [center - bottom_half_width, h - 1],
                [center + bottom_half_width, h - 1],
                [center + top_half_width, 0],
                [center - top_half_width, 0],
            ],
            dtype=np.int32,
        )
        roi_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(roi_mask, polygon, 255)
        return cv2.bitwise_and(mask, roi_mask)

    def _filter_lane_components(self, mask, value_channel):
        h, w = mask.shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        filtered = np.zeros_like(mask)
        max_area = int(h * w * self.config.lane_max_area_ratio)
        dark_mask = cv2.threshold(
            value_channel, self.config.lane_black_max_value, 255, cv2.THRESH_BINARY_INV
        )[1]

        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            bw = stats[idx, cv2.CC_STAT_WIDTH]
            bh = stats[idx, cv2.CC_STAT_HEIGHT]
            density = area / float(max(1, bw * bh))
            aspect = max(bw, bh) / float(max(1, min(bw, bh)))

            if area < self.config.lane_min_area:
                continue
            if area > max_area:
                continue
            if bh < h * self.config.lane_min_height_ratio:
                continue
            if bw > w * self.config.lane_max_width_ratio:
                continue
            if y + bh < h * 0.35:
                continue
            if density > 0.82:
                continue
            if aspect < self.config.lane_min_aspect_ratio:
                continue
            if not self._has_dark_context(dark_mask, x, y, bw, bh):
                continue

            filtered[labels == idx] = 255

        return filtered

    def _has_dark_context(self, dark_mask, x, y, w, h):
        img_h, img_w = dark_mask.shape
        pad = max(8, int(max(w, h) * 0.35))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        context = dark_mask[y1:y2, x1:x2]
        if context.size == 0:
            return False

        dark_ratio = cv2.countNonZero(context) / float(context.size)
        return dark_ratio >= self.config.lane_min_dark_context_ratio

    def _histogram_lane_center(self, mask):
        h, w = mask.shape
        rows = min(h, self.config.lane_histogram_rows)
        bottom_band = mask[h - rows :, :]
        histogram = np.sum(bottom_band > 0, axis=0).astype(np.float32)
        if np.max(histogram) <= 0:
            return None

        left_half = histogram[: w // 2]
        right_half = histogram[w // 2 :]

        left_idx = int(np.argmax(left_half)) if np.max(left_half) > 0 else None
        right_idx = int(np.argmax(right_half)) + (w // 2) if np.max(right_half) > 0 else None

        if left_idx is not None and right_idx is not None and right_idx > left_idx:
            return (left_idx + right_idx) // 2
        if left_idx is not None:
            return left_idx + self.last_lane_width // 2
        if right_idx is not None:
            return right_idx - self.last_lane_width // 2
        return None

    def _find_lane_points(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        center = w // 2

        left_x = None
        right_x = None
        left_area = 0
        right_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.lane_min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if y + bh < int(h * 0.25):
                continue

            cx = x + bw // 2
            if cx < center:
                if area > left_area:
                    left_area = area
                    left_x = cx
            else:
                if area > right_area:
                    right_area = area
                    right_x = cx

        self._update_lane_memory("left", left_x)
        self._update_lane_memory("right", right_x)
        return self.lane_memory["left"]["x"], self.lane_memory["right"]["x"]

    def _update_lane_memory(self, side, value):
        memory = self.lane_memory[side]
        if value is not None:
            if memory["x"] is None:
                memory["x"] = value
            else:
                memory["x"] = int(
                    self.config.lane_smooth_alpha * value
                    + (1.0 - self.config.lane_smooth_alpha) * memory["x"]
                )
            memory["confidence"] = min(5, memory["confidence"] + 1)
            return

        memory["confidence"] = max(0, memory["confidence"] - 1)
        if memory["confidence"] == 0:
            memory["x"] = None

    def _estimate_lane_center(self, left_x, right_x, frame_width, histogram_center):
        left = left_x
        right = right_x

        if left is not None and right is not None:
            lane_width = right - left
            if 40 < lane_width < frame_width:
                self.last_lane_width = int(0.8 * self.last_lane_width + 0.2 * lane_width)

        if left is None and right is not None:
            left = right - self.last_lane_width
        elif right is None and left is not None:
            right = left + self.last_lane_width

        if left is None and right is None:
            return histogram_center

        contour_center = (left + right) // 2
        if histogram_center is None:
            return contour_center

        return int(0.65 * contour_center + 0.35 * histogram_center)

    def _direction_from_error(self, error: int) -> str:
        if error < -self.config.lane_turn_threshold:
            return "SOL"
        if error > self.config.lane_turn_threshold:
            return "SAG"
        return "DUZ"
