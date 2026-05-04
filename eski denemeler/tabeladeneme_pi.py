import os
import re

import cv2
import numpy as np


class TabelaAlgilayiciLite:
    def __init__(self, camera_index=0, debug=True):
        self.camera_index = camera_index
        self.DEBUG = debug
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Raspberry Pi 3 icin hafif ayarlar
        self.WIDTH = 320
        self.HEIGHT = 240
        self.template_size = (64, 64)
        self.process_every_n_frames = 2
        self.min_match_score = 0.38

        self.templates = {}
        self.frame_count = 0

        self.last_label = "YOK"
        self.label_count = 0
        self.confirm_frames = 2

        self.last_box = None
        self.last_score = 0.0
        self.last_kind = "YOK"
        self.last_symbol = None
        self.missed_frames = 0
        self.max_missed_frames = 4

    def normalize_label_name(self, file_name):
        label = os.path.splitext(file_name)[0].lower()
        label = re.sub(r"_[0-9]+$", "", label)
        return label

    def normalize_image(self, img):
        if img is None:
            return None

        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255, dtype=np.float32)
            merged = bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])
            return merged.astype(np.uint8)

        return img

    def create_red_mask(self, hsv):
        lower_red1 = np.array([0, 80, 70])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([165, 80, 70])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def create_white_mask(self, hsv):
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([179, 70, 255])
        return cv2.inRange(hsv, lower_white, upper_white)

    def keep_large_components(self, mask, min_area):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        filtered = np.zeros_like(mask)

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == label_idx] = 255

        return filtered

    def normalize_symbol_mask(self, mask):
        if mask is None or cv2.countNonZero(mask) == 0:
            return None

        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = mask[y:y + h, x:x + w]

        target_w, target_h = self.template_size
        scale = min((target_w - 8) / max(w, 1), (target_h - 8) / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        normalized = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        normalized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return normalized

    def build_symbol_descriptor(self, mask):
        if mask is None or cv2.countNonZero(mask) == 0:
            return None

        binary = (mask > 0).astype(np.float32)
        row_profile = np.mean(binary, axis=1)
        col_profile = np.mean(binary, axis=0)

        row_bins = np.array_split(row_profile, 6)
        col_bins = np.array_split(col_profile, 6)

        return {
            "total_ratio": float(np.mean(binary)),
            "row_features": np.array([float(np.mean(x)) for x in row_bins], dtype=np.float32),
            "col_features": np.array([float(np.mean(x)) for x in col_bins], dtype=np.float32),
            "top_half": float(np.mean(binary[: binary.shape[0] // 2, :])),
            "bottom_half": float(np.mean(binary[binary.shape[0] // 2 :, :])),
            "center_box": float(
                np.mean(
                    binary[
                        binary.shape[0] // 4 : 3 * binary.shape[0] // 4,
                        binary.shape[1] // 4 : 3 * binary.shape[1] // 4,
                    ]
                )
            ),
        }

    def compare_descriptors(self, symbol_desc, template_desc):
        if symbol_desc is None or template_desc is None:
            return 0.0

        row_diff = float(
            np.mean(np.abs(symbol_desc["row_features"] - template_desc["row_features"]))
        )
        col_diff = float(
            np.mean(np.abs(symbol_desc["col_features"] - template_desc["col_features"]))
        )
        total_diff = abs(symbol_desc["total_ratio"] - template_desc["total_ratio"])
        balance_diff = abs(
            (symbol_desc["bottom_half"] - symbol_desc["top_half"])
            - (template_desc["bottom_half"] - template_desc["top_half"])
        )
        center_diff = abs(symbol_desc["center_box"] - template_desc["center_box"])

        weighted_diff = (
            0.35 * row_diff
            + 0.30 * col_diff
            + 0.20 * total_diff
            + 0.10 * balance_diff
            + 0.05 * center_diff
        )
        return max(0.0, 1.0 - weighted_diff * 2.8)

    def infer_label_from_rules(self, symbol_desc):
        if symbol_desc is None:
            return "YOK", 0.0

        total_ratio = symbol_desc["total_ratio"]
        bottom_bias = symbol_desc["bottom_half"] - symbol_desc["top_half"]
        center_strength = symbol_desc["center_box"]
        middle_rows = float(np.mean(symbol_desc["row_features"][3:5]))
        top_rows = float(np.mean(symbol_desc["row_features"][:2]))

        if (
            0.04 <= total_ratio <= 0.24
            and bottom_bias > 0.06
            and middle_rows > max(0.01, top_rows * 2.0)
            and center_strength > 0.08
        ):
            return "kasis", 0.55

        return "YOK", 0.0

    def load_templates(self, template_folder="templates"):
        if not os.path.isabs(template_folder):
            template_folder = os.path.join(self.base_dir, template_folder)

        if not os.path.isdir(template_folder):
            raise FileNotFoundError(
                f"Template klasoru bulunamadi: {template_folder}"
            )

        for file_name in os.listdir(template_folder):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            label = self.normalize_label_name(file_name)
            path = os.path.join(template_folder, file_name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            img = self.normalize_image(img)
            symbol = self.extract_warning_symbol(img)
            if symbol is None:
                continue

            if label not in self.templates:
                self.templates[label] = []

            self.templates[label].append(
                {
                    "symbol": symbol,
                    "descriptor": self.build_symbol_descriptor(symbol),
                    "source": file_name,
                }
            )

        summary = {label: len(items) for label, items in self.templates.items()}
        print("Yuklenen tabelalar:", summary)

    def extract_warning_symbol(self, img):
        img = cv2.resize(img, (self.template_size[0] * 2, self.template_size[1] * 2))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        red_mask = self.create_red_mask(hsv)
        white_mask = self.create_white_mask(hsv)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        outer = max(contours, key=cv2.contourArea)
        outer_mask = np.zeros_like(red_mask)
        cv2.drawContours(
            outer_mask, [cv2.convexHull(outer)], -1, 255, thickness=cv2.FILLED
        )

        inner_region = cv2.bitwise_and(white_mask, outer_mask)
        inner_region = cv2.bitwise_and(inner_region, cv2.bitwise_not(red_mask))
        inner_region = cv2.erode(inner_region, kernel, iterations=1)
        inner_region = self.keep_large_components(inner_region, min_area=150)

        if cv2.countNonZero(inner_region) == 0:
            return None

        inner_pixels = blur[inner_region > 0]
        if inner_pixels.size == 0:
            return None

        threshold_val = int(max(0, np.mean(inner_pixels) - 20))
        _, dark_mask = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY_INV)
        symbol = cv2.bitwise_and(dark_mask, inner_region)
        symbol = cv2.morphologyEx(symbol, cv2.MORPH_OPEN, kernel)
        symbol = cv2.morphologyEx(symbol, cv2.MORPH_CLOSE, kernel)
        symbol = self.keep_large_components(symbol, min_area=25)

        return self.normalize_symbol_mask(symbol)

    def is_warning_triangle(self, contour, frame_shape):
        area = cv2.contourArea(contour)
        frame_area = frame_shape[0] * frame_shape[1]
        if area < frame_area * 0.008:
            return False

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0
        aspect_ratio = w / float(h)

        if len(approx) < 3 or len(approx) > 8:
            return False
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        if solidity < 0.75:
            return False
        if y > int(frame_shape[0] * 0.85):
            return False
        return True

    def find_sign_candidates(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_area = frame.shape[0] * frame.shape[1]
        frame_center_x = frame.shape[1] // 2

        red_mask = self.create_red_mask(hsv)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = self.keep_large_components(red_mask, min_area=int(frame_area * 0.0025))

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        debug_mask = np.zeros_like(red_mask)

        for cnt in contours:
            if not self.is_warning_triangle(cnt, frame.shape):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_distance = abs(center_x - frame_center_x) / float(frame.shape[1])
            if center_distance > 0.42:
                continue

            candidates.append((x, y, w, h))
            cv2.drawContours(debug_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return candidates, debug_mask

    def compare_with_templates(self, symbol_img):
        best_label = "YOK"
        best_score = 0.0
        symbol_desc = self.build_symbol_descriptor(symbol_img)

        for label, template_infos in self.templates.items():
            label_best_score = 0.0

            for template_info in template_infos:
                descriptor_score = self.compare_descriptors(
                    symbol_desc, template_info["descriptor"]
                )
                xor_score = 1.0 - (
                    np.sum(cv2.bitwise_xor(symbol_img, template_info["symbol"]) == 255)
                    / symbol_img.size
                )
                score = 0.75 * descriptor_score + 0.25 * xor_score
                if score > label_best_score:
                    label_best_score = score

            if label_best_score > best_score:
                best_score = label_best_score
                best_label = label

        rule_label, rule_score = self.infer_label_from_rules(symbol_desc)
        if rule_score > best_score:
            best_label = rule_label
            best_score = rule_score

        return best_label, best_score

    def stabilize_label(self, label):
        if label == self.last_label:
            self.label_count += 1
        else:
            self.last_label = label
            self.label_count = 1

        if self.label_count >= self.confirm_frames:
            return self.last_label
        return "YOK"

    def detect_sign(self, frame):
        candidates, mask = self.find_sign_candidates(frame)

        best_box = None
        best_symbol = None
        best_label = "YOK"
        best_score = 0.0

        for (x, y, w, h) in candidates:
            roi = frame[y:y + h, x:x + w]
            symbol = self.extract_warning_symbol(roi)
            if symbol is None:
                continue

            label, score = self.compare_with_templates(symbol)
            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)
                best_symbol = symbol
                best_label = label

        if best_box is not None:
            self.last_box = best_box
            self.last_symbol = best_symbol
            self.last_kind = "warning_triangle"
            self.missed_frames = 0
        else:
            self.missed_frames += 1
            if self.missed_frames <= self.max_missed_frames:
                best_box = self.last_box
                best_symbol = self.last_symbol
            else:
                self.last_box = None
                self.last_symbol = None
                self.last_kind = "YOK"

        if best_label != "YOK":
            self.last_score = 0.4 * best_score + 0.6 * self.last_score
        else:
            self.last_score *= 0.7

        final_label = self.stabilize_label(best_label) if best_label != "YOK" else "YOK"
        return final_label, best_box, mask, best_symbol, self.last_score

    def draw_overlay(self, frame, label, box, symbol, score):
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            if symbol is not None and cv2.countNonZero(symbol) > 0:
                preview = cv2.resize(symbol, (72, 72), interpolation=cv2.INTER_NEAREST)
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                px1 = min(self.WIDTH - 82, x + w + 8)
                py1 = max(10, y)
                px2 = px1 + 72
                py2 = py1 + 72
                frame[py1:py2, px1:px2] = preview
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 1)

        cv2.putText(
            frame,
            f"Tabela: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        if not cap.isOpened():
            print("Kamera acilamadi.")
            return

        last_result = ("YOK", None, np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8), None, 0.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            self.frame_count += 1

            if self.frame_count % self.process_every_n_frames == 0:
                last_result = self.detect_sign(frame)

            label, box, mask, symbol, score = last_result
            display = frame.copy()
            self.draw_overlay(display, label, box, symbol, score)

            cv2.imshow("Kamera", display)
            if self.DEBUG:
                cv2.imshow("Tabela Maskesi", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = TabelaAlgilayiciLite(camera_index=0, debug=True)
    sistem.load_templates("templates")
    sistem.run()
