import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SignResult:
    label: str = "YOK"
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    kind: str = "NONE"
    symbol_mask: Optional[np.ndarray] = None
    area: int = 0
    match_score: float = 0.0
    color_hint: str = "YOK"
    shape_hint: str = "YOK"
    geometry_hint: str = "BELIRSIZ"
    source: str = "none"
    reject_reason: str = ""


class SignDetector:
    """
    Trainer.py'deki ROI maske mantigini temel alan, false positive'i dusuk tutan
    tabela algilayici. Dis API korunur:
        SignDetector(config, template_dir)
        load_templates()
        detect(frame)
        process(frame)
        last_result / last_debug_frame
    """

    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
    TEMPLATE_SIZE = (96, 96)
    MAX_FRAME_WIDTH = 640
    MIN_CANDIDATE_AREA = 170

    BLACK_WHITE_TEST_MODE = True

    BW_TEMPLATE_THRESHOLD = 0.35
    BW_MIN_ACCEPT_SCORE = 0.34
    BW_STRONG_ACCEPT_SCORE = 0.55
    BW_AMBIGUOUS_MARGIN = 0.10
    BW_SIGN_CONFIRM_FRAMES = 2
    BW_SIGN_HOLD_FRAMES = 1

    COLOR_TEMPLATE_THRESHOLD = 0.50
    COLOR_MIN_ACCEPT_SCORE = 0.42
    COLOR_STRONG_ACCEPT_SCORE = 0.65
    COLOR_AMBIGUOUS_MARGIN = 0.10
    COLOR_SIGN_CONFIRM_FRAMES = 3
    COLOR_SIGN_HOLD_FRAMES = 2

    ORB_FEATURES = 350
    ORB_MIN_MATCHES = 6
    ORB_RATIO_TEST = 0.78

    def __init__(self, config=None, template_dir: Optional[str] = None):
        self.config = config
        self.template_dir = template_dir or self._default_template_dir()
        self.black_white_test_mode = getattr(config, "sign_black_white_test_mode", self.BLACK_WHITE_TEST_MODE)

        self.TEMPLATE_THRESHOLD = self.BW_TEMPLATE_THRESHOLD
        self.MIN_ACCEPT_SCORE = self.BW_MIN_ACCEPT_SCORE
        self.STRONG_ACCEPT_SCORE = self.BW_STRONG_ACCEPT_SCORE
        self.AMBIGUOUS_MARGIN = self.BW_AMBIGUOUS_MARGIN
        self.SIGN_CONFIRM_FRAMES = self.BW_SIGN_CONFIRM_FRAMES
        self.SIGN_HOLD_FRAMES = self.BW_SIGN_HOLD_FRAMES
        if not self.black_white_test_mode:
            self.TEMPLATE_THRESHOLD = getattr(config, "sign_match_threshold", self.COLOR_TEMPLATE_THRESHOLD)
            self.MIN_ACCEPT_SCORE = getattr(config, "sign_min_accept_score", self.COLOR_MIN_ACCEPT_SCORE)
            self.STRONG_ACCEPT_SCORE = getattr(config, "sign_strong_accept_score", self.COLOR_STRONG_ACCEPT_SCORE)
            self.AMBIGUOUS_MARGIN = getattr(config, "sign_ambiguous_margin", self.COLOR_AMBIGUOUS_MARGIN)
            self.SIGN_CONFIRM_FRAMES = getattr(config, "sign_confirm_frames", self.COLOR_SIGN_CONFIRM_FRAMES)
            self.SIGN_HOLD_FRAMES = getattr(config, "sign_hold_frames", self.COLOR_SIGN_HOLD_FRAMES)

        self.templates: List[Dict] = []
        self.templates_by_label: Dict[str, List[Dict]] = {}
        self.last_result = SignResult()
        self.hold_result = SignResult()
        self.hold_counter = 0
        self.pending_label = "YOK"
        self.pending_count = 0

        self.last_debug_frame = None
        self.last_candidate_count = 0
        self.last_best_label = "YOK"
        self.last_best_score = 0.0
        self.last_second_label = "YOK"
        self.last_second_score = 0.0
        self.last_margin = 0.0
        self.last_best_template = ""
        self.last_best_source = "none"
        self.last_best_geometry = "BELIRSIZ"
        self.last_reject_reason = ""
        self.last_debug_roi = None
        self.last_debug_mask = None
        self.last_debug_template = None
        self.debug_frame_count = 0
        self.debug_terminal = getattr(config, "sign_debug_terminal", True)
        self.debug_interval = max(1, int(getattr(config, "sign_debug_interval", 10)))

        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _default_template_dir(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        local_templates = os.path.join(parent_dir, "templates")
        if os.path.isdir(local_templates):
            return local_templates
        return os.path.join(current_dir, "templates")

    def load_templates(self) -> None:
        self.templates = []
        self.templates_by_label = {}
        if not os.path.isdir(self.template_dir):
            print(f"Template klasoru bulunamadi: {self.template_dir}")
            return

        folders = []
        roi_dir = os.path.join(self.template_dir, "roi")
        mask_dir = os.path.join(self.template_dir, "mask")
        if os.path.isdir(roi_dir):
            folders.append((roi_dir, "roi"))
        if os.path.isdir(mask_dir):
            folders.append((mask_dir, "mask"))
        if not folders:
            folders.append((self.template_dir, "roi"))

        for folder, source in folders:
            self._load_templates_from_folder(folder, source)
        self._load_samples_if_enabled()

        for template in self.templates:
            self.templates_by_label.setdefault(template["label"], []).append(template)
        print(f"Yuklenen template sayisi: {len(self.templates)}")

    def _load_samples_if_enabled(self):
        load_samples = getattr(self.config, "sign_load_samples", False)
        sample_roots = [
            os.path.join(self.template_dir, "samples"),
            os.path.join(os.path.dirname(self.template_dir), "sign_trainer", "samples"),
        ]
        for sample_root in sample_roots:
            if not load_samples and sample_root != os.path.join(self.template_dir, "samples"):
                continue
            if not os.path.isdir(sample_root):
                continue
            for label_name in os.listdir(sample_root):
                label_dir = os.path.join(sample_root, label_name)
                if os.path.isdir(label_dir):
                    self._load_templates_from_folder(label_dir, "sample", forced_label=self._label_from_file(label_name))

    def _load_templates_from_folder(self, folder, source, forced_label=None):
        for file_name in os.listdir(folder):
            if not file_name.lower().endswith(self.SUPPORTED_EXTENSIONS):
                continue
            path = os.path.join(folder, file_name)
            image = self._read_image(path)
            if image is None:
                print(f"Template okunamadi, atlaniyor: {path}")
                continue
            prepared = self._prepare_template(image, source)
            if prepared is None:
                print(f"Template islenemedi, atlaniyor: {path}")
                continue
            prepared.update({
                "label": forced_label or self._label_from_file(file_name),
                "file": file_name,
                "template_source": source,
            })
            self.templates.append(prepared)

    def process(self, frame) -> SignResult:
        self.detect(frame)
        return self.last_result

    def detect(self, frame):
        self.debug_frame_count += 1
        debug_frame = frame.copy() if frame is not None else None
        self._reset_debug_state()

        if frame is None or frame.size == 0:
            self.last_reject_reason = "NO_FRAME"
            self._print_detection_debug()
            return self._return_yok(debug_frame)
        if not self.templates:
            self.last_reject_reason = "NO_TEMPLATE"
            self._print_detection_debug()
            return self._return_yok(debug_frame)

        processed = self._preprocess_frame(frame)
        work_frame, frame_scale = self._resize_frame(processed)
        candidates = self._find_candidates(work_frame, frame_scale)
        self.last_candidate_count = len(candidates)
        best = self._classify_candidates(work_frame, candidates)
        self._update_best_debug(best)

        if best is None:
            if not self.last_reject_reason:
                self.last_reject_reason = "NO_CANDIDATE" if not candidates else "LOW_SCORE"
            self._clear_pending()
            if debug_frame is not None:
                self._draw_status(debug_frame)
            debug_frame = self._compose_debug_frame(debug_frame)
            self._print_reject_debug()
            return self._return_yok(debug_frame)

        if best["score"] < self.TEMPLATE_THRESHOLD:
            self.last_reject_reason = "LOW_SCORE"
            self._clear_pending()
            if debug_frame is not None:
                self._draw_debug(debug_frame, best["bbox"], best["label"], best["score"], "LOW")
                self._draw_status(debug_frame)
            debug_frame = self._compose_debug_frame(debug_frame)
            self._print_reject_debug()
            return self._return_yok(debug_frame, best["score"])

        confirmed_label = self._confirm(best["label"], best["score"])
        if confirmed_label == "YOK":
            self.last_reject_reason = "CONFIRM_PENDING"
            if debug_frame is not None:
                self._draw_debug(debug_frame, best["bbox"], best["label"], best["score"], "PENDING")
                self._draw_status(debug_frame)
            debug_frame = self._compose_debug_frame(debug_frame)
            self._print_reject_debug()
            return self._return_yok(debug_frame, best["score"], clear_hold=False)

        self.last_reject_reason = ""
        result = SignResult(
            label=confirmed_label,
            confidence=best["score"],
            bbox=best["bbox"],
            kind="TRAINER_MASK_GEOMETRY",
            symbol_mask=best["symbol_mask"],
            area=best["bbox"][2] * best["bbox"][3],
            match_score=best["score"],
            color_hint=best["color_hint"],
            shape_hint=best["shape_hint"],
            geometry_hint=best["geometry_hint"],
            source=best["source"],
        )
        self.last_result = result

        if best["score"] >= self.STRONG_ACCEPT_SCORE:
            self.hold_result = result
            self.hold_counter = self.SIGN_HOLD_FRAMES
        else:
            self.hold_counter = 0
            self.hold_result = SignResult()

        if debug_frame is not None:
            self._draw_debug(debug_frame, best["bbox"], confirmed_label, best["score"], best["template_file"])
            self._draw_status(debug_frame)
            debug_frame = self._compose_debug_frame(debug_frame)
        self.last_debug_frame = debug_frame
        self._print_detection_debug()
        return result.label, debug_frame

    def _preprocess_frame(self, frame):
        if frame is None or frame.size == 0:
            return frame
        filtered = cv2.GaussianBlur(frame, (3, 3), 0)
        filtered = cv2.bilateralFilter(filtered, 5, 35, 35)
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    def create_sign_mask(self, roi):
        if roi is None or roi.size == 0:
            return None
        gray = self._to_gray_image(roi)
        if gray is None:
            return None
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        mask = cv2.inRange(gray, 0, 90)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def _adaptive_helper_mask(self, roi):
        gray = self._to_gray_image(roi)
        if gray is None:
            return None
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        block_size = 31
        min_dim = min(gray.shape[:2])
        if min_dim < block_size:
            block_size = max(3, min_dim // 2 * 2 + 1)
        mask_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            7,
        )
        mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.morphologyEx(mask_adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    def shape_hint(self, mask):
        if mask is None or mask.size == 0:
            return "YOK"
        h, w = mask.shape[:2]
        vertical_ratio = cv2.countNonZero(mask[:, w // 2]) / float(max(1, h))
        upper_ratio = cv2.countNonZero(mask[h // 4, :]) / float(max(1, w))
        fill = cv2.countNonZero(mask) / float(max(1, h * w))
        if vertical_ratio > 0.45 and upper_ratio > 0.25:
            return "CIKMAZ-YOL"
        if fill < 0.18 and vertical_ratio < 0.35:
            return "PARK-LEVHASI"
        return "BELIRSIZ"

    def blue_red_hint(self, roi):
        if roi is None or roi.size == 0:
            return "YOK"
        if roi.ndim == 2:
            return "BELIRSIZ"
        if roi.ndim == 3 and roi.shape[2] == 4:
            roi = roi[:, :, :3]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
        area = float(max(1, roi.shape[0] * roi.shape[1]))
        red_ratio = cv2.countNonZero(red) / area
        blue_ratio = cv2.countNonZero(blue) / area
        if red_ratio > 0.03 and blue_ratio > 0.16:
            return "CIKMAZ-YOL"
        if red_ratio < 0.015 and blue_ratio > 0.18:
            return "PARK-LEVHASI"
        if red_ratio > 0.05:
            return "RED"
        if blue_ratio > 0.08:
            return "BLUE"
        return "BELIRSIZ"

    def _find_candidates(self, frame, frame_scale):
        candidates = []
        candidates.extend(self._find_edge_candidates(frame, frame_scale))
        candidates.extend(self._find_color_candidates(frame, frame_scale))
        return self._dedupe_candidates(candidates)[:8]

    def _group_candidate_mask(self, mask):
        kernel = np.ones((11, 11), np.uint8)
        grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        grouped = cv2.dilate(grouped, np.ones((7, 7), np.uint8), iterations=1)
        return grouped

    def _find_edge_candidates(self, frame, frame_scale):
        gray = self._to_gray_image(frame)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 130)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return self._contours_to_candidates(edges, frame, frame_scale, 0.18, "edge")

    def _find_color_candidates(self, frame, frame_scale):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
        blue = cv2.inRange(hsv, (90, 55, 45), (130, 255, 255))
        mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        return self._contours_to_candidates(mask, frame, frame_scale, 0.20, "color")

    def _contours_to_candidates(self, mask, frame, frame_scale, pad_ratio, source):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        inv_scale = 1.0 / max(frame_scale, 1e-6)
        frame_h, frame_w = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CANDIDATE_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 24 or h < 24:
                continue
            aspect = w / float(max(h, 1))
            if not (0.45 <= aspect <= 2.20):
                continue
            if w > frame_w * 0.55 or h > frame_h * 0.55:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            if cy > frame_h * 0.78:
                continue
            if cx < frame_w * 0.08 or cx > frame_w * 0.92:
                continue

            fill_ratio = area / float(max(1, w * h))
            if source == "dark" and not (0.025 <= fill_ratio <= 0.88):
                continue
            if source == "edge" and not (0.045 <= fill_ratio <= 0.98):
                continue
            if source == "color" and fill_ratio < 0.06:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if source in ("edge", "color") and not (3 <= len(approx) <= 6):
                continue

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            contour_geom = self._geometry_hint_from_mask(contour_mask)
            if source in ("edge", "color") and contour_geom == "BELIRSIZ":
                continue

            pad = max(8, int(max(w, h) * pad_ratio))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_w, x + w + pad)
            y2 = min(frame_h, y + h + pad)
            candidates.append({
                "bbox_work": (x1, y1, x2 - x1, y2 - y1),
                "bbox_orig": (
                    int(x1 * inv_scale),
                    int(y1 * inv_scale),
                    int((x2 - x1) * inv_scale),
                    int((y2 - y1) * inv_scale),
                ),
                "area": int(area * inv_scale * inv_scale),
                "source": source,
                "geometry_hint": contour_geom,
            })
        return candidates

    def _classify_candidates(self, frame, candidates):
        if not candidates:
            self.last_reject_reason = "NO_CANDIDATE"
            return None

        results = []
        for candidate in candidates:
            x, y, w, h = candidate["bbox_work"]
            roi = frame[y:y + h, x:x + w]
            prepared = self._prepare_candidate(roi)
            if prepared is None:
                continue
            color_hint = self.blue_red_hint(roi)
            shape_hint = self.shape_hint(prepared["mask"])
            for template in self.templates:
                score, reason = self._compare(prepared, template, color_hint)
                results.append({
                    "label": template["label"],
                    "score": score,
                    "bbox": candidate["bbox_orig"],
                    "template_file": template["file"],
                    "source": candidate["source"],
                    "symbol_mask": prepared["symbol_mask"],
                    "color_hint": color_hint,
                    "shape_hint": shape_hint,
                    "geometry_hint": prepared["geometry_hint"],
                    "roi": roi.copy(),
                    "mask": prepared["mask"],
                    "template_debug": template["mask"],
                    "reject_hint": reason,
                })

        if not results:
            self.last_reject_reason = "NO_CANDIDATE"
            return None

        results.sort(key=lambda item: item["score"], reverse=True)
        best = results[0]
        second = self._best_different_label(results, best["label"])
        self._remember_second(second, best)

        if best["reject_hint"] == "GEOMETRY_MISMATCH" and best["score"] < self.STRONG_ACCEPT_SCORE:
            self._update_best_debug(best)
            self.last_reject_reason = "GEOMETRY_MISMATCH"
            return None
        if best["score"] < self.MIN_ACCEPT_SCORE:
            self.last_reject_reason = "LOW_SCORE"
            return best
        if second is not None:
            margin = best["score"] - second["score"]
            if margin < self.AMBIGUOUS_MARGIN:
                self._update_best_debug(best)
                self.last_reject_reason = "AMBIGUOUS"
                return None
        return best

    def _compare(self, candidate, template, color_hint):
        mask_iou = self._mask_iou(candidate["mask"], template["mask"])
        adaptive_iou = self._mask_iou(candidate["adaptive_mask"], template["adaptive_mask"])
        mask_iou = max(mask_iou, 0.70 * adaptive_iou)
        edge_score = self._edge_similarity(candidate["edges"], template["edges"])
        shape_score = self._shape_similarity(candidate["mask"], template["mask"])
        profile_score = self._profile_similarity(candidate["profile"], template["profile"])
        symbol_score = self._mask_iou(candidate["symbol_mask"], template["symbol_mask"])
        orb_score = self._orb_similarity(candidate["descriptors"], template["descriptors"])

        score = (
            0.34 * mask_iou
            + 0.26 * edge_score
            + 0.22 * shape_score
            + 0.09 * profile_score
            + 0.05 * symbol_score
            + 0.04 * orb_score
        )

        reason = ""
        cand_geom = candidate["geometry_hint"]
        templ_geom = template["geometry_hint"]
        if cand_geom != "BELIRSIZ" and templ_geom != "BELIRSIZ":
            if cand_geom == templ_geom:
                score += 0.08
            elif {cand_geom, templ_geom} == {"UCGEN", "DIKDORTGEN"}:
                score -= 0.18
                reason = "GEOMETRY_MISMATCH"

        if color_hint != "BELIRSIZ" and color_hint == template["label"]:
            score += 0.05
        return max(0.0, min(1.0, float(score))), reason

    def _prepare_template(self, image, source):
        if source == "mask":
            return self._prepare_from_mask_image(image)
        return self._prepare_candidate(image)

    def _prepare_candidate(self, image):
        if image is None or image.size == 0:
            return None
        bgr = self._to_bgr_image(image)
        if bgr is None:
            return None
        bgr = self._trim_to_foreground(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        bgr_small = cv2.resize(bgr, self.TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        mask = self.create_sign_mask(bgr_small)
        if mask is None:
            return None
        adaptive_mask = self._adaptive_helper_mask(bgr_small)
        if cv2.countNonZero(mask) < 20:
            if adaptive_mask is not None:
                mask = adaptive_mask
        return self._prepared_from_gray_mask(gray, mask, adaptive_mask)

    def _prepare_from_mask_image(self, image):
        gray = self._to_gray_image(image)
        if gray is None:
            return None
        gray = cv2.resize(gray, self.TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(mask) > mask.size * 0.55:
            mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return self._prepared_from_gray_mask(gray, mask, None)

    def _prepared_from_gray_mask(self, gray, mask, adaptive_mask=None):
        if cv2.countNonZero(mask) < 20:
            return None
        if adaptive_mask is None:
            adaptive_mask = mask
        edges = cv2.Canny(gray, 45, 130)
        edges = cv2.bitwise_or(edges, cv2.Canny(mask, 45, 130))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        symbol_mask = self._extract_symbol_mask(mask)
        symbol_edges = cv2.Canny(symbol_mask, 45, 130)
        return {
            "gray": gray,
            "mask": mask,
            "adaptive_mask": adaptive_mask,
            "edges": edges,
            "symbol_mask": symbol_mask,
            "symbol_edges": symbol_edges,
            "geometry_hint": self._geometry_hint_from_mask(mask),
            "shape_features": self._shape_features(mask),
            "profile": self._profile_features(mask),
            "descriptors": self._orb_descriptors(gray, edges),
        }

    def _trim_to_foreground(self, bgr):
        mask = self.create_sign_mask(bgr)
        if mask is None or cv2.countNonZero(mask) < 20:
            return bgr
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bgr
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        img_h, img_w = bgr.shape[:2]
        if w < 18 or h < 18:
            return bgr
        pad = max(4, int(max(w, h) * 0.08))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        return bgr[y1:y2, x1:x2]

    def _geometry_hint_from_mask(self, mask):
        if mask is None or mask.size == 0:
            return "BELIRSIZ"
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "BELIRSIZ"
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < mask.size * 0.025:
            return "BELIRSIZ"
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return "BELIRSIZ"
        approx = cv2.approxPolyDP(contour, 0.045 * perimeter, True)
        corners = len(approx)
        if corners == 3:
            return "UCGEN"
        if corners == 4:
            return "DIKDORTGEN"
        return "BELIRSIZ"

    def _extract_symbol_mask(self, mask):
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symbol = np.zeros_like(mask)
        total_area = h * w
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total_area * 0.002:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            touches_border = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
            too_large = area > total_area * 0.42 or cw > w * 0.86 or ch > h * 0.86
            if touches_border or too_large:
                continue
            cv2.drawContours(symbol, [contour], -1, 255, thickness=cv2.FILLED)
        if cv2.countNonZero(symbol) >= 12:
            return self._clean_symbol_mask(symbol)

        central = np.zeros_like(mask)
        y1, y2 = int(h * 0.15), int(h * 0.88)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        central[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return self._clean_symbol_mask(central)

    def _shape_features(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"fill": 0.0, "extent": 0.0, "solidity": 0.0}
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return {
            "fill": cv2.countNonZero(mask) / float(max(1, mask.size)),
            "extent": area / float(max(1, w * h)),
            "solidity": area / float(max(1.0, hull_area)),
        }

    def _profile_features(self, mask):
        binary = (mask > 0).astype(np.float32)
        row_profile = np.mean(binary, axis=1)
        col_profile = np.mean(binary, axis=0)
        return {
            "fill": float(np.mean(binary)),
            "rows": np.array([float(np.mean(chunk)) for chunk in np.array_split(row_profile, 8)]),
            "cols": np.array([float(np.mean(chunk)) for chunk in np.array_split(col_profile, 8)]),
            "hist": cv2.calcHist([mask], [0], None, [16], [0, 256]).flatten().astype(np.float32) / float(mask.size),
        }

    def _profile_similarity(self, a, b):
        diff = (
            0.22 * abs(a["fill"] - b["fill"])
            + 0.36 * float(np.mean(np.abs(a["rows"] - b["rows"])))
            + 0.36 * float(np.mean(np.abs(a["cols"] - b["cols"])))
            + 0.06 * float(np.mean(np.abs(a["hist"] - b["hist"])))
        )
        return max(0.0, 1.0 - diff * 2.6)

    def _shape_similarity(self, a, b):
        a_contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        b_contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not a_contours or not b_contours:
            return 0.0
        a_cnt = max(a_contours, key=cv2.contourArea)
        b_cnt = max(b_contours, key=cv2.contourArea)
        distance = cv2.matchShapes(a_cnt, b_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        return max(0.0, 1.0 - min(1.0, distance * 2.2))

    def _edge_similarity(self, a, b):
        return max(self._correlation(a, b), self._mask_iou(a, b))

    def _mask_iou(self, a, b):
        a_bool = a > 0
        b_bool = b > 0
        intersection = np.logical_and(a_bool, b_bool).sum()
        union = np.logical_or(a_bool, b_bool).sum()
        return float(intersection) / float(max(1, union))

    def _correlation(self, a, b):
        try:
            result = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            return max(0.0, float(score))
        except cv2.error:
            return 0.0

    def _orb_descriptors(self, gray, edges):
        keypoints = self.orb.detect(gray, edges)
        if not keypoints:
            return None
        _, descriptors = self.orb.compute(gray, keypoints)
        return descriptors

    def _orb_similarity(self, candidate_descriptors, template_descriptors):
        if candidate_descriptors is None or template_descriptors is None:
            return 0.0
        if len(candidate_descriptors) < 2 or len(template_descriptors) < 2:
            return 0.0
        try:
            matches = self.matcher.knnMatch(candidate_descriptors, template_descriptors, k=2)
        except cv2.error:
            return 0.0

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ORB_RATIO_TEST * second.distance:
                good.append(first)
        if not good:
            return 0.0
        match_ratio = len(good) / float(max(self.ORB_MIN_MATCHES, min(len(candidate_descriptors), len(template_descriptors))))
        distance_score = 1.0 - min(1.0, float(np.mean([m.distance for m in good])) / 80.0)
        return max(0.0, min(1.0, 0.70 * match_ratio + 0.30 * distance_score))

    def _dedupe_candidates(self, candidates):
        candidates.sort(key=lambda item: item["area"], reverse=True)
        kept = []
        for candidate in candidates:
            if all(self._iou(candidate["bbox_work"], old["bbox_work"]) < 0.42 for old in kept):
                kept.append(candidate)
        return kept

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / float(max(1, union))

    def _confirm(self, label, score):
        if score >= self.STRONG_ACCEPT_SCORE:
            self.pending_label = label
            self.pending_count = self.SIGN_CONFIRM_FRAMES
            return label
        if label == self.pending_label:
            self.pending_count += 1
        else:
            self.pending_label = label
            self.pending_count = 1
        if self.pending_count >= self.SIGN_CONFIRM_FRAMES:
            return label
        return "YOK"

    def _clear_pending(self):
        self.pending_label = "YOK"
        self.pending_count = 0

    def _return_yok(self, debug_frame, confidence=0.0, clear_hold=True):
        if clear_hold:
            self.hold_counter = 0
            self.hold_result = SignResult()
        self.last_result = SignResult(label="YOK", confidence=confidence, reject_reason=self.last_reject_reason)
        self.last_debug_frame = debug_frame
        return "YOK", debug_frame

    def _reset_debug_state(self):
        self.last_candidate_count = 0
        self.last_best_label = "YOK"
        self.last_best_score = 0.0
        self.last_second_label = "YOK"
        self.last_second_score = 0.0
        self.last_margin = 0.0
        self.last_best_template = ""
        self.last_best_source = "none"
        self.last_best_geometry = "BELIRSIZ"
        self.last_reject_reason = ""
        self.last_debug_roi = None
        self.last_debug_mask = None
        self.last_debug_template = None

    def _update_best_debug(self, best):
        if best is None:
            return
        self.last_best_label = best["label"]
        self.last_best_score = best["score"]
        self.last_best_template = best["template_file"]
        self.last_best_source = best["source"]
        self.last_best_geometry = best["geometry_hint"]
        self.last_debug_roi = best.get("roi")
        self.last_debug_mask = best.get("mask")
        self.last_debug_template = best.get("template_debug")

    def _remember_second(self, second, best):
        if second is None:
            self.last_second_label = "YOK"
            self.last_second_score = 0.0
            self.last_margin = best["score"]
            return
        self.last_second_label = second["label"]
        self.last_second_score = second["score"]
        self.last_margin = best["score"] - second["score"]

    def _best_different_label(self, results, label):
        for item in results[1:]:
            if item["label"] != label:
                return item
        return None

    def _compose_debug_frame(self, original):
        if original is None:
            return None
        h, w = original.shape[:2]
        panel_w = max(240, w // 2)
        panel_h = max(160, h // 2)

        original_panel = self._panel_image(original, panel_w, panel_h, "Original kamera")
        roi_panel = self._panel_image(self.last_debug_roi, panel_w, panel_h, "Aday ROI")
        mask_panel = self._panel_image(self.last_debug_mask, panel_w, panel_h, "ROI maskesi")
        template_panel = self._panel_image(self.last_debug_template, panel_w, panel_h, "En iyi template")

        top = np.hstack((original_panel, roi_panel))
        bottom = np.hstack((mask_panel, template_panel))
        return np.vstack((top, bottom))

    def _panel_image(self, image, width, height, title):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        if image is None:
            cv2.putText(panel, "YOK", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            resized = self._fit_image(image, width, height - 24)
            y = 24 + (height - 24 - resized.shape[0]) // 2
            x = (width - resized.shape[1]) // 2
            panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(panel, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return panel

    def _fit_image(self, image, max_width, max_height):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((max_height, max_width, 3), dtype=np.uint8)
        scale = min(max_width / float(w), max_height / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_debug(self, frame, bbox, label, score, note):
        x, y, w, h = bbox
        color = (0, 255, 0) if score >= self.TEMPLATE_THRESHOLD else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x, max(18, y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, str(note), (x, max(36, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_status(self, frame):
        text = (
            f"cand={self.last_candidate_count} "
            f"best={self.last_best_label} {self.last_best_score:.2f} "
            f"second={self.last_second_label} {self.last_second_score:.2f} "
            f"margin={self.last_margin:.2f} src={self.last_best_source} "
            f"geom={self.last_best_geometry} pending={self.pending_label}:{self.pending_count} "
            f"hold={self.hold_counter} reject={self.last_reject_reason} {self.last_best_template}"
        )
        cv2.putText(frame, text[:190], (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    def _print_detection_debug(self):
        if not self.debug_terminal or self.debug_frame_count % self.debug_interval != 0:
            return
        status = self.last_reject_reason or "ACCEPTED"
        print(
            f"sign {status}: cand={self.last_candidate_count} "
            f"best={self.last_best_label} {self.last_best_score:.2f} "
            f"second={self.last_second_label} {self.last_second_score:.2f} "
            f"margin={self.last_margin:.2f} src={self.last_best_source} "
            f"geom={self.last_best_geometry} pending={self.pending_label}:{self.pending_count} "
            f"hold={self.hold_counter} template={self.last_best_template}"
        )

    def _print_reject_debug(self):
        if not self.debug_terminal or not self.last_reject_reason:
            return
        if self.last_reject_reason not in {"NO_CANDIDATE", "LOW_SCORE", "AMBIGUOUS", "GEOMETRY_MISMATCH", "NO_TEMPLATE"}:
            return
        print(
            f"sign reject={self.last_reject_reason} cand={self.last_candidate_count} "
            f"best={self.last_best_label} {self.last_best_score:.2f} "
            f"second={self.last_second_label} {self.last_second_score:.2f} "
            f"margin={self.last_margin:.2f} src={self.last_best_source} geom={self.last_best_geometry}"
        )

    def _to_bgr_image(self, image):
        if image is None:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            return (rgb * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
        if image.ndim == 3:
            return image[:, :, :3]
        return None

    def _to_gray_image(self, image):
        if image is None:
            return None
        if image.ndim == 2:
            return image
        bgr = self._to_bgr_image(image)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _clean_symbol_mask(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]
        if w <= self.MAX_FRAME_WIDTH:
            return frame, 1.0
        scale = self.MAX_FRAME_WIDTH / float(w)
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale

    def _label_from_file(self, file_name):
        name = os.path.splitext(file_name)[0].lower()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"^[0-9]+[_-]?", "", name)
        name = re.sub(r"[_-]?[0-9]+$", "", name)
        name = name.replace("_roi", "").replace("-roi", "")
        name = name.replace("_mask", "").replace("-mask", "")
        name = name.replace("_", "-")
        aliases = {
            "stop": "STOP",
            "dur": "STOP",
            "sol": "SOL",
            "left": "SOL",
            "sag": "SAG",
            "right": "SAG",
            "ileri": "ILERI",
            "forward": "ILERI",
            "park-levhasi": "PARK-LEVHASI",
            "park-levhas": "PARK-LEVHASI",
            "yaya-gecidi": "YAYA-GECIDI",
            "hemzemin-gecit": "HEMZEMIN-GECIT",
            "kasisli-yol": "KASISLI-YOL",
            "sollama-serbest": "SOLLAMA-SERBEST",
            "cikmaz-yol": "CIKMAZ-YOL",
        }
        return aliases.get(name, name.upper())

    def _read_image(self, path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    def _find_warning_triangle(self, frame):
        _, debug_frame = self.detect(frame)
        return self.last_result.bbox, debug_frame
