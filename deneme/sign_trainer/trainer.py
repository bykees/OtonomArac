import os
import sys
import time

import cv2
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DENEME_DIR = os.path.dirname(CURRENT_DIR)

if DENEME_DIR not in sys.path:
    sys.path.insert(0, DENEME_DIR)

from pi_system.camera import CameraStream
from pi_system.config import AppConfig
from pi_system.sign_detection import SignDetector


SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")

LABEL_KEYS = {
    ord("1"): "KASISLI-YOL",
    ord("2"): "HEMZEMIN-GECIT",
    ord("3"): "YAYA-GECIDI",
    ord("4"): "PARK-LEVHASI",
    ord("5"): "SOLLAMA-SERBEST",
    ord("6"): "CIKMAZ-YOL",
    ord("7"): "STOP",
    ord("8"): "SOL",
    ord("9"): "SAG",
    ord("0"): "ILERI",
}


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    config = AppConfig()
    camera = CameraStream(config.camera)
    TEMPLATES_DIR = os.path.join(DENEME_DIR, "templates")
    detector = SignDetector(config.detection, template_dir=TEMPLATES_DIR)
    detector.load_templates()
    camera.open()

    last_roi = None
    last_mask = None
    last_prediction = "YOK"
    last_score = 0.0
    color_hint = "YOK"

    print("Tabela trainer basladi.")
    print("Tuslar: 1=KASIS, 2=HEMZEMIN, 3=YAYA, 4=PARK, 5=SOLLAMA, 6=CIKMAZ")
    print("Tuslar: 7=STOP, 8=SOL, 9=SAG, 0=ILERI, s=tahmin etiketiyle kaydet, r=manuel ROI, q=cikis")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Kamera goruntusu alinamadi.")
                break

            display = frame.copy()
            sign, debug_frame = detector.detect(frame)
            result = detector.last_result
            last_prediction = result.label
            last_score = result.match_score if result.match_score else result.confidence

            shape = "YOK"
            color_hint = "YOK"

            if result.bbox is not None:
                x, y, w, h = result.bbox
                last_roi = frame[y:y + h, x:x + w].copy()

                last_mask = create_sign_mask(last_roi)
                shape = shape_hint(last_mask)
                color_hint = blue_red_hint(last_roi)

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.putText(
                display,
                f"Prediction: {last_prediction} score={last_score:.2f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Color hint: {color_hint}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                "1 KASIS | 2 HEMZEMIN | 3 YAYA | 4 PARK | 5 SOLLAMA | 6 CIKMAZ | r ROI | s save | q quit",
                (10, display.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                display,
                f"Shape hint: {shape}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Trainer Camera", display)

            if debug_frame is not None:
                cv2.imshow("Trainer Debug", debug_frame)

            if last_roi is not None:
                cv2.imshow("Last ROI", last_roi)

            if last_mask is not None:
                cv2.imshow("Last Mask", last_mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("r"):
                selected = cv2.selectROI("Trainer Camera", frame, False, False)
                sx, sy, sw, sh = selected

                if sw > 0 and sh > 0:
                    last_roi = frame[int(sy):int(sy + sh), int(sx):int(sx + sw)].copy()
                    last_mask = create_sign_mask(last_roi)
                    color_hint = blue_red_hint(last_roi)

                    print("Manuel ROI secildi:", selected)
                    print("Color hint:", color_hint)
                else:
                    print("ROI secilmedi.")

            if key == ord("s"):
                if last_prediction != "YOK":
                    save_sample(last_prediction, last_roi, last_mask)
                continue

            if key in LABEL_KEYS:
                save_sample(LABEL_KEYS[key], last_roi, last_mask)

    finally:
        camera.release()
        cv2.destroyAllWindows()


def create_sign_mask(roi):
    if roi is None:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = cv2.inRange(gray, 0, 90)

    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask

def shape_hint(mask):
    if mask is None:
        return "YOK"

    h, w = mask.shape

    # Ortadaki dikey çizgiyi kontrol et
    vertical_slice = mask[:, w//2]

    vertical_ratio = cv2.countNonZero(vertical_slice) / h

    # Üstte yatay çizgi var mı (çıkmaz yol)
    top_slice = mask[h//4, :]

    horizontal_ratio = cv2.countNonZero(top_slice) / w

    # karar
    if vertical_ratio > 0.5 and horizontal_ratio > 0.3:
        return "CIKMAZ-YOL"

    if vertical_ratio < 0.3 and horizontal_ratio < 0.2:
        return "PARK-LEVHASI"

    return "BELIRSIZ"

def blue_red_hint(roi):
    if roi is None:
        return "YOK"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    blue_mask = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))

    total_area = roi.shape[0] * roi.shape[1]

    red_ratio = cv2.countNonZero(red_mask) / total_area
    blue_ratio = cv2.countNonZero(blue_mask) / total_area

    if red_ratio > 0.03 and blue_ratio > 0.20:
        return "CIKMAZ-YOL"

    if red_ratio < 0.01 and blue_ratio > 0.20:
        return "PARK-LEVHASI"

    return "BELIRSIZ"


def save_image_unicode(path, image):
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        return False

    encoded.tofile(path)
    return os.path.exists(path)


def save_sample(label, roi, mask):
    if roi is None:
        print("Kaydedilecek ROI yok.")
        return

    label_dir = os.path.join(SAMPLES_DIR, safe_name(label))
    os.makedirs(label_dir, exist_ok=True)

    stamp = int(time.time() * 1000)
    roi_path = os.path.join(label_dir, f"{stamp}_roi.png")
    mask_path = os.path.join(label_dir, f"{stamp}_mask.png")

    ok_roi = save_image_unicode(roi_path, roi)
    ok_mask = True

    if mask is not None:
        ok_mask = save_image_unicode(mask_path, mask)

    if ok_roi and ok_mask:
        print(f"Kaydedildi: {label} -> {roi_path}")
    else:
        print("KAYIT BASARISIZ!")
        print("ROI path:", roi_path)
        print("MASK path:", mask_path)


def safe_name(label):
    return label.lower().replace(" ", "-").replace("_", "-")


if __name__ == "__main__":
    main()