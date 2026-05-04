import cv2
import numpy as np


class SeritTakip:
    def __init__(self, camera_index=1, debug=True):
        self.camera_index = camera_index
        self.DEBUG = debug

        # Raspberry Pi 3 için düşük çözünürlük
        self.WIDTH = 320
        self.HEIGHT = 240

        # Sadece alt kısmı işle
        self.ROI_START = 0.65  # alt %35

        # Şerit parametreleri
        self.MAX_CONF = 5
        self.DEFAULT_LANE_WIDTH = 120
        self.LANE_MIN_AREA = 120
        self.LANE_STD_COEF = 0.9

        # Yumuşatma
        self.LANE_SMOOTH_ALPHA = 0.30
        self.ERROR_SMOOTH_ALPHA = 0.35

        # Şerit hafızası
        self.lane_memory = {
            "left": {"x": None, "confidence": 0},
            "right": {"x": None, "confidence": 0}
        }

        self.last_lane_width = self.DEFAULT_LANE_WIDTH
        self.last_error = 0

    # -------------------------------------------------
    # Yardımcı fonksiyonlar
    # -------------------------------------------------
    def smooth_value(self, old, new, alpha=None):
        if alpha is None:
            alpha = self.LANE_SMOOTH_ALPHA

        if old is None:
            return int(new)

        return int(alpha * new + (1 - alpha) * old)

    def update_lane_memory(self, side, x_value):
        if x_value is not None:
            self.lane_memory[side]["x"] = self.smooth_value(
                self.lane_memory[side]["x"], x_value
            )
            self.lane_memory[side]["confidence"] = min(
                self.MAX_CONF, self.lane_memory[side]["confidence"] + 1
            )
        else:
            self.lane_memory[side]["confidence"] = max(
                0, self.lane_memory[side]["confidence"] - 1
            )
            if self.lane_memory[side]["confidence"] == 0:
                self.lane_memory[side]["x"] = None

    def adaptive_lane_mask(self, gray_roi):
        h, w = gray_roi.shape
        mask = np.zeros_like(gray_roi)

        # Her 5 satır için ayrı threshold
        for y in range(0, h, 5):
            row = gray_roi[y:y + 5]
            if row.size == 0:
                continue

            # Eski koddaki mantık: koyu / açık farkını satıra göre ayır
            thresh = np.mean(row) - self.LANE_STD_COEF * np.std(row)
            mask[y:y + 5] = np.where(row < thresh, 255, 0)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_lane_points(self, mask):
        h, w = mask.shape

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        left_x = None
        right_x = None
        left_best_area = 0
        right_best_area = 0
        center = w // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.LANE_MIN_AREA:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = x + cw // 2

            if cx < center:
                if area > left_best_area:
                    left_best_area = area
                    left_x = cx
            else:
                if area > right_best_area:
                    right_best_area = area
                    right_x = cx

        return left_x, right_x

    def get_lane_center(self, left_x, right_x, frame_width):
        self.update_lane_memory("left", left_x)
        self.update_lane_memory("right", right_x)

        left = self.lane_memory["left"]["x"]
        right = self.lane_memory["right"]["x"]

        # Şerit genişliğini güncelle
        if left is not None and right is not None:
            measured_width = right - left
            if 50 < measured_width < 220:
                self.last_lane_width = int(
                    0.9 * self.last_lane_width + 0.1 * measured_width
                )

        # Tek şerit varsa diğerini tahmin et
        if left is None and right is not None:
            left = right - self.last_lane_width
        elif right is None and left is not None:
            right = left + self.last_lane_width

        # İkisi de yoksa merkeze dön
        if left is None and right is None:
            return frame_width // 2, None, None

        lane_center = (left + right) // 2
        return lane_center, left, right

    def decide_direction(self, lane_center, frame_width):
        frame_center = frame_width // 2
        raw_error = lane_center - frame_center

        error = int(
            (1 - self.ERROR_SMOOTH_ALPHA) * self.last_error +
            self.ERROR_SMOOTH_ALPHA * raw_error
        )
        self.last_error = error

        if error < -45:
            return "SERT SOL", error
        elif error < -20:
            return "SOL", error
        elif error > 45:
            return "SERT SAG", error
        elif error > 20:
            return "SAG", error
        else:
            return "DUZ", error

    def preprocess(self, frame):
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = blur.shape
        y_start = int(h * self.ROI_START)
        roi = blur[y_start:h, :]

        lane_mask = self.adaptive_lane_mask(roi)

        return frame, lane_mask, y_start

    def draw_debug(self, frame, lane_mask, y_start, lane_center, left_x, right_x, direction, error):
        debug_frame = frame.copy()
        h, w, _ = debug_frame.shape
        roi_h = lane_mask.shape[0]

        # ROI başlangıcı
        cv2.line(debug_frame, (0, y_start), (w, y_start), (0, 255, 255), 2)

        # Sol şerit
        if left_x is not None:
            cv2.circle(debug_frame, (left_x, y_start + roi_h // 2), 5, (255, 0, 0), -1)

        # Sağ şerit
        if right_x is not None:
            cv2.circle(debug_frame, (right_x, y_start + roi_h // 2), 5, (0, 255, 0), -1)

        # Şerit merkezi
        if lane_center is not None:
            cv2.line(debug_frame, (lane_center, y_start), (lane_center, h), (255, 255, 0), 2)

        # Ekran merkezi
        frame_center = w // 2
        cv2.line(debug_frame, (frame_center, y_start), (frame_center, h), (0, 0, 255), 2)

        cv2.putText(debug_frame, f"Yon: {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(debug_frame, f"Hata: {error}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(debug_frame, f"Serit Genisligi: {self.last_lane_width}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Kamera", debug_frame)
        cv2.imshow("Lane Mask", lane_mask)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        if not cap.isOpened():
            print(f"Kamera acilamadi. Index: {self.camera_index}")
            return

        print(f"Kamera acildi. Index: {self.camera_index}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kare okunamadi.")
                break

            frame, lane_mask, y_start = self.preprocess(frame)

            left_x, right_x = self.find_lane_points(lane_mask)
            lane_center, left_mem, right_mem = self.get_lane_center(left_x, right_x, self.WIDTH)
            direction, error = self.decide_direction(lane_center, self.WIDTH)

            print(f"Sol: {left_mem}, Sag: {right_mem}, Hata: {error}, Yon: {direction}")

            if self.DEBUG:
                self.draw_debug(
                    frame,
                    lane_mask,
                    y_start,
                    lane_center,
                    left_mem,
                    right_mem,
                    direction,
                    error
                )

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = SeritTakip(camera_index=1, debug=True)
    sistem.run()