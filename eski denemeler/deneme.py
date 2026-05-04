import cv2
import numpy as np


class OtonomSurusSistemi:
    def __init__(self, camera_index=1, debug=False):
        self.camera_index = camera_index
        self.DEBUG = debug

        # --- Parametreler ---
        self.MAX_CONF = 5
        self.DEFAULT_LANE_WIDTH = 160
        self.LANE_MIN_AREA = 300
        self.RED_MIN_AREA = 600
        self.BLUE_MIN_AREA = 1800
        self.TURN_ERROR_THRESHOLD = 25

        self.LANE_STD_COEF = 0.9
        self.RED_CONFIRM_FRAMES = 3
        self.SIGN_CONFIRM_FRAMES = 4

        self.LANE_SMOOTH_ALPHA = 0.30
        self.ERROR_SMOOTH_ALPHA = 0.35

        # Ok yönü tahmini için
        self.ARROW_MIN_WHITE_AREA = 120
        self.ARROW_CENTER_MARGIN = 0.08  # merkeze çok yakınsa karar verme

        # --- Durum hafızası ---
        self.lane_memory = {
            "left": {"x": None, "confidence": 0},
            "right": {"x": None, "confidence": 0}
        }

        self.last_lane_width = self.DEFAULT_LANE_WIDTH
        self.last_error = 0

        self.kirmizi_sayaci = 0
        self.tabela_sayaci = 0
        self.son_tabela = "YOK"

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

    def adaptive_lane_mask(self, v_channel):
        h, w = v_channel.shape
        mask = np.zeros_like(v_channel)

        for y in range(0, h, 5):
            row = v_channel[y:y + 5]
            if row.size == 0:
                continue

            thresh = np.mean(row) - self.LANE_STD_COEF * np.std(row)
            mask[y:y + 5] = np.where(row < thresh, 255, 0)

        kernel = np.ones((5, 5), np.uint8)
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
        merkez = w // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.LANE_MIN_AREA:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = x + cw // 2

            if cx < merkez:
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

        if left is not None and right is not None:
            measured_width = right - left
            if measured_width > 60:
                self.last_lane_width = int(
                    0.9 * self.last_lane_width + 0.1 * measured_width
                )

        if left is None and right is not None:
            left = right - self.last_lane_width
        elif right is None and left is not None:
            right = left + self.last_lane_width

        if left is None and right is None:
            return frame_width // 2, None, None

        lane_center = (left + right) // 2
        return lane_center, left, right

    def detect_red_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        red_detected = False
        best_box = None
        best_area = 0

        frame_h, frame_w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.RED_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if y > int(frame_h * 0.55):
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue

            rect_area = w * h
            if rect_area <= 0:
                continue

            fill_ratio = area / float(rect_area)
            if fill_ratio < 0.45:
                continue

            if area > best_area:
                best_box = (x, y, w, h)
                best_area = area
                red_detected = True

        if red_detected:
            self.kirmizi_sayaci += 1
        else:
            self.kirmizi_sayaci = max(0, self.kirmizi_sayaci - 1)

        return self.kirmizi_sayaci >= self.RED_CONFIRM_FRAMES, best_box

    def detect_arrow_direction(self, sign_roi):
        """
        Mavi tabela içindeki beyaz oku yaklaşık olarak yön tayini eder.
        Çalışma mantığı:
        - Beyaz alanı çıkar
        - En büyük beyaz contour'u bul
        - Contour ağırlık merkezini ROI merkezine göre karşılaştır
        """

        if sign_roi is None or sign_roi.size == 0:
            return "YOK", None

        hsv = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2HSV)

        # Beyaz sembol maskesi
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([179, 90, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return "YOK", white_mask

        best_cnt = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area and area > self.ARROW_MIN_WHITE_AREA:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return "YOK", white_mask

        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return "YOK", white_mask

        cx = int(M["m10"] / M["m00"])
        h, w = white_mask.shape[:2]
        center_x = w // 2

        normalized_offset = (cx - center_x) / float(w)

        if normalized_offset < -self.ARROW_CENTER_MARGIN:
            return "SOL", white_mask
        elif normalized_offset > self.ARROW_CENTER_MARGIN:
            return "SAG", white_mask
        else:
            return "YOK", white_mask

    def detect_blue_sign(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        sign_detected = False
        sign_box = None
        best_area = 0
        detected_sign = "YOK"

        frame_h, frame_w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.BLUE_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Tabela yaklaşık kare olmalı
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.70 or aspect_ratio > 1.30:
                continue

            # Çok aşağıdaki mavi bölgeleri ele
            if y > int(frame_h * 0.70):
                continue

            rect_area = w * h
            if rect_area <= 0:
                continue

            fill_ratio = area / float(rect_area)
            if fill_ratio < 0.50:
                continue

            # En iyi adayı seç
            if area > best_area:
                # Tabela ROI
                sign_roi = frame[y:y + h, x:x + w]

                # İçerikteki okun yönünü analiz et
                arrow_dir, _ = self.detect_arrow_direction(sign_roi)

                # Gerçekten okun yönü çıkarılamadıysa bu adayı kullanma
                if arrow_dir == "YOK":
                    continue

                best_area = area
                sign_box = (x, y, w, h)
                detected_sign = arrow_dir
                sign_detected = True

        if sign_detected:
            if detected_sign == self.son_tabela:
                self.tabela_sayaci += 1
            else:
                self.tabela_sayaci = 1
                self.son_tabela = detected_sign
        else:
            self.tabela_sayaci = max(0, self.tabela_sayaci - 1)
            if self.tabela_sayaci == 0:
                self.son_tabela = "YOK"

        if self.tabela_sayaci >= self.SIGN_CONFIRM_FRAMES:
            return self.son_tabela, sign_box

        return "YOK", sign_box

    def decide_direction(self, lane_center, frame_width, tabela="YOK"):
        frame_center = frame_width // 2
        raw_error = lane_center - frame_center

        error = int(
            (1 - self.ERROR_SMOOTH_ALPHA) * self.last_error +
            self.ERROR_SMOOTH_ALPHA * raw_error
        )
        self.last_error = error

        if tabela == "SOL":
            return "SOL", error
        elif tabela == "SAG":
            return "SAG", error

        if error < -self.TURN_ERROR_THRESHOLD:
            return "SOL", error
        elif error > self.TURN_ERROR_THRESHOLD:
            return "SAG", error
        else:
            return "DUZ", error

    def draw_debug(
        self,
        frame,
        lane_mask,
        lane_center,
        left_x,
        right_x,
        direction,
        error,
        red_box=None,
        sign_box=None,
        tabela="YOK"
    ):
        h, w = frame.shape[:2]
        frame_center = w // 2

        if left_x is not None:
            cv2.line(frame, (left_x, h // 2), (left_x, h), (255, 0, 0), 2)

        if right_x is not None:
            cv2.line(frame, (right_x, h // 2), (right_x, h), (0, 255, 255), 2)

        if lane_center is not None:
            cv2.line(frame, (lane_center, 0), (lane_center, h), (0, 255, 0), 2)

        cv2.line(frame, (frame_center, 0), (frame_center, h), (0, 0, 255), 2)

        if red_box is not None:
            x, y, bw, bh = red_box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(
                frame, "KIRMIZI", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        if sign_box is not None and tabela != "YOK":
            x, y, bw, bh = sign_box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(
                frame, f"TABELA: {tabela}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        cv2.putText(
            frame, f"Yon: {direction}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Hata: {error}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Serit Genisligi: {self.last_lane_width}", (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        cv2.imshow("Lane Mask", lane_mask)
        cv2.imshow("Otonom Surus", frame)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("Kamera acilamadi.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame okunamadi.")
                break

            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]

            # --- Şerit algılama ---
            roi = frame[h // 2:, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            lane_mask = self.adaptive_lane_mask(v_channel)
            left_x, right_x = self.find_lane_points(lane_mask)
            lane_center, left_mem, right_mem = self.get_lane_center(left_x, right_x, w)

            # --- Kırmızı algılama ---
            red_detected, red_box = self.detect_red_light(frame)

            # --- Tabela algılama ---
            tabela, sign_box = self.detect_blue_sign(frame)

            # --- Karar verme ---
            if red_detected:
                direction = "DUR"
                error = self.last_error
            else:
                direction, error = self.decide_direction(lane_center, w, tabela)

            if self.DEBUG:
                print(
                    f"Yon: {direction} | Hata: {error} | "
                    f"Tabela: {tabela} | Kirmizi: {red_detected}"
                )

                self.draw_debug(
                    frame,
                    lane_mask,
                    lane_center,
                    left_mem,
                    right_mem,
                    direction,
                    error,
                    red_box,
                    sign_box,
                    tabela
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = OtonomSurusSistemi(camera_index=1, debug=True)
    sistem.run()