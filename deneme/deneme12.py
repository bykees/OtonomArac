import cv2
import numpy as np


class SeritTakip:
    def __init__(self, camera_index=0, debug=True):
        self.camera_index = camera_index
        self.DEBUG = debug

        # Raspberry Pi 3 için düşük çözünürlük
        self.WIDTH = 320
        self.HEIGHT = 240

        # Sadece alt kısmı işle
        self.ROI_START = 0.65  # alt %35

        # Otomatik threshold sınırları
        self.MIN_THRESH = 120
        self.MAX_THRESH = 230
        # beyaz yoğunluğunu azalt/arttır
        self.THRESH_OFFSET = 35

        # Histogram için minimum beyaz yoğunluğu
        self.MIN_WHITE_SUM = 18

        # Son bilinen şerit konumları
        self.last_left_x = None
        self.last_right_x = None
        self.last_lane_width = 120

        # Hata yumuşatma
        self.last_error = 0
        self.error_alpha = 0.35

        # Ani sıçrama filtresi
        self.MAX_JUMP = 45

    def compute_auto_threshold(self, roi_gray):
        mean_val = np.mean(roi_gray)
        max_val = np.max(roi_gray)

        # Ortam parlaklığına göre dinamik eşik
        thresh = int(mean_val + self.THRESH_OFFSET)

        # Eğer görüntü çok parlaksa, biraz daha yukarı çek
        if max_val > 245:
            thresh += 10

        thresh = max(self.MIN_THRESH, min(thresh, self.MAX_THRESH))
        return thresh

    def preprocess(self, frame):
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = blur.shape
        y_start = int(h * self.ROI_START)
        roi = blur[y_start:h, :]

        auto_thresh = self.compute_auto_threshold(roi)

        _, binary = cv2.threshold(roi, auto_thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return frame, binary, y_start, auto_thresh

    def find_lane_peaks(self, binary):
        h, w = binary.shape
        mid = w // 2

        histogram = np.sum(binary == 255, axis=0)

        left_hist = histogram[:mid]
        right_hist = histogram[mid:]

        left_x = None
        right_x = None

        if np.max(left_hist) > self.MIN_WHITE_SUM:
            left_x = int(np.argmax(left_hist))

        if np.max(right_hist) > self.MIN_WHITE_SUM:
            right_x = int(np.argmax(right_hist)) + mid

        return left_x, right_x, histogram

    def stabilize_positions(self, left_x, right_x):
        # Hafıza kullan
        if left_x is None and self.last_left_x is not None:
            left_x = self.last_left_x

        if right_x is None and self.last_right_x is not None:
            right_x = self.last_right_x

        # Tek şerit varsa diğerini tahmin et
        if left_x is not None and right_x is None:
            right_x = left_x + self.last_lane_width

        elif right_x is not None and left_x is None:
            left_x = right_x - self.last_lane_width

        # Şerit genişliği mantıklıysa güncelle
        if left_x is not None and right_x is not None:
            lane_width = right_x - left_x
            if 60 < lane_width < 220:
                self.last_lane_width = lane_width

        # Ani sıçrama filtresi
        if self.last_left_x is not None and left_x is not None:
            if abs(left_x - self.last_left_x) > self.MAX_JUMP:
                left_x = self.last_left_x

        if self.last_right_x is not None and right_x is not None:
            if abs(right_x - self.last_right_x) > self.MAX_JUMP:
                right_x = self.last_right_x

        # Hafıza güncelle
        if left_x is not None:
            self.last_left_x = left_x

        if right_x is not None:
            self.last_right_x = right_x

        return left_x, right_x

    def calculate_error(self, left_x, right_x):
        if left_x is None or right_x is None:
            return self.last_error, None

        lane_center = (left_x + right_x) // 2
        frame_center = self.WIDTH // 2
        raw_error = lane_center - frame_center

        smooth_error = int((1 - self.error_alpha) * self.last_error + self.error_alpha * raw_error)
        self.last_error = smooth_error

        return smooth_error, lane_center

    def draw_debug(self, frame, binary, y_start, left_x, right_x, lane_center, error, auto_thresh):
        debug_frame = frame.copy()
        h, w, _ = debug_frame.shape
        roi_h = binary.shape[0]

        # ROI çizgisi
        cv2.line(debug_frame, (0, y_start), (w, y_start), (0, 255, 255), 2)

        # Araç merkezi
        frame_center = w // 2
        cv2.line(debug_frame, (frame_center, y_start), (frame_center, h), (0, 0, 255), 2)

        # Şerit işaretleri
        if left_x is not None:
            cv2.circle(debug_frame, (left_x, y_start + roi_h // 2), 5, (255, 0, 0), -1)

        if right_x is not None:
            cv2.circle(debug_frame, (right_x, y_start + roi_h // 2), 5, (0, 255, 0), -1)

        # Şerit merkezi
        if lane_center is not None:
            cv2.line(debug_frame, (lane_center, y_start), (lane_center, h), (255, 255, 0), 2)

        cv2.putText(debug_frame, f"Hata: {error}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(debug_frame, f"Thresh: {auto_thresh}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Kamera", debug_frame)
        cv2.imshow("Binary ROI", binary)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        if not cap.isOpened():
            print("Kamera acilamadi.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kare okunamadi.")
                break

            frame, binary, y_start, auto_thresh = self.preprocess(frame)
            left_x, right_x, histogram = self.find_lane_peaks(binary)
            left_x, right_x = self.stabilize_positions(left_x, right_x)
            error, lane_center = self.calculate_error(left_x, right_x)

            print(f"Sol: {left_x}, Sag: {right_x}, Hata: {error}, Thresh: {auto_thresh}")

            if self.DEBUG:
                self.draw_debug(frame, binary, y_start, left_x, right_x, lane_center, error, auto_thresh)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = SeritTakip(camera_index=0, debug=True)
    sistem.run()