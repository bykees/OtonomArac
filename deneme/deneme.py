import cv2
import numpy as np


class OtonomSurusSistemi:
    def __init__(self, camera_index=0, debug=False):
        self.DEBUG = debug

        # --- Parametreler ---
        self.MAX_CONF = 5
        self.DEFAULT_LANE_WIDTH = 160
        self.LANE_MIN_AREA = 300
        self.RED_MIN_AREA = 1000
        self.BLUE_MIN_AREA = 1500
        self.TURN_ERROR_THRESHOLD = 25

        self.LANE_STD_COEF = 0.9

        self.RED_CONFIRM_FRAMES = 3
        self.SIGN_CONFIRM_FRAMES = 3

        self.LANE_SMOOTH_ALPHA = 0.3

        # --- Durum hafızası ---
        self.lane_memory = {
            "left": {"x": None, "confidence": 0},
            "right": {"x": None, "confidence": 0}
        }

        self.last_lane_width = self.DEFAULT_LANE_WIDTH
        self.kirmizi_sayaci = 0
        self.tabela_sayaci = 0
        self.son_tabela = "YOK"
        self.son_hata = 0

        # --- Kamera ---
        self.kamera = cv2.VideoCapture(camera_index)
        if not self.kamera.isOpened():
            raise RuntimeError("Kamera acilamadi")

        self.kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # --- 1. ŞERİT MASKESİ ---
    def adaptive_lane_mask(self, v_channel):
        h, w = v_channel.shape
        maske = np.zeros_like(v_channel)

        for y in range(0, h, 5):
            row = v_channel[y:y + 5]
            thresh = np.mean(row) - self.LANE_STD_COEF * np.std(row)       
            thresh = max(0, min(255, int(thresh)))
            maske[y:y + 5] = cv2.inRange(v_channel[y:y + 5], 0, thresh)

        return maske

    # --- 2. ŞERİT TAKİBİ ---
    def serit_takip(self, frame, hsv_frame):
        H, W = frame.shape[:2]

        roi_w_start = int(W * 0.1)
        roi_w_end = int(W * 0.9)
        roi_h_start = int(H * 0.7)

        roi_hsv = hsv_frame[roi_h_start:H, roi_w_start:roi_w_end]
        v = roi_hsv[:, :, 2]

        blur_v = cv2.GaussianBlur(v, (5, 5), 0)
        maske = self.adaptive_lane_mask(blur_v)

        kernel = np.ones((5, 5), np.uint8)
        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)

        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_left, best_right = None, None
        max_l_area, max_r_area = 0, 0

        for c in konturlar:
            area = cv2.contourArea(c)
            if area <= self.LANE_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2

            if cx < maske.shape[1] // 2:
                if area > max_l_area:
                    max_l_area = area
                    best_left = cx
            else:
                if area > max_r_area:
                    max_r_area = area
                    best_right = cx

        # Hafıza güncelleme
        for side, current_x in [("left", best_left), ("right", best_right)]:
            if current_x is not None:
                self.lane_memory[side]["x"] = current_x
                self.lane_memory[side]["confidence"] = min(
                    self.MAX_CONF,
                    self.lane_memory[side]["confidence"] + 2
                )
            else:
                self.lane_memory[side]["confidence"] = max(
                    0,
                    self.lane_memory[side]["confidence"] - 1
                )
                if self.lane_memory[side]["confidence"] == 0:
                    self.lane_memory[side]["x"] = None

        lx = self.lane_memory["left"]["x"]
        rx = self.lane_memory["right"]["x"]

        left_ok = self.lane_memory["left"]["confidence"] >= 2 and lx is not None
        right_ok = self.lane_memory["right"]["confidence"] >= 2 and rx is not None

        # Son geçerli şerit genişliğini koru
        if left_ok and right_ok:
            self.last_lane_width = abs(rx - lx)

        current_w = self.last_lane_width if self.last_lane_width is not None else self.DEFAULT_LANE_WIDTH

        # Yol merkezi hesabı
        if left_ok and right_ok:
            yol_merkez = (lx + rx) // 2
        elif left_ok:
            yol_merkez = lx + (current_w // 2)
        elif right_ok:
            yol_merkez = rx - (current_w // 2)
        else:
            yol_merkez = maske.shape[1] // 2

        hata = yol_merkez - (maske.shape[1] // 2)
        self.son_hata = int((1 - self.LANE_SMOOTH_ALPHA) * self.son_hata + self.LANE_SMOOTH_ALPHA * hata)

        return self.son_hata, lx, rx, yol_merkez, roi_w_start, roi_h_start, maske

    # --- 3. KIRMIZI IŞIK ANALİZİ ---
    def kirmizi_isik_analiz(self, frame, hsv_frame):
        H, W = frame.shape[:2]

        mask1 = cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_frame, np.array([160, 120, 70]), np.array([179, 255, 255]))
        m = mask1 + mask2

        konturlar, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tespit = False

        for c in konturlar:
            area = cv2.contourArea(c)
            if area <= self.RED_MIN_AREA:
                continue

            cevre = cv2.arcLength(c, True)
            if cevre <= 0:
                continue

            dairesellik = (4 * np.pi * area) / (cevre * cevre)
            if dairesellik <= 0.7:
                continue

            xk, yk, wk, hk = cv2.boundingRect(c)

            if not (0.7 < (float(wk) / max(hk, 1)) < 1.3):
                continue

            if yk >= H * 0.6:
                continue

            mx, my = xk + wk // 2, yk + hk // 2
            roi = m[max(0, my - 3):min(H, my + 3), max(0, mx - 3):min(W, mx + 3)]

            if cv2.countNonZero(roi) > 5:
                tespit = True
                break

        self.kirmizi_sayaci = self.kirmizi_sayaci + 1 if tespit else 0
        return self.kirmizi_sayaci >= self.RED_CONFIRM_FRAMES

    # --- 4. TABELA ANALİZİ ---
    def tabela_analiz(self, frame, hsv_frame):
        H, W = frame.shape[:2]

        maske_mavi = cv2.inRange(hsv_frame, np.array([90, 100, 50]), np.array([145, 255, 255]))
        konturlar, _ = cv2.findContours(maske_mavi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anlik_tabela = "YOK"

        for c in konturlar:
            area = cv2.contourArea(c)
            if area < self.BLUE_MIN_AREA:
                continue

            cevre = cv2.arcLength(c, True)
            if cevre == 0:
                continue

            dairesellik = (4 * np.pi * area) / (cevre * cevre)
            if dairesellik < 0.65:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 20:
                continue

            x1 = max(int(x - radius), 0)
            y1 = max(int(y - radius), 0)
            x2 = min(int(x + radius), W)
            y2 = min(int(y + radius), H)

            roi_hsv = hsv_frame[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                continue

            roi_hsv = cv2.resize(roi_hsv, (100, 100))

            circle_mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(circle_mask, (50, 50), 45, 255, -1)

            # Beyaz/gri ok alanı
            ok_mask = cv2.inRange(
                roi_hsv,
                np.array([0, 0, 150]),
                np.array([180, 85, 255])
            )
            ok_mask = cv2.bitwise_and(ok_mask, circle_mask)

            # Gürültü azalt
            kernel = np.ones((3, 3), np.uint8)
            ok_mask = cv2.morphologyEx(ok_mask, cv2.MORPH_OPEN, kernel)
            ok_mask = cv2.morphologyEx(ok_mask, cv2.MORPH_CLOSE, kernel)

            if self.DEBUG:
                cv2.imshow("Goz: Tabela Ok Maskesi", ok_mask)

            sol_yari = cv2.countNonZero(ok_mask[:, 0:45])
            sag_yari = cv2.countNonZero(ok_mask[:, 55:100])

            if sol_yari > sag_yari + 100:
                anlik_tabela = "SOLA_DON"
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                break
            elif sag_yari > sol_yari + 100:
                anlik_tabela = "SAGA_DON"
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                break

        # Ardışık doğrulama
        if anlik_tabela != "YOK":
            if anlik_tabela == self.son_tabela:
                self.tabela_sayaci += 1
            else:
                self.son_tabela = anlik_tabela
                self.tabela_sayaci = 1
        else:
            self.son_tabela = "YOK"
            self.tabela_sayaci = 0

        if self.tabela_sayaci >= self.SIGN_CONFIRM_FRAMES:
            return self.son_tabela

        return "YOK"

    # --- 5. KARAR MERKEZİ ---
    def karar_merkezi(self, isik, tabela, hata):
        if isik:
            return "DUR", (0, 0, 255)

        if tabela != "YOK":
            return tabela, (255, 255, 0)

        if hata < -self.TURN_ERROR_THRESHOLD:
            return "SOLA_KIR", (0, 255, 0)

        if hata > self.TURN_ERROR_THRESHOLD:
            return "SAGA_KIR", (0, 255, 0)

        return "DUZ_GIT", (0, 255, 0)

    # --- 6. ANA DÖNGÜ ---
    def calistir(self):
        while True:
            ret, kare = self.kamera.read()
            if not ret:
                print("Kameradan kare alinamadi.")
                break

            hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
            H, W = kare.shape[:2]

            isik_var = self.kirmizi_isik_analiz(kare, hsv)
            tabela_durum = self.tabela_analiz(kare, hsv)
            hata, lx, rx, yo, roi_x_start, roi_y_start, lane_mask = self.serit_takip(kare, hsv)
            emir, renk = self.karar_merkezi(isik_var, tabela_durum, hata)

            # Görselleştirme
            y_cizim = roi_y_start + int((H - roi_y_start) * 0.5)

            if lx is not None and self.lane_memory["left"]["confidence"] >= 2:
                cv2.circle(kare, (roi_x_start + lx, y_cizim), 10, (255, 0, 0), -1)

            if rx is not None and self.lane_memory["right"]["confidence"] >= 2:
                cv2.circle(kare, (roi_x_start + rx, y_cizim), 10, (0, 0, 255), -1)

            cv2.circle(kare, (roi_x_start + yo, y_cizim), 12, (0, 255, 0), -1)

            cv2.putText(kare, f"KOMUT: {emir}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)
            cv2.putText(kare, f"HATA: {hata}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(kare, f"KIRMIZI: {isik_var}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(kare, f"TABELA: {tabela_durum}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("ANA EKRAN", kare)

            if self.DEBUG:
                cv2.imshow("Serit Maskesi", lane_mask)

            tus = cv2.waitKey(1) & 0xFF
            if tus == ord('q'):
                break

        self.kapat()

    def kapat(self):
        self.kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = OtonomSurusSistemi(camera_index=0, debug=True)
    sistem.calistir()