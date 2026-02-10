import cv2
import numpy as np

# --- GLOBAL HAFIZA (Robotun Belleği ve Sayaçları) ---
lane_memory = {
    "left":  {"x": None, "confidence": 0},
    "right": {"x": None, "confidence": 0}
}
MAX_CONF = 5
kirmizi_sayaci = 0
son_hata = 0

# --- 1. ALGI KATMANI: GELİŞMİŞ ŞERİT ANALİZİ (v4.5 - Lane ID & Adaptive Mask) ---

def adaptive_lane_mask(v_channel):
    """Satır bazlı, standart sapmalı adaptif eşikleme (Işık değişimlerine dirençli)."""
    h, w = v_channel.shape
    maske = np.zeros_like(v_channel)
    for y in range(0, h, 5):
        row = v_channel[y:y+5]
        # Bölgesel parlaklık analizi
        thresh = np.mean(row) - 1.0 * np.std(row)
        maske[y:y+5] = cv2.inRange(v_channel[y:y+5], 0, thresh)
    return maske

def serit_takip_v45(frame, hsv_frame):
    global lane_memory, son_hata
    H, W = frame.shape[:2]

    # ROI: Yola odaklan (Alt kısım)
    roi_w_start = int(W*0.1)
    roi_hsv = hsv_frame[int(H*0.7):H, roi_w_start:int(W*0.9)]
    v = roi_hsv[:,:,2]

    blur = cv2.GaussianBlur(v, (5,5), 0)
    maske = adaptive_lane_mask(blur)
    
    # --- ROBOTUN GÖZÜ 1: YOL MASKE ---
    cv2.imshow("Robot Gozu: Yol Maskesi", maske)

    kernel = np.ones((5,5), np.uint8)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in konturlar:
        if cv2.contourArea(c) > 400:
            x, y, w, h = cv2.boundingRect(c)
            candidates.append(x + w//2)

    # --- KİMLİK EŞLEŞTİRME (ID Tracking) ---
    for side in ["left", "right"]:
        lane_memory[side]["confidence"] = max(0, lane_memory[side]["confidence"] - 1)

    for cx in candidates:
        if lane_memory["left"]["x"] is None and lane_memory["right"]["x"] is None:
            side = "left" if cx < maske.shape[1] // 2 else "right"
        else:
            l_dist = abs(cx - lane_memory["left"]["x"]) if lane_memory["left"]["x"] is not None else 9999
            r_dist = abs(cx - lane_memory["right"]["x"]) if lane_memory["right"]["x"] is not None else 9999
            side = "left" if l_dist < r_dist else "right"

        lane_memory[side]["x"] = cx
        lane_memory[side]["confidence"] = min(MAX_CONF, lane_memory[side]["confidence"] + 2)

    lx, rx = lane_memory["left"]["x"], lane_memory["right"]["x"]
    current_w = abs(rx - lx) if (lx is not None and rx is not None) else 320
    
    if lane_memory["left"]["confidence"] >= 2 and lane_memory["right"]["confidence"] >= 2:
        yol_merkez = (lx + rx) // 2
    elif lane_memory["left"]["confidence"] >= 2:
        yol_merkez = lx + (current_w // 2)
    elif lane_memory["right"]["confidence"] >= 2:
        yol_merkez = rx - (current_w // 2)
    else:
        yol_merkez = maske.shape[1] // 2

    hata = yol_merkez - (maske.shape[1] // 2)
    son_hata = int(0.7 * son_hata + 0.3 * hata) 
    return son_hata, lx, rx, yol_merkez, roi_w_start

# --- 2. ALGI KATMANI: KIRMIZI IŞIK ANALİZİ (Gelişmiş Filtreli) ---

def kirmizi_isik_analiz(frame, hsv_frame):
    global kirmizi_sayaci
    H, W = frame.shape[:2]
    m1 = cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv_frame, np.array([160, 120, 70]), np.array([179, 255, 255]))
    maske = m1 + m2
    
    # --- ROBOTUN GÖZÜ 2: KIRMIZI MASKESİ ---
    cv2.imshow("Robot Gozu: Kirmizi Filtre", maske)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tespit = False
    for c in konturlar:
        alan = cv2.contourArea(c)
        if alan > 1000:
            cevre = cv2.arcLength(c, True)
            if cevre > 0 and (4*np.pi*alan)/(cevre*cevre) > 0.7:
                xk, yk, wk, hk = cv2.boundingRect(c)
                oran = float(wk) / hk
                if 0.7 < oran < 1.3 and yk < H*0.6:
                    # Doluluk Testi (Hız tabelasını eler)
                    mx, my = xk + wk//2, yk + hk//2
                    merkez_b = maske[max(0,my-5):min(H,my+5), max(0,mx-5):min(W,mx+5)]
                    if cv2.countNonZero(merkez_b) > 15:
                        tespit = True
                        cv2.rectangle(frame, (xk, yk), (xk+wk, yk+hk), (0, 0, 255), 2)
                        break
    if tespit: kirmizi_sayaci += 1
    else: kirmizi_sayaci = 0
    return kirmizi_sayaci >= 3

# --- 3. ALGI KATMANI: TABELA ---

def tabela_analiz(frame, hsv_frame):
    maske = cv2.inRange(hsv_frame, np.array([100, 150, 50]), np.array([140, 255, 255]))
    
    # --- ROBOTUN GÖZÜ 3: MAVİ MASKE ---
    cv2.imshow("Robot Gozu: Tabela Filtre", maske)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in konturlar:
        if cv2.contourArea(c) > 2500:
            tx, ty, tw, th = cv2.boundingRect(c)
            t_ici = maske[ty:ty+th, tx:tx+tw]
            sol, sag = cv2.countNonZero(t_ici[:, 0:tw//2]), cv2.countNonZero(t_ici[:, tw//2:tw])
            cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 2)
            return "SOLA_DON" if sol > sag + 150 else "SAGA_DON"
    return "YOK"

# --- 4. KARAR KATMANI ---

def karar_merkezi(isik, tabela, hata):
    if isik: return "DUR", (0, 0, 255)
    if tabela != "YOK": return tabela, (255, 255, 0)
    if hata < -30: return "SOLA_KIR", (0, 255, 0)
    if hata > 30: return "SAGA_KIR", (0, 255, 0)
    return "DUZ_GIT", (0, 255, 0)

# --- 5. ANA DÖNGÜ ---

kamera = cv2.VideoCapture(0)
while True:
    ret, kare = kamera.read()
    if not ret: break
    kare = cv2.flip(kare, 1)
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    H, W = kare.shape[:2]

    hata, lx, rx, yo, r_start = serit_takip_v45(kare, hsv)
    isik_var = kirmizi_isik_analiz(kare, hsv)
    tabela_durum = tabela_analiz(kare, hsv)
    emir, renk = karar_merkezi(isik_var, tabela_durum, hata)

    # Görselleştirme
    if lx is not None and lane_memory["left"]["confidence"] >= 2:
        cv2.circle(kare, (r_start + lx, int(H*0.85)), 10, (255, 0, 0), -1)
    if rx is not None and lane_memory["right"]["confidence"] >= 2:
        cv2.circle(kare, (r_start + rx, int(H*0.85)), 10, (0, 0, 255), -1)
    cv2.circle(kare, (r_start + yo, int(H*0.85)), 12, (0, 255, 0), -1)
    
    cv2.putText(kare, f"KOMUT: {emir}", (20, 50), 1, 2, renk, 3)
    cv2.imshow("MEB Otonom Arac v4.5 FULL", kare)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

kamera.release()
cv2.destroyAllWindows()