import cv2
import numpy as np

# Kamerayı başlatıyoruz
kamera = cv2.VideoCapture(0)

while True:
    ret, kare = kamera.read()
    if not ret: break
    
    # Test kolaylığı için görüntüyü aynalıyoruz
    kare = cv2.flip(kare, 1)
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    y, x, _ = kare.shape
    merkez = x // 2

    # --- 1. GELİŞMİŞ KIRMIZI IŞIK VE TABELA ANALİZİ ---
    m_kirmizi1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    m_kirmizi2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([179, 255, 255]))
    maske_kirmizi = m_kirmizi1 + m_kirmizi2
    
    isik_durumu = "YOK"
    kirmizi_tabela_tipi = "YOK"

    konturlar_k, _ = cv2.findContours(maske_kirmizi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c_k in konturlar_k:
        alan_k = cv2.contourArea(c_k)
        if alan_k > 1500:
            cevre_k = cv2.arcLength(c_k, True)
            yaklasik = cv2.approxPolyDP(c_k, 0.04 * cevre_k, True)
            kenar_sayisi = len(yaklasik)
            xk, yk, wk, hk = cv2.boundingRect(c_k)
            
            # ÜÇGEN KONTROLÜ (Kenar sayısı 3 veya gürültüyle 4 ise)
            if 3 <= kenar_sayisi < 4:
                kirmizi_tabela_tipi = "UCGEN TABELA (DIKKAT)"
                cv2.rectangle(kare, (xk, yk), (xk+wk, yk+hk), (0, 165, 255), 2)

            elif kenar_sayisi == 4:
                kirmizi_tabela_tipi = "KIRMIZI KARE"
                cv2.rectangle(kare, (xk, yk), (xk+wk, yk+hk), (0, 165, 255), 2)
            
            # DAİRESEL ŞEKİLLER (Işık mı yoksa Halka Tabela mı?)
            else:
                circularity_k = (4 * np.pi * alan_k) / (cevre_k * cevre_k)
                if circularity_k > 0.7:
                    # MERKEZ DOLULUK TESTİ (Işık içi doludur, tabela içi boştur)
                    mx, my = xk + wk//2, yk + hk//2
                    # Merkezin 5x5 piksellik alanına bakıyoruz
                    merkez_bolge = maske_kirmizi[max(0, my-5):min(y, my+5), max(0, mx-5):min(x, mx+5)]
                    
                    if cv2.countNonZero(merkez_bolge) > 15:
                        isik_durumu = "KIRMIZI" # İçi dolu, bu bir trafik ışığı
                        cv2.rectangle(kare, (xk, yk), (xk+wk, yk+hk), (0, 0, 255), 3)
                    else:
                        kirmizi_tabela_tipi = "YASAK / HIZ TABELASI" # İçi boş halka
                        cv2.rectangle(kare, (xk, yk), (xk+wk, yk+hk), (255, 0, 255), 2)

    # --- 2. DİĞER IŞIKLAR (SARI VE YEŞİL) ---
    maske_sari = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    maske_yesil = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
    
    if isik_durumu == "YOK":
        if np.sum(maske_sari) > 1000000: isik_durumu = "SARI"
        elif np.sum(maske_yesil) > 1000000: isik_durumu = "YESIL"

    # --- 3. MAVİ TABELA ALGILAMA (DAİRE TESTLİ) ---
    m_mavi = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))
    konturlar_m, _ = cv2.findContours(m_mavi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tabela_karari = "YOK"
    for c_m in konturlar_m:
        alan_m = cv2.contourArea(c_m)
        cevre_m = cv2.arcLength(c_m, True)
        if alan_m > 2000 and cevre_m > 0:
            circ_m = (4 * np.pi * alan_m) / (cevre_m * cevre_m)
            if circ_m > 0.65:
                tx, ty, tw, th = cv2.boundingRect(c_m)
                t_ici = m_mavi[ty:ty+th, tx:tx+tw]
                sol_p = cv2.countNonZero(t_ici[:, 0:tw//2])
                sag_p = cv2.countNonZero(t_ici[:, tw//2:tw])
                if sol_p > sag_p + 150: tabela_karari = "SOLA DON"
                elif sag_p > sol_p + 150: tabela_karari = "SAGA DON"
                cv2.rectangle(kare, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 2)

    # --- 4. ŞERİT TAKİBİ ---
    roi = hsv[int(y*0.75):y, 0:x]
    m_siyah = cv2.inRange(roi, np.array([0,0,0]), np.array([180,255,40]))
    sol_t = m_siyah[:, 0:merkez]; sag_t = m_siyah[:, merkez:x]
    lcx, rcx = -1, -1

    def merkez_bul(taraf):
        k, _ = cv2.findContours(taraf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if k:
            en_b = max(k, key=cv2.contourArea)
            if cv2.contourArea(en_b) > 500:
                M = cv2.moments(en_b)
                if M["m00"] > 0: return int(M["m10"]/M["m00"])
        return -1

    lcx = merkez_bul(sol_t)
    rcx_h = merkez_bul(sag_t)
    if rcx_h != -1: rcx = rcx_h + merkez

    if lcx != -1 and rcx != -1: yol_o = (lcx + rcx) // 2
    elif lcx != -1: yol_o = lcx + 100
    elif rcx != -1: yol_o = rcx - 100
    else: yol_o = merkez
    hata = yol_o - merkez

    # --- 5. GÖRSELLEŞTİRME VE KARAR ---
    if lcx != -1: cv2.circle(kare, (lcx, y-30), 10, (255, 0, 0), -1) # Sol (Mavi)
    if rcx != -1: cv2.circle(kare, (rcx, y-30), 10, (0, 0, 255), -1) # Sağ (Kırmızı)
    cv2.circle(kare, (yol_o, y-30), 12, (0, 255, 0), -1)           # Hedef (Yeşil)

    if isik_durumu == "KIRMIZI":
        mesaj, renk = "DUR: KIRMIZI ISIK!", (0,0,255)
    elif isik_durumu == "SARI":
        mesaj, renk = "YAVASLA: SARI ISIK", (0,255,255)
    elif isik_durumu == "YESIL":
        mesaj, renk = "YOL ACIK: YESIL ISIK", (0,255,0)
    elif kirmizi_tabela_tipi != "YOK":
        mesaj, renk = kirmizi_tabela_tipi, (0,165,255)
    elif tabela_karari != "YOK":
        mesaj, renk = f"TABELA: {tabela_karari}", (255, 255, 0)
    elif lcx == -1 and rcx == -1:
        mesaj, renk = "YOL KAYIP!", (255, 255, 255)
    elif hata < -30: mesaj, renk = "SOLA KIR", (0, 255, 0)
    elif hata > 30: mesaj, renk = "SAGA KIR", (0, 255, 0)
    else: mesaj, renk = "DUZ GIT", (0, 255, 0)

    cv2.putText(kare, mesaj, (20, 45), 1, 2, renk, 2)
    cv2.imshow("MEB Otonom Arac v3.2 - Akilli Sistem", kare)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

kamera.release()
cv2.destroyAllWindows()