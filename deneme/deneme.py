import cv2
import numpy as np

# --- GLOBAL DEĞİŞKENLER (Robotun Hafızası) ---
kirmizi_sayaci = 0 

# --- ALGI KATMANI (Perception Layer) ---

def kirmizi_isik_analiz(frame, hsv_frame):
    """Kırmızı ışığı; renk, konum, yuvarlaklık, oran ve boyut testlerinden geçirir."""
    global kirmizi_sayaci
    
    # 1. Renk Süzgeci
    m1 = cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv_frame, np.array([160, 120, 70]), np.array([179, 255, 255]))
    maske = m1 + m2
    
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    isik_tespit_edildi = False
    
    for c in konturlar:
        alan = cv2.contourArea(c)
        cevre = cv2.arcLength(c, True)
        
        # A. Boyut Filtresi (Çok küçük gürültüleri ve aşırı büyük hataları eler)
        if 1000 < alan < (frame.shape[0] * frame.shape[1] // 4):
            if cevre > 0:
                # B. Yuvarlaklık Testi (Circularity)
                yuvarlaklik = (4 * np.pi * alan) / (cevre * cevre)
                
                if yuvarlaklik > 0.7:
                    xk, yk, wk, hk = cv2.boundingRect(c)
                    
                    # C. Oran Kontrolü (Aspect Ratio - Kareye yakın olmalı)
                    oran = float(wk) / hk
                    if 0.7 < oran < 1.3:
                        
                        # D. Konum Kontrolü (Ekranın üst kısmında mı?)
                        if yk < frame.shape[0] * 0.6:
                            
                            # E. Merkez Doluluk Testi (Hız tabelasından ayırmak için)
                            mx, my = xk + wk//2, yk + hk//2
                            merkez_bolge = maske[max(0, my-5):min(frame.shape[0], my+5), 
                                                 max(0, mx-5):min(frame.shape[1], mx+5)]
                            
                            if cv2.countNonZero(merkez_bolge) > 15:
                                isik_tespit_edildi = True
                                # Tespit edilen ışığı kare içine alalım
                                cv2.rectangle(frame, (xk, yk), (xk+wk, yk+hk), (0, 0, 255), 2)
                                break 

    # F. Güven Filtresi (Temporal Filtering - Arka arkaya 3 kare görme şartı)
    if isik_tespit_edildi:
        kirmizi_sayaci += 1
    else:
        kirmizi_sayaci = 0 
    
    return kirmizi_sayaci >= 3

def tabela_analiz(frame, hsv_frame):
    """Mavi tabelayı ve içindeki okun yönünü analiz eder."""
    alt_mavi = np.array([100, 150, 50])
    ust_mavi = np.array([140, 255, 255])
    maske = cv2.inRange(hsv_frame, alt_mavi, ust_mavi)
    
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in konturlar:
        alan = cv2.contourArea(c)
        if alan > 2500:
            cevre = cv2.arcLength(c, True)
            circ = (4 * np.pi * alan) / (cevre * cevre)
            if circ > 0.65: # Sadece dairesel nesneler
                tx, ty, tw, th = cv2.boundingRect(c)
                t_ici = maske[ty:ty+th, tx:tx+tw]
                sol = cv2.countNonZero(t_ici[:, 0:tw//2])
                sag = cv2.countNonZero(t_ici[:, tw//2:tw])
                cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 2)
                return "SOLA_DON" if sol > sag + 150 else "SAGA_DON"
    return "YOK"

def serit_takip_analiz(hsv_frame):
    """Yoldaki şeritleri bulur ve sapma (hata) miktarını hesaplar."""
    H, W = hsv_frame.shape[:2]
    roi = hsv_frame[int(H*0.75):H, 0:W]
    maske = cv2.inRange(roi, np.array([0,0,0]), np.array([180,255,40]))
    
    def merkez_bul(taraf):
        k, _ = cv2.findContours(taraf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if k:
            b = max(k, key=cv2.contourArea)
            if cv2.contourArea(b) > 500:
                M = cv2.moments(b)
                if M["m00"] > 0: return int(M["m10"]/M["m00"])
        return -1

    lcx = merkez_bul(maske[:, 0:W//2])
    rcx_ham = merkez_bul(maske[:, W//2:W])
    rcx = rcx_ham + W//2 if rcx_ham != -1 else W
    
    yol_o = (lcx + rcx) // 2 if (lcx != -1 and rcx != W) else W//2
    return yol_o - (W // 2), lcx, rcx, yol_o

# --- KARAR KATMANI (Decision Layer) ---

def karar_verici(isik_var, tabela_yon, serit_hatasi):
    """Tüm algılardan gelen verileri birleştirip tek bir emir üretir (Hiyerarşik)."""
    # 1. Öncelik: Güvenlik (Kırmızı Işık)
    if isik_var:
        return "DUR"
    
    # 2. Öncelik: Tabelalar (Manüel Yönlendirme)
    if tabela_yon != "YOK":
        return tabela_yon
    
    # 3. Öncelik: Otonom Şerit Takibi
    if serit_hatasi < -30: return "SOLA_KIR"
    if serit_hatasi > 30: return "SAGA_KIR"
    
    return "DUZ_GIT"

# --- UYGULAMA KATMANI (Main Loop) ---

kamera = cv2.VideoCapture(0)

while True:
    ret, kare = kamera.read()
    if not ret: break
    kare = cv2.flip(kare, 1) # Bilgisayar testi için ayna modu
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    H, W = kare.shape[:2]

    # ADIM 1: ALGI (Dünyayı gözlemle)
    isik_durumu = kirmizi_isik_analiz(kare, hsv)
    tabela_durumu = tabela_analiz(kare, hsv)
    hata, lx, rx, yo = serit_takip_analiz(hsv)

    # ADIM 2: KARAR (Ne yapacağına karar ver)
    emir = karar_verici(isik_durumu, tabela_durumu, hata)

    # ADIM 3: GÖRSELLEŞTİRME (Kullanıcı için çizimler)
    # Şerit noktaları
    if lx != -1: cv2.circle(kare, (lx, H-30), 8, (255,0,0), -1)
    if rx != W: cv2.circle(kare, (rx, H-30), 8, (0,0,255), -1)
    cv2.circle(kare, (yo, H-30), 12, (0,255,0), -1)
    
    # Emir ve Durum Yazısı
    renk = (0,0,255) if emir == "DUR" else (0,255,0)
    cv2.putText(kare, f"KOMUT: {emir}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, renk, 3)

    cv2.imshow("MEB Otonom Arac v4.2 - Pro", kare)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

kamera.release()
cv2.destroyAllWindows()