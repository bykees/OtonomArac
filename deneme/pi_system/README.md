Pi 3 odakli moduler otonom arac iskeleti.

Moduller:
- `config.py`: Ortak ayarlar
- `types.py`: Veri tipleri
- `camera.py`: Kamera akisi
- `lane_follow.py`: Hafif serit takibi
- `traffic_light.py`: Trafik isigi algilama
- `sign_detection.py`: Tabela algilama
- `parking.py`: Park davranisi
- `state_machine.py`: Gorev yonetimi
- `controller.py`: Ana birlestirici
- `main.py`: Calistirma noktasi

Amac:
- Tum CV modullerini ayni anda agir sekilde calistirmamak
- Gereken modulu gereken frekansta tetiklemek
- Raspberry Pi 3 uzerinde genisletilebilir bir temel saglamak
