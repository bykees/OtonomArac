import os

klasor = "C:/Users/Hüseyin Taylan/Desktop/OtonomArac/deneme/sign_trainer/samples/cikmaz-yol"  # klasör yolunu buraya yaz

dosyalar = os.listdir(klasor)

sayac = 1

for dosya in dosyalar:
    if dosya.endswith(".jpg") or dosya.endswith(".png"):
        yeni_isim = f"cikmaz_yol{sayac}.jpg"
        
        eski_yol = os.path.join(klasor, dosya)
        yeni_yol = os.path.join(klasor, yeni_isim)
        
        os.rename(eski_yol, yeni_yol)
        sayac += 1