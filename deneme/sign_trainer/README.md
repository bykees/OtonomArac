# Sign Trainer

Bu arac ana sistemi bozmadan kamera goruntusunden tabela ornegi toplamak icin kullanilir.

## Calistirma

```bash
cd /home/robot/Desktop/otonom_arac/sign_trainer
python3 trainer.py
```

## Tuslar

- `1`: KASISLI-YOL
- `2`: HEMZEMIN-GECIT
- `3`: YAYA-GECIDI
- `4`: PARK-LEVHASI
- `5`: SOLLAMA-SERBEST
- `6`: CIKMAZ-YOL
- `7`: STOP
- `8`: SOL
- `9`: SAG
- `0`: ILERI
- `s`: Mevcut tahmin etiketiyle kaydet
- `q` veya `Esc`: Cikis

Kayitlar `samples/<etiket>/` altina ROI ve maske olarak kaydedilir.
