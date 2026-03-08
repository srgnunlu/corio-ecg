# Oturum: Phase 2 VPS Deployment & Pearson NaN Fix
**Tarih:** 2026-03-08
**Sure:** ~3 saat (tahmin)
**Faz:** Phase 2 — Digitization Robustness

## Ozet
Phase 2 kodunu lokal 50 ornekle test ettik, ardindan Vast.ai VPS'e deploy ettik. VPS'te 7 iteratif fix-push-test dongusu yasandi (eksik dependency'ler, mock'lanmasi gereken paketler). Pearson NaN bug'i iki katmanda cozuldu (metrics.py + evaluate_roundtrip.py).

## Yapilan Isler
- [x] Lokal 50 ornek round-trip evaluation calistirildi (195158c)
- [x] VPS setup scripti olusturuldu (76a8d8d)
- [x] Kaggle download desteği eklendi (ace4938, 9ce6374)
- [x] VPS dependency fix'leri: qrcode, HandwrittenText mock, CreasesWrinkles mock, ray (18ea5f7..0fcf5cb)
- [x] Pearson NaN fix — metrics.py'de constant lead guard (195158c)
- [x] Pearson NaN fix — evaluate_roundtrip.py'de np.nanmean (3d8ee4c)
- [x] VPS'te kismi calistirma basarili (disk dolu nedeniyle 125/150 goruntu)
- [ ] VPS'te buyuk diskle tam calistirma (30-40 GB gerekli)

## Alinan Kararlar
| Karar | Secilen | Neden |
|-------|---------|-------|
| VPS disk boyutu | 30-40 GB minimum | ray+PyTorch+CUDA+PTB-XL = ~20 GB, 16 GB yetmedi |
| HandwrittenText/CreasesWrinkles | sys.modules mock | TF/spacy/imutils bagimliligini yuklemek yerine mock'lamak daha hafif |
| Pearson NaN stratejisi | Iki katmanli fix | metrics.py'de guard + evaluate'de nanmean = belt and suspenders |

## Degistirilen/Olusturulan Dosyalar
```
scripts/vps_setup_and_run.sh          — VPS one-shot setup & eval (yeni, sonra fix'lendi)
scripts/download_ptbxl.py             — Kaggle/PhysioNet dual source eklendi
src/utils/ecg_render.py               — HandwrittenText + CreasesWrinkles mock'lari
src/utils/metrics.py                  — Pearson NaN guard (std==0 ve isnan)
src/training/evaluate_roundtrip.py    — np.mean -> np.nanmean (Pearson icin)
tests/test_ecg_render.py              — HARD preset testleri guncellendi
```

## Karsilasilan Sorunlar
- **Sorun:** VPS'te her push'ta yeni bir eksik dependency cikti (qrcode, tensorflow, seaborn, imutils, ray)
  **Cozum:** Tum external repo'larin requirements'larini tek seferde VPS scriptine ekleme + kullanilmayan modulleri sys.modules ile mock'lama
- **Sorun:** Pearson correlation NaN — constant lead'lerde pearsonr tanimsiz
  **Cozum:** metrics.py'de std==0 kontrolu + evaluate_roundtrip.py'de np.nanmean
- **Sorun:** VPS disk dolu (16 GB) — ray paketi ~2-3 GB
  **Cozum:** Sonraki VPS'te 30-40 GB disk secilmeli

## Teknik Notlar
- **Lokal 50-ornek sonuclari:** Clean CosSim=0.9041, Moderate=0.9054, Hard=0.9069, Agreement ~%85.6
- **VPS kismi sonuclari:** Clean CosSim=0.9273 (14 rec), Moderate=0.9398 (5), Hard=0.9427 (5) — lokal ile tutarli
- **Digitization failure rate ~%44** clean/moderate'da — sadece hard'da 50/50 basarili. Muhtemelen DPI ayariyla ilgili (clean/moderate 200 DPI, hard 150 DPI). Arastirmak lazim.
- **ECG-Image-Kit -c flag cakismasi:** python -c ile crop -c ayni args listesinde, test'te son -c'yi almak gerekiyor
- **ray paketi:** Open-ECG-Digitizer'in src/utils.py'si `from ray.tune import Stopper` yapiyor ama biz ray kullanmiyoruz. Belki bu import'u da mock'layabiliriz disk tasarrufu icin.

## Siradaki
- [ ] **Digitization failure rate arastirmasi:** Neden clean/moderate'da %44 basarisiz? DPI mi, goruntu boyutu mu, digitizer config'i mi?
- [ ] **VPS buyuk diskle tam calistirma:** 30-40 GB disk, --max-samples 500+ ile
- [ ] **ray mock'lama olasiligi:** VPS'te ray yuklemeden Open-ECG-Digitizer calistirmak icin sys.modules mock denenebilir (~2-3 GB tasarruf)
- [ ] **Task 11 tamamlama:** Full PTB-XL test run sonuclari (2000 kayit)
- [ ] **Phase 2 branch'i main'e merge:** Tum testler gecince
