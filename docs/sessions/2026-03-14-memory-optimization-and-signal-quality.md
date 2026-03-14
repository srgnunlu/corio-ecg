# Oturum: Bellek Optimizasyonu, MPS Hybrid ve Sinyal Kalitesi
**Tarih:** 2026-03-14
**Süre:** ~4 saat
**Faz:** Phase 2 (Test UI) + Phase 3 hazırlık

## Özet
Mac Mini M4 (24GB) üzerinde 35+ GB bellek kullanımı sorunu çözüldü — MPS/CPU hybrid yaklaşımıyla peak 8 GB'a düşürüldü. ECG visualization 6x2+1R formatına geçirildi. HR estimation Pan-Tompkins autocorrelation ile yeniden yazıldı (219→72 bpm düzeltmesi). Bandpass filter eklendi. Tanı kalitesinin kök nedeni analiz edildi: ECGFounder domain gap (Phase 3 gerekli).

## Yapılan İşler
- [x] Bellek kullanımı analizi ve root cause tespiti (69b57b8)
- [x] MPS hybrid yaklaşımı: segmentation U-Net → Metal GPU, pipeline → CPU (69b57b8)
- [x] Portrait fotoğraf auto-rotation eklendi (69b57b8)
- [x] Upscale limiti (max 2x) eklendi (69b57b8)
- [x] MAX_IMAGE_DIMENSION = 2400 eklendi (69b57b8)
- [x] Dewarping retry varsayılan olarak kapatıldı (69b57b8)
- [x] Rhythm strip lead assignment düzeltildi: "Any" → "II" (69b57b8)
- [x] Layout config'te V3 typo düzeltildi (69b57b8)
- [x] inference_wrapper.py'de tensor memory optimizasyonu (69b57b8)
- [x] ECG visualization 6x2+1R formatına geçirildi (a5acb2c)
- [x] Bandpass filter (0.5-40 Hz) eklendi — grid artifact temizliği (856d69a)
- [x] HR estimation: Pan-Tompkins autocorrelation yöntemiyle yeniden yazıldı (856d69a)
- [x] Tanı kalitesi root cause analizi yapıldı

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| MPS kullanımı | Sadece segmentation U-Net | perspective_detector ve layout_identifier'da MPS tensor indexleme bug'ları var |
| MIN_IMAGE_DIMENSION | 1800 (korundu) | 1700px'de grid detection ve signal separation başarısız (test edildi: 2 line vs 4 line) |
| Dewarping retry | Varsayılan kapalı | Pipeline'ı 2x çalıştırıyor, peak memory'yi 2x'e çıkarıyor |
| Visualization formatı | 6x2+1R | Türkiye klinik standardı, PMCardio ile uyumlu, her lead 5 saniye |
| Rhythm strip lead mapping | "II" (sabit) | "Any" cosine similarity matching aVF'ye eşliyordu (corr=0.49) |
| HR estimation yöntemi | Autocorrelation (diff²) | Peak detection z-score sonrası T-dalgalarını QRS olarak sayıyordu |
| Bandpass filter | 0.5-40 Hz, Butterworth 3 | Grid artifact'ları (>40 Hz) ve baseline drift (<0.5 Hz) temizliği |

## Değiştirilen/Oluşturulan Dosyalar
```
src/pipeline/digitize.py          — MPS hybrid, MAX_IMAGE_DIMENSION, auto-rotate, upscale cap, bandpass filter, dewarping retry, tensor cleanup
src/pipeline/diagnose.py          — HR estimation integration, rate-consistency adjustments
src/web/app.py                    — Dewarping retry default=False, MPS cache clear
src/web/ecg_plot.py               — 3x4+1R → 6x2+1R layout değişikliği
src/utils/rhythm.py               — Pan-Tompkins autocorrelation HR estimation (yeni yazıldı)
tests/test_digitize.py            — Yeni testler (quality guards, retry logic, image sizing)
tests/test_rhythm.py              — HR estimation testleri
tests/test_web_app.py             — Trim payload ve plot testleri
external/.../inference_wrapper.py — MPS seg device, tensor del, identifier cap (gitignored)
external/.../lead_identifier.py   — Specified vs wildcard rhythm lead logic (gitignored)
external/.../lead_layouts_*.yml   — rhythm_leads: "Any" → "II", V3 typo fix (gitignored)
```

## Karşılaşılan Sorunlar
- **Sorun:** 35+ GB bellek kullanımı → macOS restart
  **Çözüm:** PyTorch CPU conv2d im2col temporary matrisler nedeniyle. Segmentation U-Net'i MPS'e taşıyarak Metal allocator'ın freed memory'yi OS'e geri vermesi sağlandı. Peak: 17 GB → 8 GB.

- **Sorun:** MPS'e tam geçiş — perspective_detector crash
  **Çözüm:** MPS'te tensor indexleme bug'ı (boundary index out of bounds). Hybrid yaklaşım: sadece segmentation MPS'te, geri kalan CPU'da.

- **Sorun:** Portrait fotoğraflar tamamen başarısız
  **Çözüm:** Auto-rotation: h > w ise 90° döndür. ECG kağıtları her zaman landscape.

- **Sorun:** HR estimation 75 bpm → 219 bpm
  **Çözüm:** Z-score normalizasyon T-dalgalarını QRS kadar büyütüyor → peak detection hepsini sayıyor. diff(signal)² → autocorrelation yöntemiyle T-dalgaları eleniyor.

- **Sorun:** Rhythm strip yarıda kalıyor (6x2+1R layout)
  **Çözüm:** rhythm_leads: "Any" cosine similarity matching yanlış lead'e eşliyordu. "II" sabit atama ile çözüldü.

## Teknik Notlar
- **PyTorch CPU conv2d** im2col temporary matris boyutu: (H×W) × (C×k×k). 1800px görüntüde tek conv ~1.7 GB. macOS malloc freed sayfaları OS'e geri vermiyor → RSS sürekli artıyor.
- **Apple Metal allocator** freed memory'yi OS'e düzgün geri veriyor → MPS kullanımı RSS şişmesini önlüyor.
- **MPS tensor indexleme bug:** `accumulator[y_coords.long(), x_coords_clamped]` — boundary index (20492000) MPS'te out of bounds oluyor, CPU'da çalışıyor. PyTorch issue.
- **Upscaling zorunluluğu:** MIN_IMAGE_DIMENSION=1800 olmadan digitizer çalışmıyor (2 line at 1700px vs 4 line at 1800px). Model bu çözünürlüğe bağımlı.
- **inference_wrapper.py değişiklikleri external/ (gitignored)** — yeni clone/VPS'te tekrar uygulanmalı.
- **ECGFounder tanı kalitesi** digitize sinyallerde düşük — model temiz WFDB ile eğitildi, domain gap var. Phase 3 fine-tuning gerekli.
- **Lead Name U-Net** düşük çözünürlükte (cap'li) text etiketleri okuyamıyor → lead assignment hataları → ECGFounder yanlış tanı.

## Sıradaki
- [ ] Phase 3 planlaması: ECGFounder fine-tuning stratejisi (PTB-XL render → digitize → train)
- [ ] Lead assignment kalitesini artırmak: layout_hint verildiğinde pozisyon tabanlı kesin atama
- [ ] external/ değişikliklerini kalıcı hale getirmek (patch dosyası veya fork)
- [ ] CUDA VPS'te gerçek fotoğraflarla toplu kalite testi (n=50+)
- [ ] Gradio UI iyileştirmeleri: sinyal kalite skoru, güvenilirlik göstergesi
