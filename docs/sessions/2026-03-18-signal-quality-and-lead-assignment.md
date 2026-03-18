# Oturum: Sinyal Kalitesi İyileştirme + Lead Assignment Araştırması
**Tarih:** 2026-03-18
**Süre:** ~3 saat
**Faz:** Phase 2 (Test UI) — sinyal kalitesi + digitize pipeline iyileştirmeleri

## Özet
Visualization alignment fix'i tamamlandı (3x4+1R yarı boş hücre sorunu çözüldü). Bandpass filtre 0.5–40 Hz yerine highpass 0.5 Hz + wavelet denoising (db4) getirildi — ECGFounder'ın eğitim sırasında görmediği 40 Hz cutoff kaldırıldı. HR estimation'a bandpass pre-filter eklendi. Position-based lead assignment denenip geri alındı (raw_lines piksel koordinatı, canonical µV — birim uyumsuzluğu).

## Yapılan İşler
- [x] `_align_leads_to_origin()` çağrısı `_postprocess()`'e eklendi — 3x4 yarı boş hücre sorunu çözüldü (9692679)
- [x] `_extract_lead_segment()` basit column-based extraction'a döndürüldü (9692679)
- [x] Bandpass 0.5–40 Hz → highpass 0.5 Hz + wavelet denoising (db4, level 2) (5c86efb)
- [x] `src/utils/signal_clean.py` oluşturuldu: highpass, wavelet_denoise, einthoven_consistency (5c86efb)
- [x] PyWavelets bağımlılığı eklendi (5c86efb)
- [x] Einthoven tutarlılık kontrolü DigitizeInfo'ya ve Gradio debug paneline eklendi (5c86efb)
- [x] HR estimation'a 5–30 Hz bandpass pre-filter eklendi (8e23e61)
- [x] Wavelet noise_levels 3→2 düşürüldü — level 3 (31–62 Hz) QRS slope içeriyor (8e23e61)
- [x] Position-based lead assignment denendi ve geri alındı (44aab46 → 171dff3)
- [x] raw_lines vs canonical_lines format farkı debug edildi ve belgelendi

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| Bandpass yerine ne | Highpass + wavelet | ECGFounder filtresiz WFDB ile eğitildi; 40 Hz QRS morfolojisini bozuyordu |
| Wavelet noise_levels | 2 (62.5+ Hz) | Level 3 (31–62 Hz) QRS slope içeriği taşıyor, threshold'lamak HR estimation'ı bozdu |
| HR pre-filter | 5–30 Hz bandpass (fonksiyon içi) | HR estimation'ı upstream processing'den bağımsız kılar |
| Lead assignment override | GERİ ALINDI | raw_lines piksel koordinatı, canonical µV — doğrudan yer değiştirme birim bozukluğu yaratıyor |
| Tile vs mirror-reflect | Tile kalacak | Bandpass/wavelet tile sınırlarını zaten yumuşatıyor, yeterli |

## Değiştirilen/Oluşturulan Dosyalar
```
src/utils/signal_clean.py       — YENİ: highpass_filter, wavelet_denoise, einthoven_consistency
src/pipeline/lead_assignment.py — YENİ: position-based assignment (şu an kullanılmıyor, gelecek için)
src/pipeline/digitize.py        — alignment aktif, bandpass→highpass+wavelet, Einthoven entegre
src/utils/rhythm.py             — HR estimation'a 5–30 Hz bandpass pre-filter eklendi
src/web/ecg_plot.py             — _extract_lead_segment basitleştirildi (column-based)
src/web/app.py                  — Einthoven skoru debug panele eklendi
src/utils/__init__.py           — signal_clean export'ları eklendi
pyproject.toml                  — PyWavelets>=1.4.0 eklendi
tests/test_signal_clean.py      — YENİ: 12 test (highpass, wavelet, einthoven)
```

## Karşılaşılan Sorunlar
- **Sorun:** Wavelet noise_levels=3 HR estimation'ı bozdu (160 bpm → 27 bpm)
  **Çözüm:** Level 3 (31–62 Hz) QRS slope içeriği taşıyor, noise_levels=2'ye düşürüldü + HR'a kendi bandpass pre-filter eklendi

- **Sorun:** Position-based lead assignment digitizer çıkışını bozdu
  **Çözüm:** raw_lines piksel koordinatı (Y pixel), canonical_lines µV — birim farkı. Override geri alındı.

- **Sorun:** Gradio restart'ta eski kod cache'den çalışıyor
  **Çözüm:** `pkill -9 -f "src.web.app"` ile tüm process'leri öldürüp yeniden başlatmak gerekli

## Teknik Notlar
- **raw_lines format:** shape=(n_traces, n_pixel_samples), dtype=float32, değerler piksel Y-koordinatı
- **canonical_lines format:** shape=(12, 10000), dtype=float32, değerler µV
- **canonical boyutu 10000 (5000 DEĞİL!)** — _postprocess 5000'e resample ediyor
- **Digitizer kolon ataması doğru:** 3x4'te col0=[I,II,III], col1=[aVR,aVL,aVF], col2=[V1-3], col3=[V4-6]
- **Satır ataması sorunlu olabilir:** Aynı kolondaki lead'lerin sırası (üst/orta/alt) yanlış olabilir
- **Einthoven skoru 3x4'te yanıltıcı:** Lead II rhythm strip'ten (10s), I+III col0'dan (2.5s tile) → farklı zaman pencereleri → düşük korelasyon
- **Lead assignment doğru çözüm:** canonical_lines reorder (aynı birimde), raw_lines'la cross-correlation ile satır sırası belirle

## Sıradaki
- [ ] Lead assignment düzeltmesi: canonical_lines üzerinde reorder (raw_lines ile cross-correlation kullanarak satır sıralaması belirle, piksel→µV dönüşümü YAPMA)
- [ ] Phase 3 planlaması: ECGFounder fine-tuning (PTB-XL render → digitize → train)
- [ ] Einthoven kontrolünü 3x4 rhythm strip durumu için ayarla (farklı zaman pencereleri → düşük skor beklenen)
- [ ] `src/pipeline/lead_assignment.py` canonical reorder yaklaşımıyla güncelle
- [ ] Gradio'yu push et: `git push origin feature/test-ui-gradio`
