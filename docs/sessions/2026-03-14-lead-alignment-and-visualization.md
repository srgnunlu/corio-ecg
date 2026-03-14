# Oturum: Lead Alignment ve Visualization İyileştirmeleri
**Tarih:** 2026-03-14
**Süre:** ~2 saat
**Faz:** Phase 2 (Test UI) — digitize kalite iyileştirmeleri

## Özet
ECGFounder'a gönderilen sinyallerde lead alignment sorunu çözüldü — paper ECG layout'larında her lead farklı zaman offset'inde duruyor, model ise tüm lead'lerin aynı zaman penceresinde olmasını bekliyor. İki farklı alignment stratejisi eklendi: model input için zero-padded shift, digitize çıkışı için tile-based shift. ECG visualization'da segment extraction da aktif bölge tespitine geçirildi.

## Yapılan İşler
- [x] `_align_leads_for_model()` eklendi — diagnose.py'de model'e gönderilmeden önce her lead'in aktif verisini sample 0'a kaydırır, kalan kısım sıfır (c273d88)
- [x] `_align_leads_to_origin()` eklendi — digitize.py'de shift + tile ile sinyal boyu doldurulur, sinus arrest false positive önlenir (c273d88)
- [x] `_extract_lead_segment()` yeniden yazıldı — sabit column offset yerine dinamik aktif bölge tespiti (c273d88)

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| Model input alignment | Zero-padded (tile yok) | Tile boundary'leri yapay discontinuity yaratıyor, ECGFounder bunu aritmiye benzetiyor |
| Digitize output alignment | Tile-based | Sıfır kuyruk sinus arrest gibi yorumlanıyor, tile fizyolojik olarak geçerli (kalp atışı tekrarlanır) |
| Visualization segment extraction | Aktif bölge tespiti | Sabit column offset digitize sonrası veri konumunu bulamıyordu |

## Değiştirilen/Oluşturulan Dosyalar
```
src/pipeline/diagnose.py  — _align_leads_for_model() eklendi, diagnose() entegrasyonu
src/pipeline/digitize.py  — _align_leads_to_origin() eklendi (tile-based)
src/web/ecg_plot.py       — _extract_lead_segment() yeniden yazıldı (aktif bölge tespiti)
```

## Teknik Notlar
- **İki farklı alignment stratejisi var ve bu kasıtlı:** Model input'ta zero-pad (tile discontinuity aritmiye benziyor), digitize output'ta tile (zero tail sinus arrest'e benziyor). Her birinin trade-off'u farklı.
- **Aktif bölge tespiti:** `np.abs(lead) > peak * threshold` ile lead'deki gerçek sinyal bölgesi bulunuyor. Threshold: model alignment'ta 0.01, visualization'da 0.02.
- **`memory_test.py`** — geçici debug dosyası, commit edilmedi, temizlenebilir.

## Sıradaki
- [ ] Phase 3 planlaması: ECGFounder fine-tuning (PTB-XL render → digitize → train)
- [ ] Lead assignment kalitesini artırmak: layout_hint verildiğinde pozisyon tabanlı kesin atama
- [ ] external/ değişikliklerini kalıcı hale getirmek (patch dosyası veya fork)
- [ ] CUDA VPS'te gerçek fotoğraflarla toplu kalite testi (n=50+)
- [ ] Gradio UI iyileştirmeleri: sinyal kalite skoru, güvenilirlik göstergesi
