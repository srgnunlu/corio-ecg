# Oturum: Goldberger + Re-projection probları reddi → kalibrasyon ankoru pivotu
**Tarih:** 2026-06-16
**Süre:** ~4 saat (tahmin)
**Faz:** Phase 3 — Quality Gate hardening (fidelity abstention)

## Özet
Pertürbasyon pilotunun ardından iki yeni referans-bağımsız prob daha denendi
(Goldberger limb-lead tutarlılığı ve image-space re-projection fidelity) ve
**ikisi de reddedildi**. Profesyonel firmaların (PMcardio) ve SOTA'nın
(PhysioNet Challenge 2024, npj/Emory) yaklaşımı araştırıldı. Dört başarısız
probun ortak kök nedeni netleşti: kaçan false-accept'ler **yorum katmanında**
(grid-ölçek kalibrasyonu + lead ataması) hatalı ve bu katman hiçbir
öz-tutarlılık/öz-örtüşme kontrolüne görünmüyor. Sonraki yön (mutlak fiziksel
ankor = kalibrasyon pulse + lead etiketleri) için detaylı implementasyon planı
yazıldı. Implementasyon başka oturuma bırakıldı.

## Yapılan İşler
- [x] Goldberger limb-lead consistency feature + faz-toleranslı best-lag metrik (`2dfd53a`)
- [x] Goldberger pilot tune setinde koşturuldu → REDDEDİLDİ (margin -0.528)
- [x] Profesyonel/SOTA araştırması (PMcardio, PhysioNet 2024, npj/Emory, SQI literatürü)
- [x] External digitizer enstrümante edildi: `signal_probability` + `extraction_crop_x0` dışarı verildi (vendored patch)
- [x] Image-space re-projection fidelity feature (precision/recall/f1) (`49e86b4`)
- [x] Re-projection pilot tune setinde koşturuldu → REDDEDİLDİ (margin -0.150)
- [x] Kalibrasyon/atama ankoru için detaylı implementasyon planı (`9ccec9a`)
- [x] Memory güncellendi (2 yeni rejection memo + MEMORY.md)
- [x] 268 test geçiyor (254 → 268, +14)

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| Goldberger faz uyumu | Best-lag cross-correlation (±1.3s) | 3x4 kağıtta limb lead'ler farklı sütun=farklı zaman penceresi; sıfır-lag mükemmel kayıtta bile başarısız |
| Re-projection mimarisi | External'ı enstrümante et (kullanıcı seçimi) | Segmentasyon maskesi (en bağımsız referans) return'de yoktu; temiz/döngüsel-olmayan metrik için şart |
| Ink eşiği | 0.5 değil **0.25** | signal-prob haritasının 99.9 persentili ~0.56; 0.5 maskeyi yapay seyrek bırakıyor |
| Sonraki yön | Kalibrasyon-pulse / grid-ölçek + lead-atama ankoru | Dört prob da yorum-katmanı hatasını kaçırdı; tek çözüm mutlak fiziksel ankor |
| Implementasyon | Sonraki oturuma ertelendi | Kullanıcı önce plan istedi |

## Değiştirilen/Oluşturulan Dosyalar
```
src/utils/signal_clean.py              — goldberger_consistency + best-lag yardımcı (REJECTED, research-only)
tests/test_signal_clean.py             — 6 Goldberger testi
scripts/goldberger_pilot.py            — Goldberger tune pilotu
src/quality/reprojection.py            — re-projection fidelity metriği (REJECTED, research-only)
tests/test_reprojection.py             — 8 re-projection testi
scripts/reproject_one.py               — izole dijitalleştirme + fidelity worker'ı
scripts/reprojection_pilot.py          — re-projection tune pilotu
src/pipeline/digitize.py               — DigitizeInfo'ya signal_probability/raw_lines/crop_x0 + _to_numpy
vendor/patches/open-ecg-digitizer.patch — wrapper: signal_probability + extraction_crop_x0 + signal_extractor.crop_x0
docs/experiments/phase3-goldberger-consistency-v1.md   — Goldberger deney kaydı
docs/experiments/phase3-reprojection-fidelity-v1.md    — re-projection deney kaydı
docs/plans/phase3-calibration-anchor-plan.md           — SONRAKİ OTURUM İÇİN PLAN
results/pmcardio-holdout/tune/physiological-consistency/goldberger_pilot.json
results/pmcardio-holdout/tune/reprojection/reprojection_pilot.json
```

## Karşılaşılan Sorunlar
- **Sorun:** Re-projection precision faithful kontrolde bile düşük (0.35).
  **Çözüm:** Teşhis edildi — ink eşiği 0.5 çok yüksek; signal-prob haritası 0.56'yı nadiren aşıyor. Eşik 0.25'e çekilince precision ~1.0'a çıktı.
- **Sorun:** Re-projection precision tüm kayıtlarda ~1.0'a doygunlaştı.
  **Çözüm:** Beklenen davranış — extractor zaten prob haritasını takip ediyor; bu yüzden precision ayırt edici değil, recall asıl sinyal (o da fidelity ile korele çıkmadı → red).

## Teknik Notlar
- **Kök içgörü:** iphone/26 (true corr 0.515, en kötü) image-space'de mürekkebe kusursuz oturuyor (recall 0.813). Yani izi sadakatle çıkarılmış ama **kalibrasyon-ölçek ve/veya lead ataması** yanlış. Kalibrasyon hatası tanımı gereği kendiyle tutarlı → hiçbir referans-bağımsız kontrol göremez.
- **Grid px/mm:** `PixelSizeFinder` grid_prob otokorelasyonundan, 5mm grid varsayımıyla. Tek ölçek tahmini. Kalibrasyon pulse (1mV=10mm) bağımsız ikinci ankor; şu an HİÇ işlenmiyor.
- **Vendored patch akışı:** external düzenle → `git -C external/open-ecg-digitizer diff | sed 's/^ $//' > vendor/patches/open-ecg-digitizer.patch` → `git apply --reverse --check` ile doğrula. Boş context satırlarındaki trailing space temizlenir.
- **canonical_lines:** x-ekseni piksel kolonları, değerler mV (grid ölçeğiyle kalibre). raw_lines: piksel-Y, extraction_prob ile aynı (H,W); preprocess_lines crop_x0 kadar kolon kırpar.
- **mypy/ruff:** Proje mypy zaten temiz değil (external dinamik importlar, np.load no-any-return pattern'ı). Yeni dosyalar ruff-temiz, mypy'da yeni hata yok. Test E402 pre-existing (lint sadece src/scripts).

## Sıradaki
- [ ] **`docs/plans/phase3-calibration-anchor-plan.md` ile başla** — pick-up checklist orada
- [ ] **Phase 0 failure triage:** iphone/26 ve doogee/74 hatasını sınıflandır (ölçek mi, permütasyon mu) → Track A vs B önceliğini belirle. Veri zaten diskte (pmcardio_reference_fidelity.json per_lead).
- [ ] **Track A feasibility gate (KRİTİK):** 3-4 tune kaydında kalibrasyon pulse'ının extraction'da gerçekten var olup olmadığını incele. Yoksa Option B (image-space) veya descope.
- [ ] Track A: `src/quality/calibration_pulse.py` + worker + pilot + testler
- [ ] Track B: lead-label pozisyonlarını dışarı ver (patch) + `src/quality/lead_assignment_check.py`
- [ ] Başarılı olursa: feature contract + gate entegrasyonu, gate'i yeniden dondur, SONRA tek seferlik locked test
- [ ] 60-grup PMcardio test split KİLİTLİ kalmaya devam; push YAPILMADI
