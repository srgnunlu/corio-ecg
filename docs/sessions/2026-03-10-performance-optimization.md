# Oturum: 22x Performance Optimization + Layout Hint UI
**Tarih:** 2026-03-10
**Sure:** ~1 saat
**Faz:** Phase 6 — Test Web UI (performance + quality)

## Ozet
Digitizer pipeline'i 22x hizlandirildi (281s → 13s). Uc katmanli downscaling stratejisi uygulanarak segmentation, cropping ve signal extraction darbogazlari cozduldu. Gradio UI'a layout secenegi ve timing breakdown eklendi. Gercek ECG fotograflariyla test edildi.

## Yapilan Isler
- [x] Segmentation downscale (CPU: 2000px cap) — 54s → 10s
- [x] Cropping downscale (1200px) — 34s → 0.1s (onceki oturumda hazir, test edildi)
- [x] Signal extraction downscale (1200px) — 236s → 0.3s (onceki oturumda hazir, test edildi)
- [x] Layout hint parametresi eklendi (`digitize(layout_hint="3x4")`)
- [x] Gradio'ya layout dropdown eklendi (Auto/3x4/3x4+1R/6x2)
- [x] Gradio'ya timing breakdown eklendi (debug panel)
- [x] Gradio'ya toplam sure (end-to-end) eklendi
- [x] Gradio'ya ornek goruntuler eklendi (clean/moderate/hard)
- [x] Gercek ECG fotograflariyla test yapildi
- [x] VPS kapatildi (para yemesin)

## Alinan Kararlar
| Karar | Secilen | Neden |
|-------|---------|-------|
| Seg downscale boyutu | 2000px | 1500px kaliteyi bozuyordu (gercek foto), 2000px iyi denge |
| Layout default | Auto-detect | Cogu ECG 3x4+1R ama diger formatlar da var |
| CUDA seg downscale | Hayir (full res) | CUDA zaten 0.48s, downscale gereksiz |

## Performans Karsilastirmasi
| Asama | Onceki (VPS, patch'siz) | Simdi (Mac CPU) | Iyilesme |
|-------|------------------------|-----------------|----------|
| Segmentation | 54s (CPU) / 0.48s (CUDA) | 10s (CPU) | 5.4x |
| Cropping | 34s | 0.1s | 340x |
| Signal extraction | 236s | 0.3s | 787x |
| **Toplam inference** | **281s** | **13s** | **22x** |

## Gercek Foto Test Sonuclari
- Layout hint (3x4+1R) ile kalite onemli olcude iyilesiyor (cost: 0.41 vs 1.18)
- Bazi derivasyonlarda dalga distorsiyonu devam ediyor (ozellikle V2)
- Lead I hic algilanmiyor (digitizer'da disabled)
- Sentetik vs gercek foto arasinda kalite farki var — fine-tuning gerekli

## Degistirilen Dosyalar
```
src/pipeline/digitize.py  — layout_hint param, CPU seg downscale (2000px)
src/web/app.py            — layout dropdown, timing breakdown, sample images
external/.../inference_wrapper.py — seg downscale, crop downscale, extraction downscale (gitignored)
```

## Siradaki
- [ ] Digitizer kalite iyilestirme (gercek fotograflar icin)
- [ ] Lead I detection sorununu arastir
- [ ] V2 dusuk enerji sorununu arastir
- [ ] Phase 5: LLM report generation
- [ ] inference_wrapper.py patch'lerini kalici hale getir (fork veya patch file)
