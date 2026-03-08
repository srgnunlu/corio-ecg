# Oturum: Vast.ai CUDA Deployment & Performance Profiling
**Tarih:** 2026-03-08
**Süre:** ~2 saat
**Faz:** Phase 6 — Test Web UI (VPS deployment)

## Özet
Digitizer'a CUDA desteği eklendi ve Vast.ai RTX 3090 sunucusuna deploy edildi. Timing profiling ile asıl darbogazin SignalExtractor'in Python for-loop'u oldugu kesfedildi (236 sn). Cozum olarak probability map downscaling patch'i hazırlandi ama VPS'e henuz uygulanamadi (terminal formatting sorunu).

## Yapilan Isler
- [x] Digitizer'a akilli device secimi eklendi — CUDA > CPU, MPS skip (daec0e4)
- [x] VPS kurulum scripti olusturuldu — `scripts/setup_vps.sh` (daec0e4)
- [x] Gradio app'e `--share` flag'i eklendi (daec0e4)
- [x] `ray.tune` mock eklendi — agir bagimlilik yerine sahte modul (48f7a7d)
- [x] Cift upscaling sorunu bulundu ve duzeltildi — `resample_size=None` (dc81019)
- [x] Timing diagnostics aktif edildi — `enable_timing=True` (dc81019)
- [x] GitHub repo public yapildi (`gh repo edit --visibility public`)
- [x] Branch push edildi: `feature/test-ui-gradio` -> remote
- [ ] SignalExtractor downscaling patch'i VPS'e uygulanamadi (terminal formatting)

## Alinan Kararlar
| Karar | Secilen | Neden |
|-------|---------|-------|
| GPU | RTX 3090 ($0.169/saat) | 24GB VRAM yeterli, ucuz, test icin ideal |
| Template | PyTorch (Vast) | CUDA + PyTorch hazir geliyor |
| Device policy | CUDA varsa CUDA, MPS ise CPU | MPS resize bug'i CUDA'da yok |
| resample_size | None (devre disi) | Config'deki 3000 + bizim 1800 = cift upscale, 5000px gorsel |
| Kalite vs hiz | Kaliteyi dusurme | Upscaling kaldirilmamali, signal extraction optimizasyonu dogru cozum |

## Degistirilen/Olusturulan Dosyalar
```
src/pipeline/digitize.py    — CUDA device selection, ray mock, resample_size=None, timing
src/web/app.py              — --share flag for remote Gradio access
scripts/setup_vps.sh        — Vast.ai one-command setup script (NEW)
external/open-ecg-digitizer/src/model/inference_wrapper.py — signal extraction downscale patch (LOCAL ONLY, not on VPS yet)
```

## Karsilasilan Sorunlar
- **Sorun:** `ray.tune` import hatasi — digitizer'in utils.py'si ray kullanıyor
  **Cozum:** `_mock_ray_tune_if_missing()` fonksiyonu — ray yoksa sahte modul enjekte ediyor

- **Sorun:** Cift upscaling — bizim MIN_IMAGE_DIMENSION=1800 + wrapper'in resample_size=3000
  **Cozum:** `cfg.MODEL.KWARGS.resample_size = None` ile wrapper resampling devre disi

- **Sorun:** VPS terminal cok satirli Python/komut kopyalarken bosluk/satir kirma
  **Cozum:** COZULMEDI — heredoc ve python -c yaklasımlari da bozuldu

## Teknik Notlar — TIMING PROFILING SONUCLARI (KRITIK)
VPS RTX 3090, gorsel 3085x1800 px (upscaled from 1200x700):
```
Initial resampling       0.00 s  (resample_size=None, skip)
Segmentation (U-Net)     0.48 s  (CUDA, hizli!)
Perspective detection    7.96 s  (CPU)
Cropping                34.25 s  (CPU, darbogazlardan biri)
Pixel size search        2.71 s  (CPU)
Dewarping                0.00 s  (devre disi)
Signal extraction      236.02 s  (CPU, ASIL DARBOGAZ — %84)
TOPLAM                 281.44 s
```

**SignalExtractor darbogazinin nedeni:** `_trace_horizontal_path()` metodu `for x in range(1, W)` ile her sutunu tek tek tariyor. 3085 pixel genislik = 3084 iterasyon x bolge sayisi x 4 iterasyon. Tamamen CPU-bound, CUDA ile hizlandirilamaz.

**Cozum (hazir ama uygulanmadi):** Segmentation sonrasi probability map'i 1200px'e kucult, signal extraction'a kucuk map ver. Ayni sekilde cropping icin de. Local'de `inference_wrapper.py` patch'lendi.

## Vast.ai VPS Bilgileri
- Instance ID: 32551456, Host: 87485
- IP: 79.112.58.103
- GPU: RTX 3090, CUDA 13.1, PyTorch 2.10, Python 3.12
- Fiyat: $0.169/saat
- Gradio URL: https://XXXXX.gradio.live (her restart'ta degisir)
- SUNUCU ACIK KALDIKCA PARA YIYOR — isi bitince destroy et!

## Siradaki
- [ ] **ONCELIK 1:** inference_wrapper.py patch'ini VPS'e uygula (signal extraction + cropping downscale)
- [ ] Patch sonrasi timing'i karsilastir — hedef: 281 sn -> ~20-30 sn
- [ ] Eger hala yavas: MIN_IMAGE_DIMENSION'i 1200'e dusurmeyi dene (upscale kaldir)
- [ ] Lead detection 0/12 sorunu — CUDA'da da cozulmedi, arastir
- [ ] Basarili olursa: VPS setup scriptini guncelle (patch dahil)
- [ ] Sunucuyu kapat/destroy et (para yemesin)
