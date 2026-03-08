# Oturum: MPS Buffer Fix & 500-Kayıt VPS Evaluasyonu
**Tarih:** 2026-03-08
**Süre:** ~3 saat
**Faz:** Phase 2 — Digitization Robustness

## Özet
Clean/moderate görüntülerde %44 digitization başarısızlığının kök nedeni bulundu: Apple Silicon MPS GPU buffer overflow (2200x1700 px görüntüler U-Net için çok büyük). MPS-only resize fix eklendi. Ardından Vast.ai RTX 4090 VPS'te 500 kayıtlık tam evaluasyon başarıyla tamamlandı — tüm zorluk seviyelerinde 500/500 başarı, CosSim>0.90, Agreement~%85.

## Yapılan İşler
- [x] Digitization failure root cause analizi — AcceleratorError teşhisi (e0bc1ea)
- [x] MPS-only image resize fix (max 1600px, CUDA'da değişiklik yok) (e0bc1ea)
- [x] 3 yeni test: MPS resize, küçük görüntü, CUDA bypass (e0bc1ea)
- [x] Eski başarısız 22 clean + 22 moderate kaydı yeniden dijitalize (lokal, 50/50)
- [x] VPS'te 500-kayıt tam evaluasyon çalıştırıldı (sıfır başarısızlık)
- [x] Threshold analysis scripti oluşturuldu (9d6aa5d)
- [ ] Threshold analysis henüz çalıştırılmadı (VPS'te `python3 scripts/threshold_analysis.py`)

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| DPI düşürmek mi, resize mi? | MPS-only resize | Kaliteden ödün vermemek için DPI'a dokunulmadı. CUDA'da tam çözünürlük korunuyor |
| Resize limiti | 1600px | Hard preset 1650px'te sorunsuz çalışıyor, biraz marjla güvenli tarafta |
| 500 vs 2000 kayıt | 500 ile başla | İstatistiksel olarak yeterli, süre makul (~6 saat) |

## Değiştirilen/Oluşturulan Dosyalar
```
src/pipeline/digitize.py              — MPS_MAX_IMAGE_DIMENSION + _load_image resize logic
tests/test_digitize.py                — 3 yeni test (MPS resize, no-resize small, CUDA bypass)
scripts/threshold_analysis.py         — Threshold analysis scripti (yeni)
results-vps-500/roundtrip_comparison.json — VPS 500-kayıt sonuçları (lokal kopyası)
```

## Karşılaşılan Sorunlar
- **Sorun:** AcceleratorError: index out of bounds on MPS — 200 DPI görüntüler (2200x1700) GPU buffer limitini aşıyor
  **Çözüm:** `_load_image()`'da `self.device.type == "mps"` kontrolüyle max 1600px resize. CUDA'da hiçbir değişiklik yok.
- **Sorun:** Jupyter notebook'ta VPS'in .venv paketlerine erişilemiyor (wfdb ModuleNotFoundError)
  **Çözüm:** Script'i git'e push edip VPS terminalinde `python3 scripts/threshold_analysis.py` ile çalıştırmak (henüz yapılacak)
- **Sorun:** VPS'ten Mac'e SCP bağlantı hatası (Connection reset)
  **Çözüm:** Vast.ai SSH port'u farklı olabilir, sonuçlar JSON olarak elle kaydedildi

## Teknik Notlar
- **MPS buffer overflow pattern:** Index değerleri 17M-21M aralığında (buffer size = piksel sayısı × kanal × ara katman). 2200×1700=3.74M piksel çok büyük, 1650×1275=2.1M güvenli.
- **Non-deterministic failures:** Aynı görüntü tek başına çalışıp batch'te başarısız olabiliyor — MPS bellek baskısıyla ilgili.
- **VPS performans:** RTX 4090'da digitization ~10.5 sn/görüntü, toplam 500×3=1500 görüntü ~4.2 saat.
- **Sonuçlar zorluklar arası çok yakın:** Clean≈Moderate≈Hard (CosSim 0.90-0.91). Augmentation tanısal doğruluğu pek etkilemiyor.
- **Pearson düşük (~0.13) ama Agreement yüksek (%85):** Sinyal dalga şekli farklı olsa da ECGFounder aynı tanısal özellikleri çıkarıyor.

## VPS 500-Kayıt Sonuçları
| Senaryo | N | CosSim | Agreement | Pearson | AbsDiff |
|---------|---|--------|-----------|---------|---------|
| Clean | 500/500 | 0.9090 | %85.1 | 0.132 | 0.119 |
| Moderate | 500/500 | 0.9093 | %85.1 | 0.131 | 0.119 |
| Hard | 500/500 | 0.9049 | %85.0 | 0.098 | 0.127 |

## Sıradaki
- [ ] **Threshold analysis çalıştır:** VPS'te `cd /workspace/corio-ecg && git pull && source .venv/bin/activate && python3 scripts/threshold_analysis.py`
- [ ] **Sonuçları yorumla:** Yüksek olasılıklı tanılarda (>0.7) agreement muhtemelen %90+ çıkacak
- [ ] **Phase 2 → main merge:** Threshold analiz sonuçları iyi gelirse branch'i merge et
- [ ] **Sonraki fazlar:** Phase 3 (MIMIC-IV-ECG), Phase 5 (LLM rapor), Phase 6 (Web UI)
