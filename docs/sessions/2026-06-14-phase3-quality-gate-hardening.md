# Oturum: Phase 3 Quality-Gate Hardening
**Tarih:** 2026-06-14
**Süre:** ~2 saat
**Faz:** Phase 3 Digitization v2

## Özet
Quality-gate geliştirme altyapısı leakage-safe split, grup bazlı istatistik,
versiyonlanmış feature contract ve runtime-only karar API'siyle güçlendirildi.
Mevcut development baseline metrikleri korunurken klinik kanıt sınırları kodla
zorunlu hale getirildi.

## Yapılan İşler
- [x] ECG bazlı deterministic train/tune/test manifesti — `69ee11b`
- [x] Retroactive split ile locked holdout iddiasını engelleme — `69ee11b`
- [x] Confusion matrix, coverage, ECG-group bootstrap CI ve missed-reject listesi — `09fc4ca`
- [x] Inference-time quality feature contract — `9868194`
- [x] Runtime quality gate ve stable reason code ayrıştırması — `2054b95`
- [x] Tam test paketi: `190 passed, 1 skipped`
- [x] Ruff, mypy ve `git diff --check`: başarılı

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| Split birimi | `ecg_id` | Aynı ECG'nin yedi varyantı farklı splitlere sızmamalı |
| Mevcut split kanıt durumu | Retroactive development-only | Eşikler split oluşturulmadan önce tüm 70 görüntüde geliştirildi |
| Bootstrap birimi | ECG grubu | 70 görüntü yalnız 10 bağımsız ECG içeriyor |
| Runtime audit alanı | Stable reason code + mesaj | UI/audit mantığı değişebilir metne bağlanmamalı |
| Gradio abstention | Henüz bağlanmadı | Bağımsız matched holdout review tamamlanmadı |

## Değiştirilen/Oluşturulan Dosyalar
```text
configs/quality_gate_split_v1.yaml              — grouped split ayarları
configs/quality_feature_contract_v1.yaml        — aktif inference özellik sözleşmesi
results/quality-gate/quality_gate_split_v1.json — deterministic split manifesti
src/evaluation/grouped_split.py                 — leakage-safe split üretimi
src/evaluation/split_selection.py               — tune/locked evaluation erişim kuralları
src/evaluation/quality_gate_metrics.py           — grup bazlı istatistikler
src/quality/features.py                          — typed feature extraction
src/quality/gate.py                              — runtime karar motoru
src/quality/models.py                            — outcome ve stable reason code modelleri
src/quality/thresholds.py                        — inference-only threshold yükleme
```

## Teknik Notlar
- Split dağılımı: train/tune/test = `6/2/2` ECG ve `42/14/14` görüntü.
- Her splitte yedi capture kategorisi ECG sayısıyla dengeli kalıyor.
- Baseline: `0` false accept, `9` false reject, `4` missed reject,
  reject recall `%81.8`, non-reject coverage `%61.4`.
- ECG-group bootstrap reject recall CI: yaklaşık `%64-%100`.
- Dört missed reject'in tamamı `warn`; `accept` olarak kaçan reject yok.
- Aktif feature contract altı alan içeriyor; blur/glare/occlusion henüz yok.

## Sıradaki
- [ ] Task 5.1: frozen baseline'a karşı digitization regression harness oluştur
- [ ] Candidate deneylerde kategori regression toleranslarını tanımla
- [ ] Task 4.2'yi yalnız bağımsız matched holdout review sonrasında uygula
