# Phase C.3 — Rhythm Analysis, Normal ECG Detection & Structured Report (v1)

**Tarih:** 2026-06-17
**Branch:** `feature/phase3-failure-triage` (push edilmedi)
**Durum:** Tamamlandı — sentetik doğrulama yapıldı, gerçek-foto concordance bekliyor.
**Master roadmap:** Faz C.3 (`docs/plans/master-roadmap-v1.md`)

## Özet (TLDR)

ECGFounder'ın 150 sınıflık ham olasılık çıktısının yanına, klinisyenin beklediği
**yorumlanabilir ritim katmanını** ve **tek cümlelik kararı** ekledik:

- **C.3.1 Ritim analizi** (`src/measurement/rhythm_analysis.py`): sinüs vs atriyal
  fibrilasyon ayrımı, hız sınıfı (brady/normal/taşi), PVC ektopi bayrağı —
  C.1'de kurtarılan 10 s ritim şeridi üzerinden, scipy-only kural tabanlı.
- **C.3.2 Normal EKG tespiti** (`src/report/structured_report.py`): tüm bulgular
  normalse "Normal ECG" der; aksi halde **neden** anormal/indeterminate olduğunu
  açıkça listeler.
- **C.3.3 Yapılandırılmış rapor**: HR + ritim + interval + AI top-5 + genel yorumu
  birleştiren `ECGReport` nesnesi + JSON çıktısı; web app'e başlık kartı olarak
  entegre edildi.

**Test:** 18 yeni test (`test_rhythm_analysis.py`, `test_structured_report.py`),
tüm suite **313 passed, 1 skipped**. Yeni dosyalar ruff temiz; mypy yeni
modüllerde hata yok (kalan hatalar scipy stub eksikliği, mevcut).

## Tasarım kararları

### Ritim analizi (C.3.1)

Kaynak önceliği **ritim şeridi** (C.1 fix'iyle kurtarılan gerçek 10 s Lead II).
Tiled tanı sinyali ~2.5 s'yi tekrarladığı için gerçek RR dizisi taşımaz ve ritim
körüdür — yalnızca son çare fallback.

| Karar | Yöntem | Eşik |
|---|---|---|
| Düzenlilik | RR varyasyon katsayısı (CV = std/mean) | regular < 0.12 |
| AF | yüksek CV **VE** P dalgası yokluğu (her ikisi de) | CV ≥ 0.16, P-frac < 0.35 |
| Sinüs | düzenli **VE** P dalgası mevcut | P-frac ≥ 0.50 |
| Hız sınıfı | medyan RR → bpm | <60 brady, >100 taşi |
| PVC | erken atım **VE** belirgin geniş QRS | RR_prev < 0.80·medyan, QRS > 1.35·medyan & ≥110 ms |

**Neden iki koşullu AF/PVC?** Tek başına düzensizlik PVC kaynaklı olabilir (P
dalgaları korunur → AF değil); tek başına erken atım PAC'dir; tek başına geniş QRS
dal bloğudur. İki koşulu birlikte arayarak yanlış pozitifleri eler. Genlik z-score
normalize olduğu için tüm QRS-genişlik eşikleri **hastanın kendi medyanına göreli**
(mutlak mV anlamsız), gürültüyü elemek için yumuşak mutlak taban (110 ms) eklendi.

P dalgası varlığı `delineation.delineate_beat` ile atım-başına `p_onset`
saptanma oranından geliyor — yeni delineation kodu yazılmadı, C.2 altyapısı tekrar
kullanıldı.

### Normal EKG tespiti (C.3.2)

**Strict + açıklanabilir.** "Normal" demek klinik ağırlık taşır, bu yüzden TÜM
kriterler sağlanmalı:

1. Sinüs ritmi (`rhythm_basis == "sinus"`)
2. Hız 60–100 bpm
3. Ektopi yok
4. Normal PR (ölçülmüş **ve** normal)
5. Normal QRS (<120 ms, ölçülmüş)
6. Normal QTc (cinsiyet eşiğiyle)
7. Eşik üstü anlamlı ECGFounder patolojisi yok

**Ölçülemeyen interval normal sayılmaz** — doğrulayamadığımızı iddia etmeyiz.
Bu durumda, pozitif bir anormallik de yoksa karar **"Indeterminate ECG"** olur
(normal olduğunu kanıtlayamadık, ama anormal olduğunu da kanıtlayamadık).
Pozitif anormallik (non-sinüs ritim, ektopi, AI patolojisi, ölçülmüş anormal hız)
varsa **"Abnormal ECG"**.

"Anlamlı patoloji" = eşik üstü olup `_BENIGN_LABELS` whitelist'inde olmayan tanılar.
Whitelist: NORMAL SINUS RHYTHM, NORMAL ECG, SINUS RHYTHM, OTHERWISE NORMAL ECG,
BORDERLINE ECG, SINUS TACHY/BRADYCARDIA, WITH SINUS ARRHYTHMIA, EARLY REPOLARIZATION.
(Sinüs brady/taşi hız kriteriyle ele alındığı için patoloji sayılmaz.)

### Yapılandırılmış rapor (C.3.3)

`ECGReport` dataclass: HR, ritim sınıfı/temeli/hız, ektopi, PR/QRS/QT/QTc + bayraklar,
AI top-5, `is_normal`, `overall_assessment`, `abnormal_reasons`, `normal_criteria`.
`report_to_dict()` JSON-ready çıktı verir (ileride PDF/API için; floatlar
yuvarlanır). Web app'te bulgular bölümünün en üstüne renkli **başlık kartı**
eklendi: yeşil=Normal, kırmızı=Abnormal, sarı=Indeterminate; ritim + HR satırı,
PVC uyarısı, anormallik gerekçe listesi ve "karar destek — sorumluluk hekimde"
disclaimer'ı.

## Doğrulama

- **Sentetik:** üçgen-dalga sentetik EKG'lerle (bilinen ritim) — normal sinüs,
  brady, taşi, AF (düzensiz + P yok), PVC (erken + geniş atım) doğru sınıflanıyor;
  düzensiz-ama-P-var AF olarak işaretlenmiyor; ritim şeridi tiled sinyale tercih
  ediliyor.
- **Rapor:** normal/abnormal/indeterminate dallarının hepsi test edildi; JSON
  round-trip doğrulandı; HR fallback (estimated_hr) çalışıyor.

## Bilinen kısıtlar / sonraki adım

- **Sentetik-only.** Gerçek telefon fotoğraflarında ritim/normal concordance
  ölçülmedi (Faz B veri darboğazı; PMcardio'da tanı/ritim etiketi yok). Kurallar
  konservatif tutuldu.
- AF saptaması Lead II şeridine bağlı; şerit yoksa (3x4+1R olmayan layout'lar)
  kalite "low/unmeasurable" düşer ve karar indeterminate'e kayar — bilinçli.
- PVC sayımı kaba; bigemini/kuplet ayrımı veya PAC sınıfı yok (C.5 kapsamı).
- Rapor LLM doğal-dil özeti içermiyor (D.2 kapsamı) — şimdilik deterministik.

## Dosyalar

- `src/measurement/rhythm_analysis.py` — ritim sınıflandırma (yeni)
- `src/report/__init__.py`, `src/report/structured_report.py` — rapor + normal tespit (yeni)
- `src/web/app.py` — `_format_report_headline` + analiz akışına entegrasyon
- `tests/test_rhythm_analysis.py`, `tests/test_structured_report.py` — 18 test (yeni)
