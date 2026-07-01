# Phase 4 — VT/SVT Wide-Complex Tachycardia Criteria (v1)

**Tarih:** 2026-06-20 (audit v1) · doküman 2026-07-02
**Branch:** `feature/phase4-vtsvt` (push edilmedi)
**Durum:** Motor + audit tamam; **negatif-kontrol** doğrulaması yapıldı. VT-pozitif
duyarlılık doğrulaması bekliyor (veri yok).
**Master roadmap:** Faz 4 — VT/SVT specialization (Brugada/Vereckei), `SPEC.md`

## Özet (TLDR)

Geniş-kompleks taşikardide (WCT) **VT vs SVT-with-aberrancy** ayrımı için
deterministik, denetlenebilir bir kriter motoru. Dijitalize sinyalden Brugada ve
Vereckei/aVR kriterlerini hesaplar; her kararı **kanıt (`evidence`) + kısıt
(`limitations`)** listesiyle döndürür.

En önemli tasarım kararı: **motor dışlama yoluyla asla "SVT" demez.** Yalnızca
VT-lehine pozitif kanıt toplar; kanıt yoksa karar *"indeterminate wide-complex
tachycardia"* olarak kalır. Bir VT'yi yanlışlıkla "SVT" deyip kaçırmak, tersinden
çok daha tehlikeli olduğu için bu asimetri bilinçlidir.

**Audit v1:** 53 PTB-XL **negatif-kontrol** kaydında (VT taklitçileri) **özgüllük
1.0, sıfır yanlış-pozitif.** Duyarlılık ölçülemedi — PTB-XL adjudike VT-pozitif
etiketi sağlamıyor.

**Test:** 28 yeni VT/SVT testi; tüm suite **378 passed, 1 skipped**.

## Tasarım kararları

### İki katmanlı kapı (over-call'u önlemek için)

| Kapı | Koşul | Amaç |
|---|---|---|
| **Scope** | hız > 100 bpm **ve** QRS ≥ 120 ms | WCT değilse hiç değerlendirme yapma (`not_wide_complex_tachycardia`) |
| **Morphology** | QRS ≥ 140 ms | Otomatik morfoloji kriterleri yalnızca belirgin geniş QRS'te aktif — dar-sınır genişlikte morfoloji ölçümü güvenilmez |

Kapsam dışıysa (`in_scope=False`) hiçbir kriter tetiklenmez; en fazla "indeterminate"
verilir, "VT" verilmez.

### Brugada kriterleri (`criteria.py::_evaluate_brugada`)

| Kriter | Ölçüm |
|---|---|
| `brugada_absent_rs_all_precordial` | 6 prekordiyal lead'in **hepsinde** RS kompleksi yok |
| `brugada_rs_interval_gt_100ms` | En geniş prekordiyal RS interval > 100 ms **ve** ölçülen QRS ile tutarlı (RS ≤ QRS + 20 ms) |
| `brugada_terminal_v1_v6_*` | V1/V6 terminal morfolojisi RBBB/LBBB paternine uyuyor (`terminal_morphology.py`) |
| `brugada_av_dissociation` | **Dışarıdan adjudike** flag (`av_dissociation_present`) |
| `brugada_capture_or_fusion_beats` | **Dışarıdan adjudike** flag (`capture_or_fusion_beats_present`) |

**RS-QRS tutarlılık kontrolü** kritik: RS interval ölçülen QRS'ten belirgin geniş
çıkıyorsa ölçüm hatalıdır (dijitalizasyon artefaktı) — kriteri tetiklemez, bunun
yerine kısıt olarak raporlar.

### Vereckei / aVR kriterleri (`criteria.py::_evaluate_vereckei`)

Yalnızca aVR üzerinden: `vereckei_initial_r_in_avr`, `vereckei_initial_r_or_q_gt_40ms`,
`vereckei_initial_downstroke_notching`, `vereckei_vi_vt_ratio_leq_1` (Vi/Vt ≤ 1).

### Feature çıkarımı (`features.py`, `waveform.py`)

- **Kaynak önceliği:** ritim şeridi (gerçek beat anchor) > tiled tanı sinyali.
- **R-peak anchor:** sırayla II, V1–V6; std > 0.05 olan ilk lead'de ≥ 3 beat.
- Prekordiyal (V1–V6) + aVR için atım-ortalamalı QRS morfolojisi çıkarılır:
  RS varlığı/interval, initial deflection yön+genişlik, notch, Vi/Vt, V1/V6 için
  qrs_pattern (r/qr/qs/rs) ve r/s oranı.

### AV disosiasyon & capture/fusion neden otomatik değil?

Bunlar VT'nin en spesifik bulguları ama otomatik saptaması güvenilmez (P-QRS
disosiasyonu tek şeritte, düşük çözünürlükte ayırt edilemez). Motora **dışarıdan
adjudike bayrak** olarak girerler; verilmezlerse `av_dissociation_not_assessed` /
`capture_or_fusion_beats_not_assessed` kısıtı raporlanır. Böylece motor "bakılmadı"
ile "yok" arasındaki farkı gizlemez.

## Audit metodolojisi

1. **Manifest** (`scripts/build_vtsvt_ptbxl_manifest.py`): PTB-XL'den VT-taklitçisi
   negatif-kontrol kohortu seçer (BBB/IVCD, paced ritim, aberran iletili atriyal
   aritmi, SVT/PSVT, geniş-QRS sinüs taşikardisi). Çıktı `data/splits/…yaml`
   (gitignore'lu, üretilen artefakt). `configs/vtsvt_audit_manifest_v1.yaml` boş
   **şablon**dur (şema + doldurma talimatı).
2. **Audit** (`scripts/evaluate_vtsvt_criteria.py`): her kayıtta kriterleri çalıştırır,
   `results/vtsvt/vtsvt_criteria_audit_v1.{csv,json}` üretir. Per-kayıt kanıt/kısıt +
   agregat confusion matrisi.

Yeniden üretim:
```bash
python scripts/build_vtsvt_ptbxl_manifest.py
python scripts/evaluate_vtsvt_criteria.py \
  --manifest data/splits/vtsvt_ptbxl_audit_manifest_v1.yaml
```

## Sonuçlar (audit v1, n=53)

| Metrik | Değer |
|---|---|
| Toplam / başarılı | 53 / 53 (0 hata) |
| Kapsama giren (in_scope) | 4 |
| VT-lehine (supports_vt) | **0** |
| Indeterminate | 4 |
| Terminal morfoloji ölçülebilir | 53 |
| **Özgüllük** | **1.0** |
| Duyarlılık | ölçülemedi (VT-pozitif yok) |
| Confusion | TN 53, **FP 0**, FN 0, TP 0 |

**Kohort dağılımı:** aberran iletili atriyal aritmi 12, BBB/IVCD 12, paced 12,
geniş-QRS sinüs taşi 12, SVT/PSVT 4, SVT+BBB 1.

**Yorum:** 4 kayıt gerçekten WCT eşiğini geçti (hız+genişlik) ama hiçbiri VT-lehine
kriter tetiklemedi → hepsi güvenli şekilde *indeterminate*. Kalan 49 kayıt WCT
tanımına girmediği için kapsam dışı bırakıldı. Motor tek bir VT-taklitçisini bile
yanlışlıkla "VT" dememiş.

## Bilinen kısıtlar / sonraki adım

- **⚠️ En kritik boşluk — duyarlılık kör nokta.** Kriterlerin gerçek VT'yi
  yakaladığını gösterecek adjudike WCT-pozitif verisi yok. Bu audit yalnızca
  **yanlış-pozitif tetikleme davranışı** ölçer; kilitli sensitivite/spesifisite
  iddiası değildir (audit'in kendi `limitations` alanı da bunu belirtir).
- AV disosiasyon + capture/fusion otomatik saptanmıyor (dışarıdan flag) — VT'nin en
  spesifik bulguları hâlâ manuel.
- Gerçek telefon fotoğrafı doğrulaması yok (PTB-XL clean sinyal + ritim şeridi).
- Düzensiz WCT için kriterler valide değil (algoritmalar regular WCT için geliştirildi);
  düzensizlikte `irregular_rhythm_algorithms_validated_for_regular_wct` kısıtı düşer.

## Dosyalar

- `src/vtsvt/` — `criteria.py` (motor), `features.py` (feature orkestrasyon),
  `waveform.py` + `qrs_shape.py` (lead morfolojisi), `terminal_morphology.py`
  (V1/V6 Brugada terminal), `models.py` (dataclass'lar), `__init__.py` (public API)
- `src/evaluation/vtsvt_audit*.py`, `vtsvt_ptbxl_manifest.py` — audit + manifest üretici
- `scripts/build_vtsvt_ptbxl_manifest.py`, `scripts/evaluate_vtsvt_criteria.py`
- `src/report/structured_report.py` (kriter entegrasyonu), `src/report/vtsvt_pdf.py`,
  `src/web/vtsvt_card.py`, `src/web/app.py` — sunum
- `configs/vtsvt_audit_manifest_v1.yaml` (şablon), `results/vtsvt/*.{csv,json}` (audit v1)
- Testler: `tests/test_vtsvt_criteria.py`, `test_vtsvt_conservative_criteria.py`,
  `test_vtsvt_presentation.py`, `test_vtsvt_ptbxl_manifest.py`,
  `test_evaluate_vtsvt_criteria_script.py`
