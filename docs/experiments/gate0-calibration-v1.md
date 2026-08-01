<!-- Purpose: Gate 0 result — per-class calibration and evidence tiering for the 150 ECGFounder heads. -->

# Gate 0 — sınıf-bazlı kalibrasyon ve kanıt katmanları (v1)

**Tarih:** 2026-08-01
**Branch:** `feature/gate0-calibration`
**Kaynak plan:** `docs/research/2026-07-06-specialization-strategy.md` § 5, "Gate 0 — Mevcut tanı katmanını dürüstleştir"

## Yönetici özeti

150 sınıflı ECGFounder çıktısı tek bir global `0.5` eşiğiyle kullanılıyordu. Bu eşik
birçok head için yanlıştı: model bazı tanıları **sıralayabiliyor** ama o tanıyı
**söyleyemiyordu**. En çarpıcı örnek inferior infarct — AUROC `0.81`, yani ayrım gücü
var, ama F1 `0.03`, yani pratikte hiç bildirilmiyor.

Gate 0 bunu yeni model eğitmeden düzeltti: PTB-XL fold 9'da sınıf başına Platt scaling
ve eşik öğrenildi, fold 10'da tek seferlik audit yapıldı.

**Held-out fold 10 sonucu (n=2198, 31 değerlendirilebilir sınıf):**

| Metrik | Ham + global 0.5 | Kalibre + sınıf-bazlı eşik | Δ |
|---|---:|---:|---:|
| Macro AUROC | 0.8679 | 0.8679 | 0.000 |
| Macro AP | 0.4454 | 0.4454 | 0.000 |
| **Micro F1** | 0.5885 | **0.6465** | **+0.058** |
| **Macro F1** | 0.3673 | **0.4129** | **+0.046** |
| **Brier** | 0.0553 | **0.0398** | **−28%** |
| **ECE** | 0.0685 | **0.0172** | **−75%** |

AUROC ve AP'nin **değişmemesi beklenen ve istenen** sonuçtur: Platt scaling monotonik
bir dönüşümdür, sıralamayı değiştirmez. Kazanç tamamen olasılık ölçeği ve karar
noktasındadır — yani modelin bildiği ama söyleyemediği şeyi söyleyebilir hale
getirmekten gelir.

## Yöntem

### Split disiplini

- **Fold 9 (n=2183 etiketli):** kalibrasyon ve eşik öğrenme. Yeni inference koşuldu.
- **Fold 10 (n=2198 etiketli):** yalnızca final audit. Mevcut baseline olasılık
  matrisi kullanıldı, hiçbir parametre bu fold'a bakılarak seçilmedi.
- `scripts/build_calibration.py` fold 10 verilirse **çalışmayı reddeder**;
  `scripts/audit_calibration.py` fold 10'da fit edilmiş bir artefakt görürse reddeder.

**Ground truth:** ECGFounder'ın resmî PTB-XL etiket CSV'si (`ecgfounder_ptbxl_label.csv`),
150 head'in kendi etiket uzayında. Semantic SCP mapping değil — üretimde kullanılan
head'lerle birebir aynı uzay olduğu için seçildi.

### Kalibrasyon

Head başına logit üzerinde tek değişkenli lojistik regresyon (Platt scaling), L2
regularization ile. 150 head'in **27'si** fit edilebilir destek buldu.

İki koruma eklendi:

1. **Negatif slope reddi.** ANTERIOR INFARCT için fit negatif slope üretti — model o
   head'de şansa yakın (AUROC ~0.55) ve fit bunu "düzeltmek" için skoru ters çeviriyordu.
   Bu kalibrasyon değil, bozuk bir head'in maskelenmesidir; üstelik AUROC'yi düşürüyordu
   (0.8679 → 0.8648). Slope ≤ 0 olan fitler reddedilir, head ham bırakılır.
2. **Precision tabanı (0.30).** Şansa yakın bir head'de saf F1 maksimizasyonu "her şeye
   evet de" noktasına yakınsar: recall 1'e gider, F1 `2p/(1+p)`'ye oturur ve bu her
   dürüst eşiği yener. İlk denemede ANTERIOR INFARCT eşiği `0.028` çıktı — pratikte her
   EKG'de ateşlenirdi. Taban eklendikten sonra bu head varsayılan `0.5`'e döndü.

   Bu tabanın etkisi ölçüldü: tabansız sürümde micro F1 **düşüyordu** (0.5885 → 0.5098)
   çünkü kazanılan macro F1 sahteydi. Tabanla birlikte her iki metrik de iyileşiyor.

### Kanıt katmanları (evidence tiers)

Her head fold 9'daki kanıtına göre etiketlenir:

| Tier | Kural | Sayı | Anlamı |
|---|---|---:|---|
| `validated` | ≥10 pozitif **ve** AUROC ≥ 0.80 **ve** F1 ≥ 0.30 | 18 | Bulgu olarak bildirilebilir |
| `provisional` | ≥10 pozitif ama eşikleri geçemiyor | 9 | Belirsizlik uyarısıyla göster |
| `research_only` | <10 pozitif — ölçülemedi | 123 | Asla bulgu değil |

## ⚠️ Kritik tasarım kısıtı: kritik tanılar gizlenmez

Tiering'i körlemesine uygulamak **tehlikeli** olurdu. Kritik tanı listesinin 8 üyesinin
7'si `research_only` çıktı:

| Tanı | Tier | Fold 9 pozitif |
|---|---|---:|
| ATRIAL FIBRILLATION | validated | 151 |
| ACUTE MI / STEMI | research_only | **0** |
| VENTRICULAR TACHYCARDIA | research_only | 5 |
| WIDE QRS TACHYCARDIA | research_only | **0** |
| WITH COMPLETE HEART BLOCK | research_only | **0** |
| SUPRAVENTRICULAR TACHYCARDIA | research_only | 5 |
| ACUTE MI | research_only | **0** |
| ACUTE PERICARDITIS | research_only | **0** |

Bu bir model zaafı değil, **veri seti zaafıdır**: PTB-XL'de STEMI, VT ve tam blok
pozitifi yok. Tiering'i filtre olarak uygulasaydık, klinisyenin asla kaçırmaması gereken
etiketleri tam olarak susturmuş olurduk.

Bu yüzden `_is_hidden_research_output()` kritik indeksleri **muaf tutar**: research_only
olsalar bile ana listede kalırlar. Test `test_critical_head_is_never_hidden` bu davranışı
kilitler.

## Üretim entegrasyonu

- `ECGDiagnoser` açılışta `configs/calibration/ptbxl_fold9_v1.json` yükler; dosya yoksa
  veya `CORIO_CALIBRATION=0` ise ham davranışa döner (regresyon yok).
- Kalibrasyon `_forward_probabilities` çıktısına, **rate-consistency düzeltmelerinden
  önce** uygulanır — fit edilen dağılımla aynı ölçek. Rate düzeltmeleri çarpansal
  olduğu için her iki ölçekte de anlamlıdır.
- Tüm eşik karşılaştırmaları tek bir yardımcıdan geçer: `is_above_threshold()`
  (`diagnose.py`). app.py, `structured_report.py` ve `diagnosis_reconciliation.py`
  bu fonksiyonu kullanır, böylece kalibre ve kalibre olmayan koşular tutarlı kalır.
- Web arayüzü: her tanının yanında tier rozeti; kritik olmayan research_only head'ler
  ayrı "Research output — not a finding" bloğuna taşındı.

## En büyük kazançlar (fold 10, held-out)

| Tanı | F1 önce | F1 sonra | Δ | Tier |
|---|---:|---:|---:|---|
| INFERIOR INFARCT | 0.029 | 0.423 | +0.393 | validated |
| LEFT ANTERIOR FASCICULAR BLOCK | 0.323 | 0.670 | +0.347 | validated |
| WITH 1ST DEGREE AV BLOCK | 0.049 | 0.377 | +0.328 | validated |
| ANTEROSEPTAL INFARCT | 0.502 | 0.637 | +0.136 | validated |
| LEFT POSTERIOR FASCICULAR BLOCK | 0.364 | 0.462 | +0.098 | validated |
| SINUS TACHYCARDIA | 0.771 | 0.865 | +0.094 | validated |
| NORMAL ECG | 0.681 | 0.759 | +0.078 | validated |
| LEFT BUNDLE BRANCH BLOCK | 0.744 | 0.821 | +0.077 | validated |

Tier bazlı fold 10 performansı: `validated` 18 sınıf, macro AUROC 0.908, macro F1 0.598.
`provisional` 9 sınıf, macro AUROC 0.741, macro F1 0.061 — bu katmanın neden ayrı
sunulduğunu gösteriyor.

## Regresyonlar

Kazanç bedava değil. Fold 10'da F1'i 0.01'den fazla gerileyen 4 sınıf:

| Tanı | F1 önce | F1 sonra | Δ | Tier |
|---|---:|---:|---:|---|
| LOW VOLTAGE QRS | 0.179 | 0.012 | −0.167 | provisional |
| LEFT ATRIAL ENLARGEMENT | 0.102 | 0.000 | −0.102 | provisional |
| PREMATURE VENTRICULAR COMPLEXES | 0.764 | 0.732 | −0.032 | validated |
| WITH QRS WIDENING | 0.028 | 0.000 | −0.028 | provisional |

Üçü `provisional`, yani zaten belirsiz işaretli head'ler; precision tabanı onlara fold 9'da
temkinli bir eşik verdi ve fold 10'da neredeyse hiç ateşlemediler. Bu tabanın bilinçli
maliyetidir: "her şeye evet de" davranışını engellerken zayıf head'lerin sessizleşmesini
kabul ediyoruz.

Tek `validated` regresyon PVC (−0.032) ve küçüktür; öğrenilen eşik (0.512) varsayılana
zaten çok yakın. Toplam 150 head'in **21'i** varsayılan `0.5`'ten farklı bir eşik aldı.

## Sınırlamalar (dürüstlük bölümü)

1. **Bu temiz sinyal kalibrasyonudur, fotoğraf kalibrasyonu değil.** Fold 9/10 PTB-XL'in
   dijital sinyalleridir. Digitize edilmiş fotoğrafın olasılık dağılımı farklıdır;
   kalibrasyonun oraya taşındığı **ölçülmedi**. Gerçek fotoğraf üzerinde tekrar
   ölçülmesi gerekir.
2. **123 head hâlâ ölçülemez durumda.** Gate 0 bunları düzeltmedi, yalnızca dürüstçe
   etiketledi. Kapsamları PTB-XL'in ötesinde veri gerektirir.
3. **`provisional` katmanı zayıf.** Fold 10 macro F1 0.061 — bu head'ler şu an klinik
   değer taşımıyor, yalnızca gizlenmiyor.
4. **Eşikler F1'e göre seçildi.** Klinik olarak duyarlılık-ağırlıklı bir kayıp
   (kaçırılan infarkt ≫ fazladan uyarı) daha doğru olabilir; bu bir sonraki iterasyon.
5. **Kritik tanı muafiyeti kalibrasyonla ilgisizdir.** O head'ler hâlâ `0.5` eşiğinde ve
   doğrulanmamış durumda — muafiyet onları görünür tutar, güvenilir yapmaz.

## Üretilen dosyalar

- `src/calibration/` — `tiers.py`, `artifact.py`, `fit.py`, `ptbxl_data.py`
- `scripts/build_calibration.py`, `scripts/audit_calibration.py`
- `configs/calibration/ptbxl_fold9_v1.json` — üretimde yüklenen artefakt
- `results/metrics/calibration_audit_fold10.json` — tam audit raporu
- `tests/test_calibration.py` (26 test), `tests/test_web_app.py` (+6 test)

Test durumu: **410 passed, 1 skipped**.

## Tekrar üretmek için

```bash
python -m src.training.evaluate --fold 9 --output-prefix ptbxl_calibration_fold9
python scripts/build_calibration.py
python scripts/audit_calibration.py
```
