<!-- Purpose: Gate 2 closing result — the official hidden-test-set score for Corio-OMI. -->

# Gate 2 — resmî test seti sonucu (gönderim 1/2)

**Tarih:** 2026-08-01
**Model:** v2a (`models/omi/omi_finetuned_v2a.pt`) — ECGFounder backbone + MLP head
**Eşik:** 0.6423, F2-maks, **yalnızca eğitim fold'larından** (fold 1–4)
**Gönderim:** yazarların platformu, `http://39.105.59.221:8080/submit`
**Protokol kotası:** 2 gönderimden **1'i kullanıldı**

## Sonuç: baseline'ı her metrikte geçti

Gizli test seti, n=1.995:

| | Sens | Spec | PPV | NPV | Acc | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| **Corio-OMI v2a** | **0.7812** | **0.8907** | **0.3289** | **0.9834** | **0.8837** | **0.4630** |
| Yayımlanmış baseline CNN | 0.6970 | 0.8730 | 0.2770 | 0.9760 | 0.8610 | 0.3960 |
| EKG uzmanı 1 | 0.2770 | 0.9720 | 0.4070 | 0.9510 | 0.9270 | 0.3300 |
| EKG uzmanı 2 | 0.4290 | 0.9410 | 0.3380 | 0.9590 | 0.9080 | 0.3780 |
| **Δ (bizim − baseline)** | **+0.084** | **+0.018** | **+0.052** | **+0.007** | **+0.023** | **+0.067** |

**Altı metriğin altısında da baseline'ın üzerinde.** Hiçbir takas yok — bu bir
Pareto üstünlüğü, yani "şurada kazandık ama burada kaybettik" demek zorunda
kalmadığımız nadir bir durum.

Protokolün go/no-go kriteri (F1 > 0.396) karşılandı: **0.4630**.

## Validation → test genellemesi sağlam

| | Sens | Spec | F1 |
|---|---:|---:|---:|
| Fold 0 (validation) | 0.800 | 0.878 | 0.446 |
| **Resmî test** | **0.781** | **0.891** | **0.463** |
| Fark | −0.019 | +0.013 | +0.017 |

Üç metrikte de fark, fold 0'ın bootstrap %95 GA genişliğinin çok içinde
(ör. F1 GA'sı [0.402, 0.486] idi; test 0.463). Yani:

- Model fold 0'a **overfit olmamış**.
- Eğitim fold'larında seçilen eşik test setinde de doğru yerde durdu — eşik
  seçimini validation'a bakarak yapmamanın karşılığı bu.

Bu, sonucun tek bir şanslı split'in eseri olmadığının en güçlü göstergesi.

## Gönderilen dosya

`results/omi/submission/omi_submission_v2a.csv` — 1.995 satır, üç kolon:
`ecg_row_record`, `ecg_med_record`, `OMI` (0/1). **Hasta verisi içermez**;
yalnızca veri setinin kendi kayıt kimlikleri ve ikili tahminler. 304 pozitif
tahmin (%15.24).

⚠️ Platform şifresiz HTTP üzerinden çalışıyor. Gönderilen içerik zaten kamuya
açık test kimlikleri olduğu için gizlilik riski yok, ama not edilmeli.

## Ne ölçülemedi

Platform yalnızca toplam metrikleri döndürüyor. Bu yüzden test setinde:

- **NSTEMI-OMI alt grubu ölçülemiyor** — projenin asıl klinik hedefi olan grup.
  Fold 0'da duyarlılık 0.597'ydi; test setinde karşılığını bilmiyoruz.
- AUROC/AUPRC yok (platform binary tahmin alıyor).
- Kalibrasyon, alt grup adaleti, Time_Interval etkisi ölçülemiyor.

Yani bu sonuç "baseline'dan iyi bir OMI sınıflayıcı" iddiasını destekliyor;
"gizli oklüzyonu yakalıyor" iddiasını **desteklemiyor**. O iddia için fold 0
kanıtı geçerli ve orada duyarlılık hâlâ 0.597.

## Kısıtlar

1. **Tek merkezli** (Chongqing) veri; dış geçerlilik Gate 4'ün konusu.
2. **Temiz sinyal** sonucudur — fotoğraf hattında geçerliliği ölçülmedi.
   Corio'nun asıl farklılaşması orada ve Aşama B'nin konusu.
3. OMI tanımı TIMI 0–1 ile sınırlı (veri setinin kendi kısıtı): spontan
   reperfüze olmuş oklüzyonlar negatif etiketli.
4. İkinci gönderim kotası duruyor; bu sonuçtan sonra kullanılması için
   yeni bir gerekçe gerekir.

## Tekrar üretmek için

```bash
TQDM_DISABLE=1 CORIO_CALIBRATION=0 python scripts/predict_omi_test.py
curl -X POST -F "file=@results/omi/submission/omi_submission_v2a.csv" \
  http://39.105.59.221:8080/submit
```

Yol haritası: [zero-shot](gate2-omi-zeroshot-v1.md) → [lineer prob](gate2-omi-linear-probe-v1.md)
→ [fine-tuning](gate2-omi-finetune-v1.md) → [eşik düzeltmesi](gate2-omi-threshold-strategy-v1.md)
→ [ablasyon](gate2-omi-head-ablation-v1.md) → **bu sonuç**.
