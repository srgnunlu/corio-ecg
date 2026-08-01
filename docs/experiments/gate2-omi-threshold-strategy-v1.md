<!-- Purpose: Fixing the operating point — F1 selection was clinically wrong for OMI. -->

# Gate 2 — eşik stratejisi düzeltmesi (v1)

**Tarih:** 2026-08-01
**Model:** değişmedi (`models/omi/omi_finetuned_v1.pt`, AUROC 0.9081 / AUPRC 0.4249)
**Yeniden eğitim yapılmadı** — yalnızca karar eşiği değişti
**Ölçüm:** fold 0 (n=3.592, 230 OMI); eşikler **eğitim fold'larında** seçildi

## Neden

[Fine-tuning deneyi](gate2-omi-finetune-v1.md) F1'de baseline'ı geçti ama
**duyarlılığı baseline'ın altındaydı** (0.600 vs 0.697). Sebep model değil,
ölçüt: F1 duyarlılık ile precision'a eşit ağırlık verir. OMI'de kaçırılan
oklüzyon gecikmiş reperfüzyon demektir; fazladan uyarı ise ikinci bir bakış.
Bu asimetri F1'de yok.

## Sonuç: model her iki yönden de baseline'dan üstün

Aynı model, dört farklı eşikle. Hepsi eğitim fold'larında seçildi, fold 0'da
raporlanıyor:

| Strateji | eşik | **Sens** | **Spec** | PPV | NPV | F1 |
|---|---:|---:|---:|---:|---:|---:|
| *Yayımlanmış baseline* | — | *0.697* | *0.873* | *0.277* | *0.976* | *0.396* |
| F1-maks (eski seçim) | 0.718 | 0.600 | 0.930 | 0.371 | 0.971 | **0.459** |
| **F2-maks (yeni seçim)** | 0.508 | **0.800** | 0.869 | 0.295 | **0.985** | 0.431 |
| Sens ≥0.697'de spec-maks | 0.656 | 0.674 | **0.912** | 0.343 | 0.976 | 0.455 |
| Spec ≥0.873'te sens-maks | 0.505 | **0.804** | 0.867 | 0.293 | 0.985 | 0.430 |

İki karşılaştırma modelin üstünlüğünü eşik seçiminden bağımsız gösteriyor:

**İso-duyarlılık:** baseline'ın duyarlılığını hedeflediğimizde spesifisite
**0.912 [0.901, 0.922]** — baseline'ın 0.873'ünün üzerinde ve %95 GA'nın alt
sınırı bile onu aşıyor. Aynı yakalama oranında belirgin şekilde daha az yanlış
alarm.

**İso-spesifisite:** baseline'ın yanlış alarm yükünü koruduğumuzda duyarlılık
**0.804 [0.750, 0.855]** — baseline'ın 0.697'sine karşı. Aynı alarm yükünde
yaklaşık **%15 daha fazla oklüzyon** yakalanıyor.

F1-maks eşiği bu kazancın hiçbirini göstermiyordu; sadece modeli baseline'dan
daha sessiz hale getiriyordu.

## Seçim: F2-maks

**Yeni birincil işletim noktası F2-maks (eşik 0.508).** Gerekçe:

- Önceden tanımlı bir klinik ölçüte dayanıyor (duyarlılığa precision'ın 2 katı
  ağırlık), baseline'ın işletim noktasına değil — yani baseline değişse de
  kriter geçerli kalır.
- Duyarlılık 0.800, baseline'ın 0.697'sinin belirgin üzerinde.
- Spesifisite 0.869, baseline'ın 0.873'üyle pratikte aynı.
- **NPV 0.985** (baseline 0.976) — dışlama gücü daha iyi, ki triyajda bu değerli.

İso-spesifisite stratejisi neredeyse aynı noktayı buluyor (eşik 0.505), bu da
seçimin eşik ayarına duyarlı olmadığını gösteriyor.

## ⚠️ Alt gruplar: duyarlılık nereden geliyor?

F2 eşiğinde:

| Alt grup | n | OMI | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|
| Tümü | 3.592 | 230 | 0.800 | 0.869 | 0.431 |
| ACS-pozitif | 526 | 229 | 0.799 | 0.465 | 0.641 |
| **STEMI etiketli** | 295 | 152 | **0.915** | 0.287 | 0.707 |
| **NSTEMI etiketli** | 231 | 77 | **0.571** | 0.630 | 0.494 |
| Aralık ≤12 sa | 1.098 | 136 | 0.779 | 0.815 | 0.505 |
| Aralık >12 sa | 2.494 | 94 | 0.830 | 0.891 | 0.359 |

**Genel 0.800 duyarlılık büyük ölçüde STEMI-OMI'den geliyor (0.915).** Asıl
klinik hedef olan NSTEMI-OMI'de duyarlılık yalnızca **0.571** — gizli
oklüzyonların %43'ü hâlâ kaçıyor.

Bu, projenin değer önerisinin tam kalbindeki eksik. STEMI-OMI'yi zaten ST
elevasyonundan görebilirsiniz; modelin var olma sebebi NSTEMI grubudur ve orada
performans hâlâ yetersiz.

Yine de F1-maks eşiğine göre NSTEMI'de belirgin iyileşme var: F1 0.409 → 0.494.

## Protokol değişikliği (şeffaflık notu)

[Gate 2 protokolü](../plans/gate2-omi-protocol-v1.md) v1'de eşik seçimini
"Gate 0'daki precision-tabanlı F1 optimizasyonu" olarak tanımlamıştı. Bu
belge onu **F2-maks** ile değiştiriyor.

Bu bir sonuç-sonrası hedef kaydırması (HARKing) riski taşır, o yüzden açıkça
kayda geçiyorum:

1. **Değişiklik gerekçesi sonuçlardan değil, ölçüt hatasından geliyor.** F1'in
   OMI için yanlış asimetri taşıdığı, sonuçlara bakmadan da doğruydu; nitekim
   Gate 0 dokümanı ve Gate 2 protokolünün "Kısıtlar" bölümü bunu zaten
   "bir sonraki iterasyon" olarak not etmişti.
2. **Eski F1 sonucu silinmedi**, yukarıdaki tabloda ve
   [fine-tuning dokümanında](gate2-omi-finetune-v1.md) duruyor.
3. **İso-duyarlılık ve iso-spesifisite karşılaştırmaları eşik seçiminden
   bağımsızdır** ve modelin üstünlüğünü tek başlarına gösterir. Asıl kanıt
   bunlardır; F2 yalnızca hangi noktada çalışacağımızı belirler.
4. **Model değişmedi**, yalnızca karar eşiği. Eşik zaten eğitim fold'larında
   seçildiği için değerlendirilen fold'a bakılmadı.

Protokolün geri kalanı (split, iki gönderim limiti, alt gruplar) aynen geçerli.

## Kısıtlar

1. Bu hâlâ **validation-fold** sonucudur; resmî test setine gönderim yapılmadı.
2. **NSTEMI duyarlılığı 0.571** — test setine gönderim yapılsa bile bu, ürünün
   klinik iddiasını sınırlar.
3. F2'deki β=2 seçimi bir konvansiyondur, kalibre edilmiş bir maliyet oranı
   değil. Gerçek maliyet oranı (kaçırılan OMI / yanlış alarm) klinik bir karardır.
4. Tek merkezli veri; dış geçerlilik Gate 4.

## Tekrar üretmek için

```bash
TQDM_DISABLE=1 CORIO_CALIBRATION=0 python scripts/tune_omi_threshold.py
```

Model skorları `results/omi/scores/` altında önbelleklenir; yeniden eğitim veya
yeniden forward pass gerekmez. Rapor
`results/omi/threshold_strategies_fold0.json`.
