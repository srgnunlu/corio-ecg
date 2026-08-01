<!-- Purpose: Gate 2 step 3 — two-stage fine-tuning of ECGFounder for OMI, and its honest limits. -->

# Gate 2 — OMI fine-tuning (v1)

**Tarih:** 2026-08-01
**Protokol:** [gate2-omi-protocol-v1.md](../plans/gate2-omi-protocol-v1.md) § 5, adım 3
**Ölçüm:** fold 0 (n=3.592, 230 OMI) — resmî test setine **dokunulmadı**
**Model:** ECGFounder backbone + tek logitli OMI head, iki aşamalı eğitim

## Yönetici özeti

Fine-tuning fold 0'da yayımlanmış baseline'ı **F1'de geçti** ve protokolün
go/no-go kriterini karşıladı. Ancak sonucu "baseline'dan iyi" diye özetlemek
yanıltıcı olur: model baseline'dan **daha az duyarlı**, kazancı özgüllük ve
PPV'den geliyor.

| Aşama | AUROC | AUPRC | F1 | ACS-poz AUROC | NSTEMI AUROC |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0.782 | 0.205 | — | 0.566/0.590 | 0.590 |
| Lineer prob | 0.844 | 0.340 | 0.357 | 0.612 | 0.565 |
| **Fine-tune** | **0.908** | **0.425** | **0.459** | **0.667** | **0.635** |

## Go/no-go: GEÇTİ

Fold 0, hasta-düzeyi bootstrap %95 GA (2.000 tekrar):

| Metrik | Değer | %95 GA |
|---|---:|---|
| AUROC | 0.9081 | [0.8908, 0.9248] |
| AUPRC | 0.4249 | [0.3610, 0.4957] |
| **F1** | **0.4585** | **[0.4086, 0.5052]** |

F1'in alt sınırı (0.4086) baseline'ın (0.396) üzerinde. Protokolün kriteri
karşılandı.

## ⚠️ Ama duyarlılık baseline'ın altında

| | Sens | Spec | PPV | NPV | F1 |
|---|---:|---:|---:|---:|---:|
| Yayımlanmış baseline | **0.697** | 0.873 | 0.277 | 0.976 | 0.396 |
| **Bu model (fold 0)** | 0.600 | **0.930** | **0.371** | 0.971 | **0.459** |
| EKG uzmanı 1 | 0.277 | 0.972 | 0.407 | 0.951 | 0.330 |
| EKG uzmanı 2 | 0.429 | 0.941 | 0.338 | 0.959 | 0.378 |

Model baseline'dan **daha muhafazakâr**: 100 OMI'nin 60'ını yakalıyor, baseline
70'ini yakalıyordu. Buna karşılık yanlış alarm başına daha isabetli (PPV 0.371
vs 0.277).

**Bu, F1'i optimize etmenin doğrudan sonucudur.** F1 duyarlılık ile precision'a
eşit ağırlık verir; OMI'de ise **kaçırmak fazladan uyarıdan çok daha
tehlikelidir** — kaçırılan oklüzyon reperfüzyonun gecikmesi demektir.

Protokol F1'i birincil ölçüt yapmıştı çünkü değerlendirme platformu binary
tahmin alıp AUPRC vermiyor. Bu ölçüt seçimi, klinik olarak yanlış yöndeki bir
işletim noktasını ödüllendirdi.

**Düzeltme (bir sonraki iterasyon):** eşik seçimi duyarlılık-ağırlıklı olmalı —
örneğin F2 skoru veya "spesifisite ≥0.87'de duyarlılığı maksimize et" kısıtı.
Eşik eğitim fold'larında seçildiği için bu, modeli yeniden eğitmeden yapılabilir.

## Eğitim dinamiği — fine-tuning'in katkısı marjinal

| Aşama | Epoch | Train loss | Val AUPRC |
|---|---:|---:|---:|
| head | 1 | 0.8785 | 0.4023 |
| head | 2 | 0.7847 | 0.4200 |
| head | 3 | 0.7618 | 0.4093 |
| **head** | **4** | 0.7457 | **0.4217** |
| head | 5–8 | 0.7351→0.7268 | 0.411–0.415 |
| **finetune** | **1** | 0.7180 | **0.4249** ← en iyi |
| finetune | 2–5 | 0.6860→0.6068 | 0.407–0.415 |

İki gözlem:

1. **Backbone'u açmak neredeyse hiçbir şey kazandırmadı.** Donuk backbone + head
   0.4217'ye ulaşıyor; son iki stage açıldıktan sonra en iyi değer 0.4249
   (+0.003). Bu fark bootstrap GA'sının genişliği (±0.07) yanında gürültü.
2. **Fine-tune aşamasında train loss düşerken validation AUPRC düşüyor** —
   klasik overfitting. 26,7 milyon parametreyi 921 pozitif örnekle eğitmek
   fazla. Early stopping (patience 4) 5. epoch'ta durdurdu.

**Çıkarım:** ECGFounder'ın temsili zaten yeterince güçlü; asıl kazanç donuk
feature üzerinde iyi bir head eğitmekten geliyor. Daha agresif fine-tuning
muhtemelen daha fazla değil, daha az kazandırır.

## Önceden tanımlı alt gruplar

| Alt grup | n | OMI | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|---:|
| Tümü | 3.592 | 230 | 0.908 | 0.425 | 0.459 |
| **ACS-pozitif** | 526 | 229 | **0.667** | 0.577 | 0.581 |
| STEMI etiketli | 295 | 152 | 0.642 | 0.641 | 0.651 |
| **NSTEMI etiketli** | 231 | 77 | **0.635** | 0.416 | 0.409 |
| Aralık ≤12 sa | 1.098 | 136 | 0.871 | **0.471** | 0.502 |
| Aralık >12 sa | 2.494 | 94 | **0.923** | 0.395 | 0.409 |
| CTO | 190 | 0 | — | — | — |
| Paced | 72 | 5 | 0.812 | 0.540 | 0.444 |
| VF_VT | 194 | 23 | 0.772 | 0.331 | 0.429 |
| Prior_PCI | 273 | 20 | 0.850 | 0.328 | 0.339 |
| Yaş <65 | 1.658 | 105 | 0.919 | 0.465 | 0.498 |
| Yaş ≥65 | 1.934 | 125 | 0.898 | 0.406 | 0.424 |
| Kadın | 1.430 | 53 | 0.931 | 0.432 | 0.444 |
| Erkek | 2.162 | 177 | 0.891 | 0.428 | 0.463 |

### Asıl hedefte ilerleme var ama iş bitmedi

**ACS-pozitif alt küme 0.612 → 0.667**, **NSTEMI 0.565 → 0.635**. Bu, üç adımda
ilk kez anlamlı bir ilerleme — zero-shot ve lineer prob bu grupta neredeyse hiç
kıpırdamamıştı.

Yine de 0.63–0.67, "ACS'li hastalar içinde kim oklüde?" sorusunda hâlâ zayıf.
Genel AUROC 0.908 bu zorluğu gizliyor; kolay negatifler skoru taşıyor.

### Time_Interval deseni sürüyor

Uzun aralıkta AUROC yüksek (0.923), AUPRC düşük (0.395); kısa aralıkta tersi
(0.871 / 0.471). Prevalans etkisi. Model, EKG'nin anjiyografiye yakın çekildiği
akut vakalarda gerçekten daha iyi.

## Model ve eğitim ayrıntıları

- **Mimari:** Net1D backbone (ECGFounder ağırlıkları) + `Dropout(0.2)` +
  `Linear(1024, 1)`. 150 sınıflık projeksiyon atıldı.
- **Aşama 1:** backbone tamamen donuk, yalnızca head (1.025 parametre), AdamW
  lr 1e-3, 8 epoch.
- **Aşama 2:** son 2 stage açıldı (26,7 M eğitilebilir), head lr 1e-3 /
  backbone lr 1e-5, 6 epoch (5'te erken durdu). Erken stage'ler donuk kaldı —
  14 bin kayıt, 10 milyon EKG'nin öğrettiği genel dalga yapısını bozmamalı.
- **Kayıp:** `BCEWithLogitsLoss`, `pos_weight = 14.6` (negatif/pozitif oranı).
  Bu olmadan kayıp "hiç OMI yok" diyerek minimize oluyor.
- **Seçim:** validation AUPRC (AUROC bu prevalansta yanıltıcı olduğunu zaten
  gösterdi), patience 4.
- **Eşik:** yalnızca eğitim fold'larından (0.7184); değerlendirilen fold'a
  bakılmadı.
- **Girdi:** ham 10 s, z-scored — [lineer prob deneyi](gate2-omi-linear-probe-v1.md)
  median beat'in daha kötü olduğunu gösterdiği için.
- **Süre:** ~20 dk (Mac Mini M4, MPS). Sinyaller float16 memmap'ten okundu.

## Kısıtlar

1. **Bu bir validation-fold sonucudur, test seti değil.** Resmî test setine
   henüz gönderim yapılmadı; protokol en fazla iki gönderime izin veriyor.
2. **Duyarlılık klinik olarak yetersiz** (0.600). Eşik stratejisi düzeltilmeden
   test setine gönderim yapılmamalı.
3. **Fine-tuning'in katkısı gürültü seviyesinde.** Bir sonraki iterasyonda
   backbone'u açmak yerine head kapasitesini (MLP, çoklu görev) denemek daha
   mantıklı.
4. **ACS-pozitif alt kümede model hâlâ zayıf** (0.667). Asıl bilimsel problem
   burada duruyor.
5. Tek merkezli veri; dış geçerlilik Gate 4'ün konusu.

## Tekrar üretmek için

```bash
TQDM_DISABLE=1 CORIO_CALIBRATION=0 python scripts/train_omi_finetune.py --tag v1
```

Ağırlıklar `models/omi/omi_finetuned_v1.pt` (gitignored, 124 MB), rapor
`results/omi/finetune_v1_fold0.json`.
