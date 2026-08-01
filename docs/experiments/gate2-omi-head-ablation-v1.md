<!-- Purpose: Gate 2 steps 5-6 — does head capacity, multi-task, or NSTEMI up-weighting help? -->

# Gate 2 — head kapasitesi, multi-task ve NSTEMI ağırlıklandırma ablasyonu (v1)

**Tarih:** 2026-08-01
**Protokol:** [gate2-omi-protocol-v1.md](../plans/gate2-omi-protocol-v1.md) § 5, adım 5–6
**Ölçüm:** fold 0 (n=3.592, 230 OMI) — resmî test setine **dokunulmadı**
**Eşik:** hepsinde F2-maks, yalnızca eğitim fold'larından
  ([düzeltme gerekçesi](gate2-omi-threshold-strategy-v1.md))

## Yönetici özeti

Üç hipotez denendi, biri işe yaradı, biri etkisiz kaldı, biri **geri teptdi**.

| Varyant | Ekleme | Genel AUPRC | NSTEMI sens | Karar |
|---|---|---:|---:|---|
| v1 | (referans, tek lineer head) | 0.4249 | 0.571 | — |
| **v2a** | **MLP head (1024→256→1)** | **0.4494** | **0.597** | ✅ **kabul** |
| v2b | + multi-task (7 yardımcı head) | 0.4471 | 0.571 | ⊘ etkisiz |
| v2c | + NSTEMI-OMI ×3 ağırlık | 0.4311 | **0.416** | ❌ **zararlı** |

**Kazanan v2a.** Basitçe head'e bir gizli katman eklemek, hem genel AUPRC'yi hem
asıl hedef olan NSTEMI duyarlılığını iyileştirdi. Sonraki adımların temeli bu.

## Tam karşılaştırma (fold 0)

| Varyant | AUROC | AUPRC | Sens | Spec | PPV | F1 |
|---|---:|---:|---:|---:|---:|---:|
| v1 @F2 | 0.9081 | 0.4249 | 0.800 | 0.869 | 0.295 | 0.431 |
| **v2a** | **0.9117** | **0.4494** | 0.800 | 0.878 | 0.309 | 0.446 |
| v2b | 0.9114 | 0.4471 | 0.778 | 0.883 | 0.312 | 0.446 |
| v2c | 0.8896 | 0.4311 | 0.648 | 0.918 | 0.351 | 0.455 |

v2a, hasta-düzeyi bootstrap %95 GA ile:

| Metrik | Değer | %95 GA |
|---|---:|---|
| AUROC | 0.9117 | [0.8940, 0.9286] |
| AUPRC | 0.4494 | [0.3813, 0.5200] |
| F1 | 0.4455 | [0.4021, 0.4862] |

Eşik 0.6423 (eğitim fold'larında F2-maks).

## Hipotez 1: head kapasitesi — İŞE YARADI

Tek `Linear(1024, 1)` yerine `1024 → 256 → ReLU → Dropout → 1`.

Gerekçe [fine-tuning deneyinden](gate2-omi-finetune-v1.md) geliyordu: backbone'un
son iki stage'ini açmak neredeyse hiçbir şey kazandırmamıştı (+0.003), dolayısıyla
darboğaz temsilde değil, onu okuyan katmanda olmalıydı.

Doğrulandı: AUPRC 0.4249 → **0.4494** (+0.025), NSTEMI AUROC 0.635 → **0.651**,
NSTEMI duyarlılık 0.571 → **0.597**. Aynı eşik stratejisi, aynı veri, aynı eğitim
programı — tek fark 262 bin ek parametre.

**Ek gözlem:** v2a'nın en iyi epoch'u yine **donuk backbone aşamasında** (head e4).
Fine-tune aşamasında train loss 0.52 → 0.36'ya düşerken validation AUPRC
0.4494'ten 0.4156'ya geriledi — v1'deki overfitting deseni aynen tekrarlandı.
Backbone'u açmak bu veri boyutunda tutarlı biçimde işe yaramıyor.

## Hipotez 2: multi-task — ETKİSİZ

Paylaşılan temsilden 7 yardımcı head: STEMI, NSTEMI, AMI, CTO ve culprit bölge
(LAD/LCX/RCA, 12 segment kolonundan toplandı; LM tüm eğitim setinde 5 vaka
olduğu için dışarıda bırakıldı). Yardımcı kayıp λ=0.3.

Sonuç: AUPRC 0.4494 → 0.4471, NSTEMI duyarlılık 0.597 → 0.571. Fark her iki
yönde de güven aralığının çok içinde — yani **etki yok**, zarar da yok.

Yorum: yardımcı etiketler (özellikle STEMI/NSTEMI) OMI ile zaten güçlü
korelasyonlu; modelin öğrenmesi gereken ek bir yapı sunmuyorlar. Culprit bölge
daha bilgilendirici olabilirdi ama yalnızca OMI pozitiflerinde tanımlı, yani
1.151 kayıtta sinyal, 16.809 kayıtta sıfır — bu dengesizlikte λ=0.3 ile
öğrenilecek çok şey yok.

## Hipotez 3: NSTEMI ağırlıklandırma — GERİ TEPTİ

NSTEMI etiketli OMI kayıtlarına (eğitimde 346 kayıt) kayıpta 3× ağırlık.

**Beklenen:** NSTEMI duyarlılığı artar, genel skor biraz düşer.
**Gerçekleşen:** NSTEMI duyarlılığı **0.571 → 0.416 düştü**, genel AUPRC de
0.4494 → 0.4311 geriledi. Her iki yönden de kayıp.

| | v2b | v2c | Δ |
|---|---:|---:|---:|
| NSTEMI sens | 0.571 | **0.416** | −0.155 |
| NSTEMI F1 | 0.524 | 0.418 | −0.106 |
| NSTEMI AUROC | 0.647 | 0.595 | −0.052 |
| Genel sens | 0.778 | 0.648 | −0.130 |
| Genel spec | 0.883 | 0.918 | +0.035 |

Neden geri teptiğine dair iki olası açıklama, ikisi de doğrulanmadı:

1. **Eşik kayması.** v2c'nin spesifisitesi belirgin yüksek (0.918), yani F2
   eşiği eğitim fold'unda daha yukarıda seçilmiş. Ağırlıklı kayıp skor
   dağılımını değiştiriyor ve seçilen eşik validation'da farklı davranıyor.
2. **Gerçek model bozulması.** AUPRC'nin de düşmesi (0.4311) eşikten bağımsız
   bir gerileme olduğunu gösteriyor — yani sorun sadece işletim noktası değil.

Muhtemelen ikisi birden. NSTEMI-OMI zor örneklerdir; onlara ağırlık vermek
modeli genel olarak daha az ayırt edici hale getirmiş görünüyor. Ağırlığın
düşürülmesi (örn. 1.5×) veya focal loss gibi başka bir yaklaşım denenebilir,
ama **bu haliyle strateji reddedildi**.

## v2a — önceden tanımlı alt gruplar

| Alt grup | n | OMI | Sens | Spec | F1 | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Tümü | 3.592 | 230 | 0.800 | 0.878 | 0.446 | 0.912 |
| ACS-pozitif | 526 | 229 | 0.799 | 0.488 | 0.649 | 0.677 |
| STEMI etiketli | 295 | 152 | 0.901 | 0.301 | 0.704 | 0.654 |
| **NSTEMI etiketli** | 231 | 77 | **0.597** | 0.662 | 0.526 | 0.651 |
| Aralık ≤12 sa | 1.098 | 136 | 0.787 | 0.815 | 0.508 | 0.873 |
| Aralık >12 sa | 2.494 | 94 | 0.819 | 0.903 | 0.380 | 0.927 |
| CTO | 190 | 0 | — | — | — | — |
| Paced | 72 | 5 | 0.600 | 0.896 | 0.400 | 0.890 |
| VF_VT | 194 | 23 | 0.739 | 0.790 | 0.447 | 0.797 |
| Prior_PCI | 273 | 20 | 0.750 | 0.787 | 0.337 | 0.837 |
| Yaş <65 | 1.658 | 105 | 0.829 | 0.885 | 0.469 | 0.923 |
| Yaş ≥65 | 1.934 | 125 | 0.776 | 0.871 | 0.426 | 0.902 |
| Kadın | 1.430 | 53 | 0.811 | 0.922 | 0.424 | 0.931 |
| Erkek | 2.162 | 177 | 0.797 | 0.846 | 0.453 | 0.896 |

Adalet açısından belirgin bir sorun yok: yaş ve cinsiyet alt gruplarında
duyarlılık 0.78–0.83 aralığında.

## Bu adımın asıl mesajı

Üç iterasyon boyunca **NSTEMI duyarlılığı 0.571 → 0.597** oynadı. Yani asıl
klinik hedefte ilerleme var ama küçük; gizli oklüzyonların hâlâ **%40'ı**
kaçıyor.

Genel 0.800 duyarlılık STEMI-OMI'den (0.901) geliyor. Model şu an esas olarak
"ST elevasyonu görünen oklüzyonu" yakalıyor — ki onu klinisyen zaten görüyor.

Mimari denemeleri (head kapasitesi, multi-task, örnek ağırlıklandırma) bu
sınırı aşamadı. Bu, sorunun mimaride değil, **ya veride ya problem tanımında**
olduğuna işaret ediyor:

- NSTEMI-OMI'nin EKG imzası gerçekten zayıf olabilir (klinik literatür de
  bunu destekler — iki EKG uzmanı bu veri setinde 0.277 ve 0.429 duyarlılık
  aldı).
- Veya ek girdi gerekiyor: troponin, semptom süresi, seri EKG.

## Kısıtlar

1. Validation-fold sonucudur; resmî test setine gönderim yapılmadı.
2. Tek fold (fold 0) üzerinde seçim yapıldı; 5-fold CV daha sağlam olurdu ama
   her varyant ~20 dk sürdüğü için ablasyon tek fold'da yapıldı.
3. NSTEMI ağırlığı yalnızca 3.0'da denendi; ara değerler test edilmedi.
4. Tek merkezli veri; dış geçerlilik Gate 4.

## Tekrar üretmek için

```bash
bash scripts/run_omi_ablation.sh
```

Raporlar `results/omi/finetune_v2{a,b,c}_fold0.json`, ağırlıklar
`models/omi/omi_finetuned_v2{a,b,c}.pt` (gitignored).
