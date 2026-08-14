<!-- Purpose: Phase B steps 2 and 4 — can a new cutoff, or an abstention gate, recover the paper round-trip loss? -->

# Aşama B — eşik yeniden seçimi ve kalite ayrıştırması (v1)

**Tarih:** 2026-08-02
**Girdi:** `results/omi/roundtrip_pilot_v3_scores.csv` (460 kayıt, 230 OMI, 456 hasta, `clean` render)
**Model:** v2a (Gate 2 kazananı), eşik 0.6423 devralındı
**Çıktı:** `results/omi/roundtrip_threshold_v1.json`, `results/omi/roundtrip_quality_v1.json`

## Özet: pilotun iki ucuz umudu da tutmadı

Pilot iki şey öngörmüştü — (a) eşik kaymış, yeniden seçmek bedava duyarlılık
getirir, (b) kaybın bir kısmı digitizer hatasından gelir, kalite sinyaliyle
ayrıştırılabilir. **İkisi de yanlış çıktı.**

| Öngörü | Sonuç |
|---|---|
| Eşiği yeniden seç → bedava duyarlılık | Açığın yalnız **%22**'sini kapatıyor, üstelik bedava değil |
| Kalite sinyali hasarı öngörür → abstention | Dört sinyalin dördü de **sıfır** öngörü gücü (\|rho\| < 0.06) |

## 1. Eşik yeniden seçimi (adım 2)

### Metodoloji — iki tuzak vardı

**Dairesellik.** Eşiği 460 kayıtta seçip aynı 460 kayıtta raporlamak, bir
tahminden çok bir üst sınır üretir. Bunun yerine **hasta-gruplu cross-fit**:
her hasta bloğu, o bloğu hiç görmemiş bir eşikle skorlanıyor (2 blok × 20
tekrar). Raporlanan her oran fit-dışı. Aynı hasta iki tarafta duramıyor.

**Prevalans.** Pilot alt kümesi %50 OMI, klinik %6.4. F-beta precision üzerinden
prevalansa bağlı; dengeli kümede seçilen eşik sistematik olarak fazla düşük
çıkar. Negatifler `prevalence_weight()` ile **14.62×** ağırlıklandırıldı.
Düzeltmeden önceki sentetik denemede eşik spesifisiteyi 0.44'e düşürüyordu —
yani düzeltme olmasaydı bu deney sahte bir kazanç raporlayacaktı.

Cross-fit'in ne kadar gerekli olduğunu sayı gösteriyor:

| Seçim yöntemi | Digitize duyarlılık | İddia edeceği kazanım |
|---|---:|---:|
| In-sample optimum (dairesel) | 0.7522 | açığın %71'i |
| **Cross-fit (fit-dışı)** | **0.6707** | **açığın %22'si** |

Dairesel yöntem kazanımı **3 kat** abartıyordu.

### Sonuç

| Kol | Eşik | Sens | Spec | PPV\* |
|---|---:|---:|---:|---:|
| Temiz sinyal, devralınan eşik | 0.6423 | 0.8000 | 0.8739 | 0.3026 |
| Temiz sinyal, yeniden seçilmiş | 0.5368 | 0.8441 | 0.8591 | 0.2919 |
| Digitize+segment, devralınan | 0.6423 | 0.6348 | 0.8957 | 0.2938 |
| **Digitize+segment, yeniden seçilmiş** | **0.6048** | **0.6707** | **0.8528** | **0.2395** |

\* PPV %6.4 prevalansta, dengeli alt kümenin kendi değeri değil.

Yeniden seçim duyarlılığı 0.6348 → **0.6707** çıkarıyor (bootstrap %95 GA
0.617–0.738), ama spesifisiteden **0.043** götürüyor. 1000 hastada:

| Eşik | Yakalanan OMI | Kaçan | Boş alarm | Boş alarm / yakalanan |
|---|---:|---:|---:|---:|
| Devralınan | 40.6 | 23.4 | 97.7 | 2.4 |
| Yeniden seçilmiş | 42.9 | 21.1 | 137.8 | 3.2 |

2.3 ek OMI yakalamanın bedeli 40 ek boş alarm. Bu bir takas, bedava kazanç değil.

### Asıl kanıt: kayıp eşikte değil, sıralamada

Üç bağımsız kesit aynı yere çıkıyor:

1. **Aynı prosedür iki kola** — her iki kol da cross-fit ile eşiğini seçtiğinde:
   temiz **0.8441** (spec 0.8591) vs digitize **0.6707** (spec 0.8528).
   Spesifisiteler neredeyse eşit, açık **0.174**. Yani yeniden seçim temiz kola
   da yarıyor; göreli açık kapanmıyor, hatta hafifçe açılıyor.
2. **Eşit yanlış-alarm yükünde** — temiz kolun kendi spesifisitesinde (0.8739)
   digitize kol yalnız **0.6609** duyarlılığa ulaşıyor (temiz 0.8000).
3. **Skor dağılımları kayma değil, sıkışma gösteriyor:**

| Kol | Pozitif medyan | Negatif medyan | Aralık |
|---|---:|---:|---:|
| Temiz | 0.8901 | 0.0726 | 0.817 |
| Digitize+segment | 0.7180 | 0.3166 | 0.401 |

Negatifler 0.07'den 0.32'ye **çıkıyor**, pozitifler 0.89'dan 0.72'ye düşüyor.
Saf bir kayma aradaki mesafeyi korurdu; burada mesafe yarıya iniyor. **Monoton
hiçbir eşik/kalibrasyon bunu geri alamaz** — ayrım gücünün kendisi kaybolmuş.

Bu, pilotun okumasını düzeltiyor: korelasyon 0.855, pilotun "yeniden kalibrasyon
yeter" bandı (≳0.9) ile "model digitize sinyali görmeli" bandı (≲0.65) arasında
kalıyordu. Ölçüm şimdi ikinciye daha yakın olduğunu söylüyor.

## 2. Kalite ayrıştırması (adım 4)

Soru: hasar kötü digitize edilmiş kayıtlarda mı toplanıyor? Eğer öyleyse
abstention (güvenilmezde tahmin üretmeme) kaybın bir kısmını kurtarır.

**Hiçbir kalite sinyali hasarı öngörmüyor** (Spearman, |skor hatası|'na karşı):

| Sinyal | rho | p |
|---|---:|---:|
| Einthoven skoru | −0.013 | 0.79 |
| Layout cost | +0.059 | 0.20 |
| Tespit edilen lead sayısı | +0.050 | 0.28 |
| Ortalama piksel/mm | −0.024 | 0.61 |

Katmanlara bakınca da aynı: ortalama mutlak skor hatası her katmanda 0.17–0.19
bandında. Einthoven'ın **en iyi** çeyreğinde AUROC düşüşü (−0.066) en kötü
çeyreğinden (−0.051) daha büyük — yani sinyal yönü bile tutarsız.

Dahası, digitizer kendini neredeyse hep sağlıklı ilan ediyor: 460 kaydın yalnız
**19'u** Einthoven uyarı eşiğinin altında, **hiçbiri** layout-cost uyarı eşiğinin
üstünde değil. Ama aynı 460 kayıt 0.165 duyarlılık kaybediyor.

**Digitizer'ın kendi kalite raporu, önemli olan hasara kör.** Bu, Aşama 3'te beş
kez tekrarlanan bulgunun altıncısı: kendine-bakan (reference-free) sinyaller
fidelity öngörmüyor.

⚠️ **Bu null sonuç `clean` rejimine koşullu.** Sentetik temiz render'da kalite
zaten baştan yüksek ve varyansı dar; öngörü gücü olmaması kısmen menzil
kısıtlamasından olabilir. Kalite gerçekten değiştiğinde (`moderate`/`hard`)
sinyal ortaya çıkabilir — sıradaki koşu tam da bunu test ediyor.

## 3. Yan bulgu: digitizer deterministik değil

v3 koşusu v2 ile aynı girdileri kullandı. Temiz kol **bit-bit aynı** çıktı
(model skorlaması deterministik), ama digitize kollar oynadı:

| | v2 | v3 | fark |
|---|---:|---:|---:|
| digitised AUROC | 0.7557 | 0.7551 | −0.0006 |
| digitised_segment AUROC | 0.8612 | 0.8637 | +0.0025 |
| digitised_segment sens | 0.6435 | 0.6348 | −0.0087 |

Yani **gürültü tabanımız ~±0.003 AUROC / ±0.009 duyarlılık**. Bundan küçük
etkileri gerçek sayamayız. Kaynak muhtemelen MPS segmentasyonu veya piksel-boyut
araması; kovalanmadı, ama bundan sonraki karşılaştırmalarda akılda tutulmalı.

## Bu ne anlama geliyor

Ucuz kazançlar tükendi. Kalan −0.17 duyarlılık gerçek bir ayrım gücü kaybı ve
üç yoldan biriyle kapanabilir:

1. **Tutarlılık eğitimi** (strateji dokümanı Aşama B/3): modeli clean ve
   re-digitized çiftlerinde aynı olasılığı vermeye zorla. Skor sıkışması tam da
   bunun hedefi — artık en güçlü aday bu.
2. **Digitizasyon kalitesini artır** — ama hangi bozulmanın skoru sıkıştırdığını
   henüz bilmiyoruz; kalite sinyalleri sessiz.
3. **Abstention** — mevcut digitizer tanılarıyla kurulamaz (bu deney). Öğrenilmiş
   bir quality head gerekir ve onu eğitmek için hasar etiketine ihtiyaç var;
   iyi haber, artık kayıt başına hasar ölçümüz (|clean − segment|) var.

## Tekrar üretmek için

```bash
python scripts/evaluate_omi_roundtrip.py --output results/omi/roundtrip_pilot_v3.json
python scripts/select_roundtrip_threshold.py
python scripts/stratify_roundtrip_quality.py
```

İlk komut ~60 dk (render'lar önbellekli), diğer ikisi saniyeler.
