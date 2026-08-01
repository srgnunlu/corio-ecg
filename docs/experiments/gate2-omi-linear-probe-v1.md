<!-- Purpose: Gate 2 step 2 — how much OMI signal is already in ECGFounder's frozen representation? -->

# Gate 2 — donuk feature üzerinde lineer prob (v1)

**Tarih:** 2026-08-01
**Protokol:** [gate2-omi-protocol-v1.md](../plans/gate2-omi-protocol-v1.md) § 5, adım 2 ve 4
**Ölçüm:** fold 0 (n=3.592, 230 OMI) — resmî test setine dokunulmadı
**Yöntem:** ECGFounder backbone tamamen donuk; 1024-boyutlu havuzlanmış temsil üzerinde
lojistik regresyon (`class_weight="balanced"`), eşik **yalnızca eğitim fold'larında** seçildi

## Yönetici özeti

Backbone OMI'yi kısmen kodluyor ve bunu 150 sınıflık head'lerden **çok daha iyi**
taşıyor. Ama tek başına yayımlanmış baseline'ı geçmiyor — ki beklenen buydu:
bu adımın amacı temsilin taşıdığı bilgiyi ölçmek, rekabet etmek değil.

| Aşama | AUROC | AUPRC | ACS-poz AUROC |
|---|---:|---:|---:|
| Zero-shot (16 head ortalaması) | 0.782 | 0.205 | 0.566 / 0.590 |
| **Lineer prob (ham 10 s)** | **0.844** | **0.340** | **0.612** |
| Lineer prob (median beat) | 0.809 | 0.248 | 0.577 |

AUPRC'nin 0.205 → 0.340'a çıkması (prevalans 0.064'ün 5.3 katı) temsilin
head'lerin okuyamadığı bilgi taşıdığını gösteriyor.

## Sonuçlar — ham 10 s girdi

Fold 0, hasta-düzeyi bootstrap %95 GA (2.000 tekrar):

| Metrik | Değer | %95 GA |
|---|---:|---|
| AUROC | 0.8435 | [0.8168, 0.8692] |
| AUPRC | 0.3401 | [0.2823, 0.4114] |
| **F1** | **0.3570** | **[0.3077, 0.4034]** |
| Sensitivite | 0.5130 | — |
| Spesifisite | 0.9069 | — |
| PPV | 0.2738 | — |
| NPV | 0.9646 | — |

**Go/no-go: HAYIR.** F1'in %95 GA'sı yayımlanmış baseline'ı (0.396) içeriyor ve
alt sınır onun altında. Protokol gereği bu bir geçiş sayılmaz.

Bu beklenen bir sonuç — donuk bir temsil üzerinde tek katmanlı lineer sınıflayıcı
ile 5-fold CV yapılmış tam eğitilmiş bir CNN'i geçmek sürpriz olurdu. Sinyalin
var olduğunu gösteriyor; onu çıkarmak fine-tuning'in işi.

## Girdi karşılaştırması: ham 10 s vs median beat

Protokolün 4. adımı. Yayımlanmış baseline median beat kullanmıştı; ECGFounder
10 s üzerinde eğitildi.

| | AUROC | AUPRC | F1 | ACS-poz AUROC |
|---|---:|---:|---:|---:|
| **Ham 10 s** | **0.8435** | **0.3401** | **0.3570** | **0.6115** |
| Median beat (tile edilmiş) | 0.8091 | 0.2476 | 0.3406 | 0.5774 |

**Ham 10 s her metrikte kazanıyor**, AUPRC'de fark belirgin (+%37 göreli).
Muhtemel sebep: median beat ECGFounder için dağılım dışı bir girdi — 1 saniyelik
beat 10 kez tekrarlanıyor, yani ritim bilgisi sahte. Backbone bu biçimi hiç
görmedi.

**Karar: sonraki tüm deneyler ham 10 s ile.** Median beat yalnızca yayımlanmış
baseline'la mimari karşılaştırması gerekirse geri gelir.

## ⚠️ Time_Interval: AUROC ve AUPRC ters yönde konuşuyor

| Alt grup | n | OMI | Prevalans | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Aralık ≤12 sa | 1.098 | 136 | 12.4% | 0.790 | **0.414** |
| Aralık >12 sa | 2.494 | 94 | 3.8% | **0.882** | 0.270 |

Uzun aralıkta AUROC daha yüksek ama AUPRC belirgin şekilde daha düşük. Bu bir
çelişki değil, düşük prevalansın AUROC'yi şişirmesi: uzun aralık grubunda kolay
negatif bol.

**AUPRC doğruyu söylüyor:** model, EKG'nin anjiyografiye yakın çekildiği akut
vakalarda daha iyi çalışıyor (0.414 vs 0.270). Klinik olarak beklenen yön — ve
Gate 1'de işaret ettiğimiz confounder'ın gerçek olduğunun kanıtı. Bu, yalnızca
AUROC raporlamanın neden yasaklandığının somut örneği.

## Diğer önceden tanımlı alt gruplar (ham 10 s)

| Alt grup | n | OMI | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|---:|
| ACS-pozitif | 526 | 229 | 0.612 | 0.549 | 0.515 |
| STEMI etiketli | 295 | 152 | 0.603 | 0.617 | 0.572 |
| **NSTEMI etiketli** | 231 | 77 | **0.565** | 0.393 | 0.400 |
| CTO | 190 | **0** | — | — | — |
| Paced | 72 | 5 | 0.791 | 0.439 | 0.500 |
| VF_VT | 194 | 23 | 0.735 | 0.371 | 0.366 |
| Prior_PCI | 273 | 20 | 0.752 | 0.217 | 0.222 |
| Yaş <65 | 1.658 | 105 | 0.868 | 0.346 | 0.353 |
| Yaş ≥65 | 1.934 | 125 | 0.825 | 0.342 | 0.361 |
| Kadın | 1.430 | 53 | 0.874 | 0.293 | 0.333 |
| Erkek | 2.162 | 177 | 0.824 | 0.357 | 0.365 |

Notlar:

- **NSTEMI grubu hâlâ en zayıf halka** (AUROC 0.565). Zero-shot'ta 0.590'dı;
  lineer prob burada neredeyse hiç kazanç sağlamadı. Klinik olarak en değerli
  grup bu — gizli oklüzyon — ve iş asıl orada.
- **CTO alt grubunda sıfır OMI.** Kronik total oklüzyon ile akut oklüzyon
  birbirini dışlıyor; veri bunu doğruluyor. Bu alt grup ölçülemez, protokolde
  kalması yine de doğru (yokluğun kendisi bir bulgu).
- Yaş ve cinsiyet alt gruplarında AUPRC farkı küçük — belirgin bir adalet sorunu
  görünmüyor, ama kadınlarda prevalans düşük (3.7% vs 8.2%) olduğu için AUROC
  farkı yine prevalans etkisi.

## Veri kalitesi notu

17.960 geliştirme kaydının **2'si** WFDB olarak çözülemedi (`14262.dat`,
`03228.dat` — "Samples were not loaded correctly"). Makalenin bildirdiği %99.99
uyumla tutarlı (2/19.955 ≈ %0.01). Bu kayıtlar hem feature çıkarımında hem
değerlendirmede dışlanıyor; `extract_features` bir geçerlilik maskesi döndürüyor.

## Sıradaki adım

Protokolün 3. adımı: yeni binary OMI head ile fine-tuning — önce backbone donuk,
sonra son stage'ler kademeli açılarak. Hedef, genel skoru değil **ACS-pozitif
alt kümedeki 0.61'i ve özellikle NSTEMI'deki 0.565'i** yükseltmek.

## Tekrar üretmek için

```bash
CORIO_CALIBRATION=0 python scripts/train_omi_linear_probe.py --input-mode raw
CORIO_CALIBRATION=0 python scripts/train_omi_linear_probe.py --input-mode median
```

Feature'lar `results/omi/features/` altında önbelleklenir (backbone donuk olduğu
için tekrar hesaplanmaları gerekmez). İlk çalıştırma ~90 s, sonrakiler saniyeler.
