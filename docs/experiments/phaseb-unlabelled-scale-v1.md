<!-- Purpose: Phase B step 6 — does the consistency term scale with unlabelled PTB-XL pairs, i.e. with ECGs that carry no OMI label at all? -->

# Aşama B — etiketsiz ölçekleme: PTB-XL çiftleri (v1)

**Tarih:** 2026-09-05
**Girdi:** `data/processed/omi-corpus/clean/manifest.csv` (etiketli, 920 kayıt, 2 fold) + `data/processed/ptbxl-corpus/clean/manifest.csv` (**etiketsiz**, 2000 PTB-XL kaydı, hasta başına bir kayıt, seed 20260814, `clean` render)
**Model:** v2a, backbone frozen, yalnız head; w=1, clean ağırlığı 0.5
**Karışım:** her 16 etiketli batch'e ayrı loader'dan 16 etiketsiz satır ekleniyor (sabit oran)
**Çıktı:** `results/omi/sweeps/ptbxl_unlabelled/` (6 JSON); referans `results/omi/sweeps/clean/` w=1 hücreleri
**Araçlar:** `scripts/build_ptbxl_consistency_corpus.py`, `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py --reference-dir`

## Özet: yön doğru, adım küçük

v1 dokümanının 3. bölümü belirsizliğin seed'den değil veriden geldiğini,
dolayısıyla daha çok seed'in değil daha çok kaydın bilgi getireceğini
söylemişti. Tutarlılık terimi etiket istemediği için (hedef, temiz sinyaldeki
teacher logiti) açık bir EKG veri seti bu işi görebilir. 2000 PTB-XL çifti
denendi; etiketli kümenin 4.3 katı.

| Kol (w=1, 6 hücre) | AUROC | AUPRC | Skor ayrımı |
|---|---:|---:|---:|
| Yalnız etiketli (clean sweep) | 0.8272 ±0.011 | 0.8131 ±0.011 | 0.575 ±0.040 |
| **+ 2000 etiketsiz PTB-XL** | **0.8382 ±0.005** | **0.8231 ±0.007** | 0.605 ±0.046 |

Aynı fold ve seed içinde eşleştirilmiş fark:

| fold | seed | ΔAUROC | ΔAUPRC | Δayrım |
|---|---|---:|---:|---:|
| 0 | 20260802 | +0.0091 | +0.0043 | +0.050 |
| 0 | 20260814 | +0.0153 | +0.0055 | +0.060 |
| 0 | 20260815 | +0.0011 | +0.0022 | −0.088 |
| 1 | 20260802 | **+0.0285** | **+0.0348** | +0.098 |
| 1 | 20260814 | +0.0114 | +0.0116 | +0.014 |
| 1 | 20260815 | +0.0003 | +0.0017 | +0.044 |

Ortalama **ΔAUROC +0.0110** (6/6 lehte, eşleştirilmiş t p=0.049, Wilcoxon
p=0.031), ΔAUPRC +0.0100 (6/6, t p=0.11, Wilcoxon p=0.031), Δayrım +0.030
(5/6, anlamsız). Kol içi sapma da düşüyor (AUROC ±0.011 → ±0.005).

Ölçek için: tutarlılık teriminin kendisi ablasyona +0.024 getirmişti; 4.3 kat
daha fazla çift onun yarısını ekliyor. Tek ölçek noktasıyla eğrinin şekli
(doygun mu, lineer mi) bilinemez.

## 1. Kazanç nerede birikiyor?

En zayıf hücre en çok kazanıyor. v1'de tutarlılık kolunun ablasyonun
gerisinde kaldığı tek hücre (fold 1, seed 20260802: −0.005) burada +0.029 ile
en büyük kazancı alıyor ve bu koldaki fold 1 sapması 0.014'ten 0.002'ye
düşüyor. Etiketsiz çiftler ortalamayı biraz, savrulmayı çok düzeltiyor.
Az veriyle eğitilen head'in şansa bağlı kötü inişlerini önlüyor gibi
görünüyor.

## 2. Teacher hedeflerinin dağılımı

PTB-XL temiz sinyalinde v2a'nın logit medyanı **−3.23**, hedeflerin yalnız
**%14'ü** sıfırın üstünde. Yani etiketsiz çiftlerin altıda beşi "bu EKG OMI
değil, fotoğrafında da değil" diyor. Invariance ağırlıkla negatif bölgede
öğreniliyor. Skor ayrımının neredeyse kıpırdamaması (+0.03, anlamsız) buna
uyuyor: pozitiflerin sıkışması bu veriyle açılmıyor, negatiflerin dağılımı
sıkılaşıyor.

Bunun ucuz bir karşı denemesi var: PTB-XL'in SCP kodlarını kullanarak
MI/iskemi kayıtlarından zenginleştirilmiş bir etiketsiz küme kurmak. Etiket
olarak değil, teacher hedeflerinin pozitif bölgeyi de örneklemesi için.

## 3. Eğitim doymamış mıydı? Hayır.

12 epoch'luk bütçede etiketsiz kol 6 hücrenin 5'inde en iyi epoch'u 10-12'de
bulmuştu, bu da bütçenin kısıtlayıcı olabileceğini düşündürdü. 24 epoch'lu
tekrar (`results/omi/sweeps/clean_e24/`, `ptbxl_unlabelled_e24/`) bunu
çürüttü:

| Kol | 12 epoch | 24 epoch | Δ |
|---|---:|---:|---:|
| yalnız etiketli | 0.8272 | 0.8272 | 0.0000 (6/6 birebir aynı) |
| + PTB-XL | 0.8382 | 0.8393 | +0.0011 (2/6, p=0.22) |

Sabır 4 olduğu için koşular zaten en iyi epoch'tan 4 epoch sonra duruyordu;
"12/12" bir tavan değil, sabır penceresinin sonuydu. Tek hücre kıpırdadı
(fold 1 / seed 20260802: en iyi epoch 18, +0.005). Eğitim bütçesi bu deneyin
darboğazı değil; kazancı sınırlayan veri ve teacher dağılımı.

## 4. Kısıtlar

- **Tek ölçek noktası.** 2000; 500/1000/5000 eğrisi olmadan "daha fazla veri
  daha fazla kazanç" denemez.
- **Optimizasyon karışığı.** Karışım kolu adım başına 32 satır görüyor,
  referans 16. Kazancın bir kısmı "daha fazla gradyan sinyali" olabilir, "PTB-XL
  bilgisi" değil. Temiz kontrol: etiketli kümeyi iki kez örnekleyerek 32'lik
  batch. Yapılmadı.
- **Alan kayması.** PTB-XL başka hastane, başka cihaz, ACS zenginleştirmesi yok.
  Teacher dağılımı bunu zaten gösteriyor.
- **Etiketsiz taraf yalnız `clean` render.** Moderate/hard etiketsiz çift
  denenmedi.
- **İki fold.** Fold düzeyinde n=2; p değerleri iyimser. Yayında "2 fold × 3
  seed, eşleştirilmiş" diye yazılmalı.
- Korpus derleme maliyeti: 2000 kayıt ~7 saat duvar süresi (makine bellek
  baskısı altındaydı; boşken ~6.5 s/kayıt, yani ~3.6 saat). Sweep 6 koşu 9 dk.

## 5. Bu ne anlama geliyor

Etiketsiz ölçekleme çalışıyor ama tek başına Aşama B'yi kapatmıyor: 0.827 →
0.838, temiz sinyaldeki ~0.91 hâlâ uzakta. Kazanç kararlılıkta ve negatif
bölgede; pozitif bölgedeki sıkışma bu yolla açılmıyor. Sıradaki mantıklı
adımlar, ucuzdan pahalıya:

1. ~~24 epoch tekrarı~~ — yapıldı, etkisiz (bölüm 3).
2. Batch-boyutu kontrolü (etiketli küme iki kez örneklenerek 32'lik batch).
3. PTB-XL'den MI-zenginleştirilmiş etiketsiz küme (pozitif bölge teacher'ları).
4. Ölçek eğrisi 500/1000/5000 — yalnız 2-3 olumlu çıkarsa.

## Tekrar üretmek için

```bash
python scripts/build_ptbxl_consistency_corpus.py --max-records 2000
SWEEP_WEIGHTS="1.0" scripts/run_consistency_sweep.sh \
  data/processed/omi-corpus/clean/manifest.csv results/omi/sweeps/ptbxl_unlabelled \
  --unlabelled-corpus data/processed/ptbxl-corpus/clean/manifest.csv \
  --unlabelled-batch-size 16
python scripts/summarise_consistency_sweep.py results/omi/sweeps/ptbxl_unlabelled \
  --reference-dir results/omi/sweeps/clean --reference 1.0 --arm 1.0
```
