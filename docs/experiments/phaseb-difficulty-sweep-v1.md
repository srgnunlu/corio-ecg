<!-- Purpose: Phase B step 7 — does the consistency-training gain survive when the paper render is degraded (moderate regime)? -->

# Aşama B — zorluk taraması: `moderate` render (v1)

**Tarih:** 2026-09-04
**Girdi:** `data/processed/omi-corpus/moderate/manifest.csv` (920 kayıt, 460 OMI, 2 fold; ECG-Image-Kit `moderate`: gürültü 25, ±2° dönme, %1 kırpma, 8000 K renk sıcaklığı)
**Model:** v2a (Gate 2 kazananı), backbone frozen, yalnız head
**Çıktı:** `results/omi/sweeps/moderate/` (12 JSON), karşılaştırma için `results/omi/sweeps/clean/` (v1 seed sweep'in 12 hücresi tek dizine toplandı)
**Araçlar:** `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py`

## Özet: kazanç `moderate`'ta yaşıyor, mekanizma daha da netleşiyor

Tutarlılık eğitimi v1 yalnız `clean` render üzerinde ölçülmüştü. Soru şuydu:
render bozulunca tutarlılık teriminin ablasyona üstünlüğü kalıyor mu, yoksa
kazanç temiz render'ın bir artefaktı mıydı?

| Kol | clean AUROC | moderate AUROC | clean ayrım | moderate ayrım |
|---|---:|---:|---:|---:|
| v2a başlangıç (digitize sinyalde) | 0.7590 ±0.001 | 0.7537 ±0.005 | 0.387 | 0.367 |
| + digitize eğitim (w=0) | 0.8033 ±0.011 | 0.7773 ±0.005 | 0.413 ±0.118 | **0.286** ±0.015 |
| + tutarlılık terimi (w=1) | **0.8272** ±0.011 | **0.7942** ±0.016 | 0.575 ±0.040 | **0.484** ±0.042 |

Her hücre 6 koşu (2 fold × 3 seed). Eşleştirilmiş fark w=1 − w=0:

| | ΔAUROC | ΔAUPRC | Δayrım |
|---|---:|---:|---:|
| clean | +0.0239 (5/6, t p=0.020) | +0.0226 | +0.162 (5/6) |
| **moderate** | **+0.0169 (6/6, t p=0.012, Wilcoxon p=0.031)** | +0.0118 (5/6) | **+0.197 (6/6, t p<0.001)** |

Üç şey söylüyor:

1. **Kazanç render'a bağlı bir artefakt değil.** Moderate'ta ortalama fark
   küçülüyor (+0.024 → +0.017) ama altı hücrenin altısında da tutarlılık lehine
   ve varyansı daha düşük.
2. **Düz digitize eğitimi skor ayrımını daraltıyor.** Clean'de w=0 ayrımı
   yerinde bırakıyordu (0.387 → 0.413, ±0.118 ile savrularak); moderate'ta
   düpedüz daraltıyor (0.367 → **0.286**), AUROC'u iyileştirirken. Yani model
   sıralamayı öğrenirken güvenini kaybediyor. Tutarlılık terimi aynı veride
   ayrımı 0.484'e açıyor. Aşama B'nin "kayıp kayma değil sıkışma" teşhisi
   burada en keskin hâlini alıyor: sıkışma, düz fine-tuning'in yan etkisi
   olarak da ortaya çıkıyor ve yalnız tutarlılık terimi onu geri alıyor.
3. **Mutlak toparlanma moderate'ta daha az.** Temiz sinyaldeki ~0.91 AUROC'a
   göre clean kolda açığın kabaca %45'i, moderate kolda %26'sı kapanıyor. Render
   bozuldukça head'in tek başına yapabileceği azalıyor; kalan açık digitizer
   tarafında.

## 1. Hücre düzeyinde sonuç (moderate)

| fold | seed | ΔAUROC | ΔAUPRC | Δayrım |
|---|---|---:|---:|---:|
| 0 | 20260802 | +0.0245 | +0.0210 | +0.304 |
| 0 | 20260814 | +0.0298 | +0.0214 | +0.158 |
| 0 | 20260815 | +0.0256 | +0.0215 | +0.159 |
| 1 | 20260802 | +0.0083 | +0.0073 | +0.182 |
| 1 | 20260814 | +0.0072 | +0.0003 | +0.206 |
| 1 | 20260815 | +0.0059 | −0.0005 | +0.175 |

v1'deki örüntü tekrar ediyor: fold 0'da etki büyük (+0.027), fold 1'de küçük
(+0.007); seed içi sapma küçük. Belirsizliğin kaynağı yine seed değil, veri.
Fold başlangıçları: fold 0 doğrulama (480 kayıt) AUROC 0.7578 / AUPRC 0.7536,
fold 1 doğrulama (440 kayıt) 0.7495 / 0.7269.

## 2. Beklenmedik bulgu: moderate render digitizer'ı neredeyse hiç zorlamıyor

Eğitimsiz v2a'nın moderate-digitize sinyaldeki AUROC'u 0.754, clean-digitize'de
0.759: fark 0.005. Digitizer'ın kendi kalite ölçüleri de aynı şeyi söylüyor
(Einthoven medyanı 0.994 vs 0.996, 920/920 kayıt 3×4+1R olarak çözüldü, sıfır
başarısızlık). ±2° dönme ve gürültü 25, Open-ECG-Digitizer'ın halihazırda
telafi ettiği bir bant. Bu, "gerçek fotoğrafa yaklaşma" iddiası için
moderate'ın zayıf bir test olduğu anlamına geliyor; `hard` (gürültü 50, ±8°,
rastgele grid rengi, 150 dpi) asıl sınav.

## 3. Kısıtlar

- **Korpus eşleşmiyor.** Clean korpus fold fold örneklenmişti; moderate'ı iki
  fold'u birlikte örnekleyerek kurdum. 460 pozitif aynı, negatiflerin çoğu
  farklı (920 kaydın yalnız 484'ü ortak; fold büyüklükleri 480/440 vs 460/460).
  Kolların **kendi içindeki** eşleştirilmiş karşılaştırma geçerli; clean ile
  moderate arasındaki mutlak farklar yaklaşık. Düzeltme kuyrukta: derleyiciye
  `--records-from` eklendi, clean kayıt kümesi üzerinde eşleşmiş moderate ve
  hard korpusları gece derleniyor (`results/omi/sweeps/moderate_matched/`,
  `results/omi/sweeps/hard/`).
- **Tek render seed'i.** Her görüntü ECG-Image-Kit'e aynı `-se 20260802` ile
  verildi; dönme açısı ve gürültü çekilişi görüntüler arasında çeşitlenmemiş
  olabilir. Korpus çeşitliliği açısından bir kısıt; sonuçların yönünü
  değiştirmesi beklenmiyor.
- **İki fold.** Altı hücre bağımsız değil, fold düzeyinde n=2; p değerleri
  iyimser. Yayında "2 fold × 3 seed, eşleştirilmiş" diye yazılmalı.
- **İşletim noktası raporlanmadı.** Crossfit eşik duyarlılıkları fold'lar arası
  çok oynak (v1'deki gibi); AUROC/AUPRC/ayrım üzerinden okunmalı.
- `clean` ve `moderate` sayıları `phaseb-consistency-training-v1.md` ile aynı
  hücreler; `phaseb-threshold-and-quality-v1.md`'deki 460'lık pilot alt
  kümesiyle doğrudan karşılaştırılamaz.

## 4. Sıradaki

1. Eşleşmiş moderate + hard sweep'leri (gece zinciri) — mutlak farkları
   düzeltir ve gerçek fotoğrafa daha yakın rejimde kazancın kalıp kalmadığını
   söyler.
2. PTB-XL etiketsiz ölçekleme (`phaseb-unlabelled-scale-v1.md`, aynı zincir).
   Uyarı: PTB-XL'de v2a'nın teacher logitlerinin yalnız %8'i pozitif; etiketsiz
   çiftler ağırlıkla negatif bölgede invariance öğretecek.
3. Kalan açık digitizer tarafında ise (moderate'ta %26 toparlanma), sonraki
   müdahale head'de değil digitize sinyalin kendisinde aranmalı.

## Tekrar üretmek için

```bash
python scripts/build_digitised_corpus.py --folds 0 1 --difficulty moderate
scripts/run_consistency_sweep.sh \
  data/processed/omi-corpus/moderate/manifest.csv results/omi/sweeps/moderate
python scripts/summarise_consistency_sweep.py results/omi/sweeps/moderate
python scripts/summarise_consistency_sweep.py results/omi/sweeps/clean
```

Korpus derlemesi ~8 s/kayıt (M4, makine boşken; eşzamanlı ağır iş swap'a
sokup 100 s/kayıta düşürüyor). 12 koşuluk sweep ~7 dk.
