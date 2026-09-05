<!-- Purpose: Phase B step 7 — does the consistency-training gain survive when the paper render is degraded (moderate, then hard)? -->

# Aşama B — zorluk taraması: `moderate` render (v1)

**Tarih:** 2026-09-04 (eşleşmemiş korpus), 2026-09-05 (eşleşmiş korpus)
**Girdi:** `data/processed/omi-corpus/moderate/manifest_matched.csv` — clean korpusla **aynı 920 kayıt** (460 OMI, 2 fold × 460), ECG-Image-Kit `moderate`: gürültü 25, ±2° dönme, %1 kırpma, 8000 K renk sıcaklığı
**Model:** v2a (Gate 2 kazananı), backbone frozen, yalnız head
**Çıktı:** `results/omi/sweeps/moderate_matched/` (12 JSON, asıl sonuç); `results/omi/sweeps/moderate/` (eşleşmemiş ilk deneme, bölüm 4); karşılaştırma `results/omi/sweeps/clean/`
**Araçlar:** `scripts/build_digitised_corpus.py --records-from`, `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py`

## Özet: kazanç `moderate`'ta tamamen korunuyor

Tutarlılık eğitimi v1 yalnız `clean` render üzerinde ölçülmüştü. Soru: render
bozulunca tutarlılık teriminin ablasyona üstünlüğü kalıyor mu, yoksa temiz
render'ın artefaktı mıydı? Aynı 920 kayıt, aynı fold'lar, aynı seed'ler:

| Kol (6 hücre) | clean AUROC | moderate AUROC | clean ayrım | moderate ayrım |
|---|---:|---:|---:|---:|
| v2a başlangıç (digitize sinyalde) | 0.7590 ±0.001 | 0.7565 ±0.000 | 0.387 | 0.380 |
| + digitize eğitim (w=0) | 0.8033 ±0.011 | 0.7993 ±0.010 | 0.413 ±0.118 | 0.387 ±0.053 |
| + tutarlılık terimi (w=1) | **0.8272** ±0.011 | **0.8251** ±0.003 | 0.575 ±0.040 | **0.580** ±0.025 |

Eşleştirilmiş fark w=1 − w=0, aynı fold ve seed içinde:

| | ΔAUROC | ΔAUPRC | Δayrım |
|---|---:|---:|---:|
| clean | +0.0239 (5/6, t p=0.020) | +0.0226 (5/6) | +0.162 (5/6) |
| **moderate (eşleşmiş)** | **+0.0258 (6/6, t p=0.003, Wilcoxon p=0.031)** | **+0.0215 (6/6, p=0.017)** | **+0.193 (6/6, p=0.001)** |

Üç sonuç:

1. **Kazanç render'a bağlı bir artefakt değil.** Moderate'ta fark clean'dekinden
   küçük değil, hatta biraz daha büyük ve daha tutarlı (6/6, kol içi sapma
   ±0.003). Tutarlılık terimi bozuk render'da da skor ayrımını ~0.19 açıyor.
2. **Moderate render digitizer'ı neredeyse hiç zorlamıyor.** Eğitimsiz v2a aynı
   kayıtlarda clean-digitize'de 0.7590, moderate-digitize'de 0.7565: fark
   **0.0025**. Eğitilmiş kollar arasındaki fark da bunun içinde (w=1: −0.002,
   anlamsız). ±2° dönme ve gürültü 25, Open-ECG-Digitizer'ın zaten telafi
   ettiği bant. Bu yüzden moderate, "gerçek fotoğrafa yaklaşma" iddiası için
   zayıf bir test; `hard` (gürültü 50, ±8°, rastgele grid rengi, 150 dpi) asıl
   sınav ve kuyrukta (`results/omi/sweeps/hard/`).
3. **Düz digitize eğitimi ayrımı ne açıyor ne kapatıyor.** w=0 her iki
   rejimde ayrımı yerinde bırakıyor (clean +0.03, moderate +0.006) ama
   savrularak (±0.118, ±0.053). Ayrımı istikrarlı biçimde açan tek şey
   tutarlılık terimi. Eşleşmemiş ilk denemede görünen "w=0 ayrımı daraltıyor"
   bulgusu (bölüm 4) eşleşmiş örneklemde **tekrarlanmadı**; o, negatif
   örnekleminin bir özelliğiydi, yöntemin değil.

## 1. Hücre düzeyinde sonuç (moderate, eşleşmiş)

| fold | seed | ΔAUROC | ΔAUPRC | Δayrım |
|---|---|---:|---:|---:|
| 0 | 20260802 | +0.0417 | +0.0397 | +0.204 |
| 0 | 20260814 | +0.0269 | +0.0331 | +0.241 |
| 0 | 20260815 | +0.0373 | +0.0318 | +0.260 |
| 1 | 20260802 | +0.0185 | +0.0078 | +0.208 |
| 1 | 20260814 | +0.0184 | +0.0108 | +0.103 |
| 1 | 20260815 | +0.0122 | +0.0059 | +0.143 |

v1'deki örüntü tekrar ediyor: fold 0'da etki büyük (+0.035), fold 1'de küçük
(+0.016); seed içi sapma küçük. Belirsizliğin kaynağı yine seed değil, veri.

## 2. Bu ne anlama geliyor

Tutarlılık eğitimi render bozulmasına dayanıklı; en azından digitizer'ın
telafi ettiği bozulma bandında. Bu, yöntemin kendisi için iyi haber ve
"fotoğraf-native OMI" iddiasının bir parçası olarak yayınlanabilir: **aynı
kayıtlarda, aynı seed'lerle, iki render rejiminde de 6/6 hücre.**

Ama Aşama B'nin asıl sorusu olan "gerçek telefon fotoğrafında ne olur?" hâlâ
açık. Moderate o soruyu sormuyor. Hard soracak; hard'da digitizer'ın kendi
hatası devreye girecek ve orada head'in tek başına ne kadar toparlayabileceği
ilk kez ölçülecek. Temiz sinyaldeki ~0.91'e karşı clean/moderate'ta kapanan
açık kabaca %45; gerisi digitizer tarafında.

## 3. Kısıtlar

- **Tek render seed'i.** Her görüntü ECG-Image-Kit'e aynı `-se 20260802` ile
  verildi; dönme açısı ve gürültü çekilişi görüntüler arasında çeşitlenmemiş
  olabilir. Korpus çeşitliliği açısından bir kısıt; yönü değiştirmesi
  beklenmiyor ama hard için farklı seed'lerle bir kontrol yapılmalı.
- **İki fold.** Altı hücre bağımsız değil, fold düzeyinde n=2; p değerleri
  iyimser. Yayında "2 fold × 3 seed, eşleştirilmiş, aynı kayıtlar" yazılmalı.
- **İşletim noktası raporlanmadı.** Crossfit eşik duyarlılıkları fold'lar arası
  çok oynak (v1'deki gibi); AUROC/AUPRC/ayrım üzerinden okunmalı.
- Sayılar `phaseb-threshold-and-quality-v1.md`'deki 460'lık pilot alt
  kümesiyle doğrudan karşılaştırılamaz.

## 4. İlk deneme: eşleşmemiş korpus ve ondan çıkan ders

Moderate korpusu ilk kez iki fold'u birlikte örnekleyerek kurdum; clean
korpus fold fold kurulmuştu. Sonuç: 460 pozitif aynı, negatiflerin çoğu farklı
(920 kaydın 484'ü ortak; fold büyüklükleri 480/440). O örneklemde:

| Kol | AUROC | ayrım |
|---|---:|---:|
| başlangıç | 0.7537 | 0.367 |
| w=0 | 0.7773 | **0.286** |
| w=1 | 0.7942 | 0.484 |

Eşleştirilmiş ΔAUROC +0.0169 (6/6, p=0.012), Δayrım +0.197 (6/6). Yön aynı,
ama iki şey farklıydı: mutlak seviyeler ~0.03 düşük ve w=0 ayrımı daraltıyordu.
Eşleşmiş korpusta ikisi de kayboldu. Yani negatif örneklemi hem seviyeyi hem
de "w=0 ne yapar" sorusunun cevabını değiştirebiliyor; tutarlılık teriminin
kazancı ise değişmiyor. Ders: rejimler arası karşılaştırma **aynı kayıtlar**
üzerinde yapılmalı, `--records-from` bunun için var. Eşleşmemiş sonuçlar
`results/omi/sweeps/moderate/` altında duruyor.

## 5. Sıradaki

1. Hard eşleşmiş sweep (zincir 4; ilk derleme ~15 kayıtta sessizce öldü,
   yeniden başlatıldı).
2. Hard için farklı render seed'leriyle çeşitlilik kontrolü.
3. Etiketsiz PTB-XL ölçeklemesi: `phaseb-unlabelled-scale-v1.md`.

## Tekrar üretmek için

```bash
python scripts/build_digitised_corpus.py --folds 0 1 --difficulty moderate \
  --records-from data/processed/omi-corpus/clean/manifest.csv
# manifest_matched.csv = moderate satırlarının clean kayıt kümesiyle kesişimi
scripts/run_consistency_sweep.sh \
  data/processed/omi-corpus/moderate/manifest_matched.csv results/omi/sweeps/moderate_matched
python scripts/summarise_consistency_sweep.py results/omi/sweeps/moderate_matched
python scripts/summarise_consistency_sweep.py results/omi/sweeps/moderate_matched \
  --reference-dir results/omi/sweeps/clean --reference 1.0 --arm 1.0
```

Korpus derlemesi ~7 s/kayıt (M4, makine boşken; eşzamanlı ağır iş swap'a
sokup 100 s/kayıta düşürüyor). 12 koşuluk sweep ~8 dk.
