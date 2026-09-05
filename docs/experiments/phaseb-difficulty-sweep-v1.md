<!-- Purpose: Phase B step 7 — does the consistency-training gain survive when the paper render is degraded (moderate and hard regimes, same recordings)? -->

# Aşama B — zorluk taraması: `moderate` ve `hard` render (v1)

**Tarih:** 2026-09-04 → 2026-09-05
**Girdi:** clean korpusla **aynı 920 kayıt** (460 OMI, 2 fold × 460) üç render rejiminde:
`clean` (200 dpi, yalnız grid), `moderate` (gürültü 25, ±2° dönme, %1 kırpma, 8000 K),
`hard` (150 dpi, gürültü 50, ±8° dönme, %3 kırpma, rastgele grid rengi). Manifestler
`data/processed/omi-corpus/{clean/manifest,moderate/manifest_matched,hard/manifest_matched}.csv`
**Model:** v2a (Gate 2 kazananı), backbone frozen, yalnız head; 2 fold × {w=0, w=1} × 3 seed
**Çıktı:** `results/omi/sweeps/{clean,moderate_matched,hard}/` (12'şer JSON); `results/omi/sweeps/moderate/` eşleşmemiş ilk deneme (bölüm 5)
**Araçlar:** `scripts/build_digitised_corpus.py --records-from [--render-only] [--no-dewarping-retry]`, `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py`

## Özet: kazanç üç rejimde de aynı büyüklükte, 18/18 hücre lehte

Tutarlılık eğitimi v1 yalnız `clean` render'da ölçülmüştü. Soru: render bozulunca
tutarlılık teriminin ablasyona üstünlüğü kalıyor mu? Aynı kayıtlar, aynı fold'lar,
aynı seed'ler:

| Kol (6 hücre) | clean | moderate | hard |
|---|---:|---:|---:|
| v2a başlangıç, AUROC | 0.7590 ±0.001 | 0.7565 ±0.000 | **0.7268** ±0.001 |
| + digitize eğitim (w=0) | 0.8033 ±0.011 | 0.7993 ±0.010 | 0.7853 ±0.008 |
| + tutarlılık terimi (w=1) | **0.8272** ±0.011 | **0.8251** ±0.003 | **0.8097** ±0.009 |
| başlangıç ayrımı | 0.387 | 0.380 | 0.337 |
| w=0 ayrımı | 0.413 ±0.118 | 0.387 ±0.053 | 0.327 ±0.082 |
| w=1 ayrımı | **0.575** ±0.040 | **0.580** ±0.025 | **0.540** ±0.052 |

Eşleştirilmiş fark w=1 − w=0, aynı fold ve seed içinde:

| | ΔAUROC | ΔAUPRC | Δayrım |
|---|---:|---:|---:|
| clean | +0.0239 (5/6, t p=0.020) | +0.0226 (5/6) | +0.162 (5/6) |
| moderate | +0.0258 (6/6, t p=0.003) | +0.0215 (6/6) | +0.193 (6/6, p=0.001) |
| **hard** | **+0.0244 (6/6, t p=0.013, Wilcoxon p=0.031)** | +0.0253 (5/6, p=0.042) | **+0.213 (6/6, p=0.001)** |

Dört sonuç:

1. **Kazanç render'a bağlı değil.** Üç rejimde ΔAUROC +0.024/+0.026/+0.024; render
   bozuldukça küçülmüyor. Ayrımı açma etkisi hard'da en büyük (+0.21).
2. **Hard gerçekten zor; moderate değil.** Eğitimsiz v2a aynı kayıtlarda clean'e göre
   moderate'ta −0.0025, hard'da **−0.032** kaybediyor. Digitizer'ın kendi ölçüleri de
   ancak hard'da bozuluyor (bölüm 2).
3. **Tutarlılık eğitimi hard'ın cezasının yarısını geri alıyor.** Eğitimsiz clean→hard
   açığı −0.032; tutarlılık kolunda aynı açık **−0.0175** (p=0.003). Hard'da toplam
   kazanç +0.083 (0.727 → 0.810), clean'deki +0.068'den büyük: render ne kadar
   bozuksa head'in geri alacağı o kadar çok.
4. **Ayrımı yalnız tutarlılık terimi açıyor.** w=0 üç rejimde de ayrımı yerinde
   bırakıyor (clean +0.03, moderate +0.006, hard −0.01) ama savrularak (±0.05…0.12);
   w=1 üçünde de +0.16…0.21 açıyor ve savrulmuyor. Eşleşmemiş ilk denemede görünen
   "w=0 daraltıyor" bulgusu (bölüm 5) hiçbir eşleşmiş rejimde tekrarlanmadı.

## 1. Hücre düzeyinde sonuç (hard)

| fold | seed | ΔAUROC | ΔAUPRC | Δayrım |
|---|---|---:|---:|---:|
| 0 | 20260802 | +0.0397 | +0.0464 | +0.162 |
| 0 | 20260814 | +0.0406 | +0.0453 | +0.187 |
| 0 | 20260815 | +0.0358 | +0.0453 | +0.121 |
| 1 | 20260802 | +0.0065 | −0.0036 | +0.319 |
| 1 | 20260814 | +0.0106 | +0.0075 | +0.227 |
| 1 | 20260815 | +0.0132 | +0.0109 | +0.264 |

Örüntü üç rejimde aynı: fold 0'da etki büyük (+0.039), fold 1'de küçük (+0.010);
seed içi sapma küçük. Belirsizliğin kaynağı seed değil, veri. Fold 1'de AUROC az
kıpırdarken ayrımın çok açılması (+0.32) dikkat çekici: sıralama zaten iyiyken
terim güveni geri getiriyor.

## 2. Digitizer rejimlere nasıl tepki verdi?

| Rejim (920 kayıt) | Düzen 3×4+1R | Yanlış düzen | Einthoven medyan / p10 | Layout cost medyan / p90 | Algılanan lead medyan |
|---|---:|---:|---:|---:|---:|
| clean | 920 | 0 | 0.996 / 0.950 | 0.049 / 0.237 | 11 |
| moderate | 917 | 3 | 0.994 / 0.933 | 0.067 / 0.306 | 11 |
| **hard** | 845 | **75** | 0.989 / **0.563** | **0.133 / 0.521** | **10** |

Moderate, Open-ECG-Digitizer'ın zaten telafi ettiği bant. Hard'da 75 kayıtta
(%8) düzen yanlış çözülüyor (çoğu 3×4+3R), Einthoven'in alt onda biri 0.56'ya
düşüyor ve layout cost 2.7 katına çıkıyor. Bu, gerçek telefon fotoğrafına
benzeyen ilk rejim. Sıfır digitizasyon başarısızlığı: 920/920.

## 3. Yöntem notu: hard korpus dewarp yeniden denemesi KAPALI derlendi

Digitizer, ilk geçişi zayıf bulduğu görüntülerde (layout cost > 1.2 ya da
algılanan lead < 10) bir dewarping yeniden denemesi çalıştırır. Hard render'da bu
yol patolojik: yalıtılmış ölçümde altı kayıt (17781, 00568, 12337, 16183, 14847,
07803) 19 saniyede swap'ı 13-14 GB büyüttü, `/usr/bin/time -l` tek kayıt için
**99.9 GB** bellek izi gösterdi ve macOS süreci traceback'siz öldürdü. İşlenen
ilk ~48 kayıtta 6 patlama, yani **~%12**. Aynı kayıtlar `enable_dewarping_retry=False`
ile 8 saniyede, sıfır swap ile bitiyor.

İki seçenek vardı: patlayan kayıtları atlamak (korpus "ilk geçişi iyi olan"
kayıtlara kayar, seçilim yanlılığı) ya da yeniden denemeyi kapatmak (920/920
geçer, %12'lik alt küme yalnız ilk geçiş sonucunu alır, hard biraz daha zorlaşır).
İkincisi seçildi. Karşılaştırılabilirlik notu: clean korpusta bu yeniden deneme
kayıtların **%1.1**'inde, moderate'ta **%3.4**'ünde tetiklenmiş olabilir (aynı
eşik koşulları); hard'da hiç çalışmadı. Yani hard, öteki iki rejime göre hafifçe
handikaplı; bu, iddianın aleyhine değil lehine bir kısıt.

Diğer inşa detayları: normal bir kaydın bellek izi bile ~17.5 GB (RSS 8.6 GB,
gerisi Metal/MPS), 24 GB makinede sınırda. Bu yüzden korpus, önce digitizer
yüklü olmadan render geçişi (0.3 GB), sonra her 8 kayıtta bir süreç yeniden
başlatma ve bekçi (kayıt > 60 s ya da swap +12 GB → kill, kayıt `poisoned.txt`'e)
ile derlendi. Retry kapalıyken bekçi hiç tetiklenmedi. 920 kayıt 94 dakika.
Tek render seed'i (`-se 20260802`) tüm görüntülerde; dönme/gürültü çekilişi
görüntüler arasında çeşitlenmemiş olabilir.

## 4. Kısıtlar

- **İki fold.** Altı hücre bağımsız değil, fold düzeyinde n=2; p değerleri
  iyimser. Yayında "2 fold × 3 seed, eşleştirilmiş, aynı kayıtlar, üç rejim" yazılmalı.
- **Hard hâlâ sentetik.** Gerçek telefon fotoğrafı değil; ECG-Image-Kit'in
  bozulmaları. Layout hataları ve Einthoven düşüşü gerçek fotoğrafa yaklaştığını
  gösteriyor ama eşitlemiyor.
- **Hard'da dewarp retry kapalı** (bölüm 3).
- **İşletim noktası raporlanmadı.** Crossfit eşik duyarlılıkları fold'lar arası
  oynak; AUROC/AUPRC/ayrım üzerinden okunmalı.
- Sayılar `phaseb-threshold-and-quality-v1.md`'deki 460'lık pilot alt kümesiyle
  doğrudan karşılaştırılamaz.

## 5. İlk deneme: eşleşmemiş moderate korpus ve ondan çıkan ders

Moderate korpusu ilk kez iki fold'u birlikte örnekleyerek kurdum; clean korpus
fold fold kurulmuştu. 460 pozitif aynı, negatiflerin çoğu farklıydı (920'nin 484'ü
ortak). O örneklemde başlangıç 0.7537, w=0 0.7773 (ayrım 0.367 → **0.286**),
w=1 0.7942; ΔAUROC +0.0169 (6/6). Yön aynıydı ama seviyeler ~0.03 düşük ve w=0
ayrımı daraltıyordu; eşleşmiş korpusta ikisi de kayboldu. Ders: rejimler arası
karşılaştırma **aynı kayıtlar** üzerinde yapılmalı; `--records-from` bunun için
var. Eşleşmemiş sonuçlar `results/omi/sweeps/moderate/` altında duruyor.

## 6. Bu ne anlama geliyor

"Fotoğraf-native OMI" iddiasının tutarlılık ayağı artık üç rejimde, aynı
kayıtlarda, 18/18 hücrede destekli ve etki büyüklüğü render'dan bağımsız.
Hard'da kalan açık (w=1 0.810 vs temiz sinyal ~0.91) hâlâ büyük ve iki parçalı:
digitizer'ın düzen/Einthoven hataları (bölüm 2) ve head'in geri alamadığı kısım.
Sıradaki müdahale ya digitizer tarafında (düzen tanıma, dewarp yolunun onarımı)
ya da head'e görüntü kalitesini girdi olarak vermekte; head'i tek başına daha
fazla eğitmekte değil (24 epoch tekrarı etkisizdi, bkz. `phaseb-unlabelled-scale-v1.md`).

## Tekrar üretmek için

```bash
# görüntüler (digitizer yüklü değil), sonra 8'lik turlarla digitize
python scripts/build_digitised_corpus.py --folds 0 1 --difficulty hard \
  --records-from data/processed/omi-corpus/clean/manifest.csv --render-only
python scripts/build_digitised_corpus.py --folds 0 1 --difficulty hard \
  --records-from data/processed/omi-corpus/clean/manifest.csv --stop-after 8 --no-dewarping-retry
# manifest_matched.csv = hard satırlarının clean kayıt kümesiyle kesişimi
scripts/run_consistency_sweep.sh \
  data/processed/omi-corpus/hard/manifest_matched.csv results/omi/sweeps/hard
python scripts/summarise_consistency_sweep.py results/omi/sweeps/hard
python scripts/summarise_consistency_sweep.py results/omi/sweeps/hard \
  --reference-dir results/omi/sweeps/clean --reference 1.0 --arm 1.0
```

Render geçişi 920 görüntü ~23 dk; digitize ~6 s/kayıt + süreç başına ~20 s
yükleme; 12 koşuluk sweep ~8 dk. Korpus derlerken eşzamanlı ağır iş koşma.
