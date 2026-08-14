<!-- Purpose: Phase B step 3 — does teaching the head that a photograph of an ECG is the same ECG recover the round-trip loss? -->

# Aşama B — tutarlılık eğitimi (v1)

**Tarih:** 2026-08-14
**Girdi:** `data/processed/omi-corpus/clean/manifest.csv` (920 kayıt = 460 clean/digitised çifti, 910 hasta, 2 fold)
**Model:** v2a (Gate 2 kazananı), backbone frozen, yalnız head eğitiliyor
**Çıktı:** `results/omi/consistency_v1*.json`, `results/omi/seed_sweep/*.json`

## Özet: Aşama B'nin ilk gerçek kazancı

Beş başarısız reference-free probe ve iki refute edilmiş ucuz umuttan (eşik
yeniden seçimi, kalite tabanlı abstention) sonra ilk çalışan müdahale.

| | AUROC | AUPRC | Skor ayrımı |
|---|---:|---:|---:|
| v2a başlangıç (digitize sinyalde) | 0.7590 | 0.7559 | 0.3867 |
| + digitize eğitim (ablasyon, w=0) | 0.8033 ±0.011 | 0.7905 ±0.018 | 0.4133 ±0.118 |
| **+ tutarlılık terimi (w=1)** | **0.8272 ±0.011** | **0.8131 ±0.011** | **0.5753 ±0.040** |

12 koşu: 2 kol × 2 fold × 3 seed. Toplam +0.068 AUROC'un kabaca üçte ikisi
modelin digitize sinyali görmesinden, üçte biri tutarlılık teriminden geliyor.

## 1. Kurulum

Her kayıt iki görünümle geliyor: temiz WFDB sinyali ve aynı kaydın kâğıt
render'ından geri digitize edilmiş hali. Kayıp üç terimden oluşuyor:

```
loss = BCE(digitised) + 0.5 · BCE(clean) + 1.0 · (teacher_logit − digitised_logit)²
```

**Teacher logit eğitimden önce donduruluyor.** Bu bir konfor tercihi değil,
zorunluluk: hedef canlı clean çıktısı olsaydı, her şeye aynı logiti veren bir
model tutarlılığı mükemmel sağlar ve hiçbir şeyi ayırt etmezdi. Sabit hedef bu
kaçış yolunu kapatıyor. Ceza logit uzayında, olasılık uzayında değil — 0 ve 1
yakınında olasılık ölçeği doyuyor ve tam da çökmesi ölçülen güvenli çağrılar
orada yaşıyor.

Backbone frozen: Gate 2 ablasyonu çözmenin +0.003 AUPRC getirdiğini, yani bu
hattın kendi gürültü tabanının içinde kaldığını göstermişti.

## 2. Ablasyon: kazanç nereden geliyor?

`--consistency-weight 0` kolu, tutarlılık terimi olmadan aynı veride aynı
eğitimi yapıyor. Aynı fold ve aynı seed içinde eşleştirilmiş fark:

| fold | seed | ΔAUROC | ΔAUPRC | Δayrım |
|---|---|---:|---:|---:|
| 0 | 20260802 | +0.0413 | +0.0403 | +0.153 |
| 0 | 20260814 | +0.0347 | +0.0340 | +0.284 |
| 0 | 20260815 | +0.0377 | +0.0385 | +0.219 |
| 1 | 20260802 | −0.0050 | −0.0075 | +0.149 |
| 1 | 20260814 | +0.0176 | +0.0183 | +0.234 |
| 1 | 20260815 | +0.0173 | +0.0119 | −0.066 |

Ortalama **+0.0239 AUROC** (eşleştirilmiş t: p=0.020; Wilcoxon: p=0.063),
6 karşılaştırmanın 5'i tutarlılık lehine.

Varyansa da bakmak gerekiyor: ablasyonun skor ayrımı **±0.118** ile savruluyor,
tutarlılık kolununki ±0.040. Ablasyon ayrımı şansa bırakıyor; tutarlılık terimi
istikrarlı biçimde açık tutuyor. Aşama B'nin teşhisi ("kayıp kayma değil, skor
sıkışması") hem ortalamada hem varyansta destekleniyor.

## 3. Metodolojik ders: tek koşuya bakmanın maliyeti

İlk fold takası koşusu (fold 1, seed 20260802) tutarlılık kolunu **geride**
gösterdi (−0.005 AUROC) ve bu tek gözlem "üstünlük gösterilemiyor" sonucuna
götürdü. Aynı hücrede diğer iki seed +0.018 ve +0.017 verdi. Tek koşu, işaretini
ters çevirecek kadar gürültülüydü.

Buna karşılık seed varyansı hücre içinde çok küçük (sd 0.0015–0.0178), fold'lar
arası fark büyük (fold 0'da etki +0.038, fold 1'de +0.010). **Belirsizliğin
kaynağı seed değil, veri.** Pratik sonucu: daha fazla seed koşmak bilgi
getirmez, daha fazla veri getirir.

## 4. Kısıtlar

- **İki fold, 460 eğitim kaydı.** Altı eşleştirilmiş fark bağımsız değil; fold
  düzeyinde yalnız 2 bağımsız birim var, o yüzden p değerleri iyimser.
  Yayında "iki fold, üç seed, eşleştirilmiş" diye açıkça yazılmalı.
- **`clean` render rejimi.** Gerçek fotoğraf değil. `moderate`/`hard` rejimlerde
  etki büyüklüğü değişebilir.
- **Sıkışma yarı yarıya kapandı.** Ayrım 0.387 → 0.575; temiz sinyaldeki ~0.817
  hâlâ uzakta.
- **İşletim noktası metriklerine tek başına güvenilmemeli.** Crossfit eşiği
  fold'lar arası çok oynak (başlangıç duyarlılığı fold 0'da 0.4904 / spesifisite
  0.8443, fold 1'de 0.6557 / 0.7078). AUROC ve AUPRC daha sağlam zemin.
- Bu sayılar `phaseb-threshold-and-quality-v1.md`'deki 0.6348/0.6707 ile
  **doğrudan karşılaştırılamaz**: farklı alt küme ve eşik o küme içinde
  seçiliyor. Geçerli olan, aynı validasyon kümesindeki before/after.

## 5. Bu ne anlama geliyor

Tutarlılık eğitimi doğru yön, ama tek başına açığı kapatmıyor. Bölüm 3'ün
bulgusu sıradaki adımı doğrudan söylüyor: darboğaz veri. Tutarlılık kaybı
**etiket gerektirmiyor** — hedef teacher logiti — dolayısıyla ölçeklemek için
OMI etiketli veri değil, yalnızca bol miktarda EKG gerekiyor. PTB-XL (21.8K
kayıt, açık lisans, halihazırda diskte) bu iş için doğru kaynak; MIMIC-IV-ECG'nin
credentialed statüsü ve DUA kısıtları burada gereksiz bir yük.

## Tekrar üretmek için

```bash
python scripts/build_digitised_corpus.py
python scripts/train_omi_consistency.py
python scripts/train_omi_consistency.py --consistency-weight 0 \
  --output results/omi/consistency_v1_ablation.json \
  --save-to models/omi/omi_consistency_v1_ablation.pt
```

Korpus kurulumu bir kereliğine ~60 dk; her eğitim koşusu Mac Mini M4'te ~2 dk.
Seed sweep'in tamamı (8 ek koşu) ~15 dk.
