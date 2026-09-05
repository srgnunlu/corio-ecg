<!-- Purpose: Phase B step 8 — can the consistency recipe (or plain fine-tuning on digitised pairs) repair ECGFounder's 150-label head, the one the web app serves, after digitisation? -->

# Aşama B — genel tanı kafası: PTB-XL çiftleriyle ince ayar ve tutarlılık (v1)

**Tarih:** 2026-09-05 → 06
**Girdi:** `data/processed/ptbxl-corpus/clean/manifest.csv` (2000 PTB-XL kaydı, hasta başına bir, `clean` render, 1998'i ECGFounder'ın resmî etiket dosyasında). Eğitim strat_fold 1-8 (1608), doğrulama 9-10 (390).
**Etiketler:** ECGFounder'ın 150 çıkışı için resmî PTB-XL hedef vektörleri. Eğitimde ≥20, doğrulamada ≥10 pozitifi olan etiketler: **23 eğitilebilir, 19 puanlanan.** Macro sayılar bu 19 etiket üzerinden.
**Model:** ECGFounder, backbone frozen, yalnız 150-çıkışlı `dense` projeksiyon eğitiliyor (OMI'deki gibi sıfırdan değil, shipped ağırlıklardan). lr 1e-4, AdamW, sabır 4.
**Çıktı:** `results/general/consistency_v1.json`, `results/general/sweeps/ptbxl_v1/` (3 seed × w∈{0,1}, 30 epoch), `results/general/sweeps/variants_v1/` (teacher varyantları + sütun görünümü, 60 epoch)
**Araçlar:** `src/training/consistency_general.py`, `scripts/train_general_consistency.py`

## Özet: kazanç var ama tutarlılık teriminden değil, ince ayarın kendisinden

Üç görünümde ölçüldü: **tiled** (korpusun sakladığı tek parça sinyal), **segment-ensemble** (web app'in yolu: dört sütun ayrı skorlanıp ortalanır) ve **clean** (WFDB sinyali, bozulmamalı).

| Kol | tiled | **segment-ensemble (ürün)** | clean | ürün ayrımı |
|---|---:|---:|---:|---:|
| ECGFounder shipped | 0.769 | 0.864 | 0.877 | 0.27 |
| düz ince ayar, 30 ep (w=0, 3 seed) | 0.858 | 0.891 | 0.907 | 0.42 |
| tutarlılık, 30 ep (w=1, 3 seed) | 0.856 | 0.882 | 0.898 | 0.38 |
| düz ince ayar, 60 ep | 0.864 | 0.890 | 0.910 | 0.42 |
| **sütun görünümüyle ince ayar, 60 ep** | 0.803 | **0.894** | 0.908 | **0.55** |

Dört sonuç:

1. **Digitize çiftlerle ince ayar ürün yolunu 0.864 → 0.89 taşıyor** ve temiz
   sinyali de iyileştiriyor (0.877 → 0.91). Shipped kafa zarar görmüyor; PTB-XL'e
   özgü küçük bir uyarlama temiz tarafta bile kazandırıyor.
2. **Tutarlılık terimi bu kafada işe yaramıyor, hatta hafif zararlı.** 3 seed'de
   w=1 − w=0: tiled −0.002, ürün yolu −0.010, clean −0.009 (seed varyansı ~1e-4,
   yani gerçek). OMI'nin tersi. Neden: OMI'de teacher (v2a'nın temiz logiti) zaten en
   iyi temiz modeldi; burada teacher shipped ECGFounder, ve ince ayar temiz modeli
   onu geçecek kadar iyileştiriyor (0.877 → 0.91). Digitize çıktıyı bayat bir hedefe
   çekmek geri götürüyor. Bunu onarmak için denenen dört varyantın hiçbiri ürün
   yolunda düz ince ayarı geçmedi (bölüm 2).
3. **Ürün yolu, tiled eğitimde 30 epoch'ta takılıyor.** Uzun eğitim tiled ve clean'i
   yükseltirken segment-ensemble 0.891 → 0.890'da kaldı: model tek parça girdide
   eğitiliyor, sütun maskeli girdide servis ediliyor. **Modeli servis edildiği gibi
   eğitmek** (dört sütun görünümü, kayıp sütun logitlerinin ortalamasında) ürün
   yolunu **0.894**'e ve ayrımı 0.42 → **0.55**'e taşıdı. Tiled'ın düşmesi beklenen
   ve önemsiz: o görünüm üründe yok.
4. **Etiket bazında kazanç, klinik olarak anlamlı yerde.** Ürün yolunda (sütun
   görünümü kafası): normal EKG 0.854 → **0.927**, inferior infarkt 0.834 → **0.883**,
   LBBB 0.978 → 0.995, RBBB 0.968 → 0.983, düşük voltaj +0.20, QRS genişlemesi +0.13.
   AF/sinüs taşikardisi zaten tavanda. ⚠️ PVC 0.954 → **0.926** geriledi ve
   nonspesifik intraventriküler iletim −0.04: 2.5 saniyelik sütun görünümü
   çok-vurulu tanıları zorluyor. Üründe PVC'yi 10 s ritim şeridi analizi de
   taşıyor; yine de bu gerileme kabul edilmeden önce ölçülmeli.

## 1. Tarama (3 seed × w∈{0,1}, 30 epoch)

Seed'ler arası fark ~0.0001 (kafa pretrained, lr küçük). w=1 − w=0 üç hücrede de
negatif: ΔAUROC −0.0020 (t p=0.001), ΔAUPRC −0.0044, Δayrım −0.074. Her iki kol
30/30'da hâlâ yükseliyordu.

## 2. Teacher varyantları (tek seed, 60 epoch)

| Varyant | tiled | ürün | clean |
|---|---:|---:|---:|
| referans: düz ince ayar | 0.864 | **0.890** | **0.910** |
| dondurulmuş teacher, w=1 | 0.862 | 0.880 | 0.899 |
| canlı teacher (modelin o anki temiz logiti), w=1 | 0.865 | 0.885 | 0.900 |
| iki aşamalı (w=0 kafasından başla, teacher = o kafa), w=1 | **0.867** | 0.888 | 0.905 |
| dondurulmuş teacher, w=0.3 | 0.864 | 0.886 | 0.905 |
| dondurulmuş teacher, w=0.1 | 0.865 | 0.888 | 0.908 |

En iyi tiled iki aşamalıda (+0.003), o kadar. Ürün yolunda ve clean'de referans
her varyantın üstünde. Tutarlılık terimi genel kafa için, teacher nasıl seçilirse
seçilsin, en iyi ihtimalle nötr. Bu, OMI bulgusuyla çelişmiyor; koşulu netleştiriyor:
**terim, teacher'ın öğrencinin ulaşabileceğinden iyi bir temiz model olduğu durumda
işe yarıyor.** OMI'de öyleydi, burada değil.

## 3. Kısıtlar

- **19 etiket.** 150'nin çoğu PTB-XL'de desteksiz; macro yalnız desteklilerden.
  Sonuç "150 etiket iyileşti" değil, "PTB-XL'in kapsadığı 19 etiket iyileşti".
- **Tek split.** Fold 1-8 / 9-10, hasta düzeyinde ayrık (PTB-XL strat_fold hasta
  bazlı). Seed varyansı önemsiz ama fold varyansı ölçülmedi.
- **`clean` render.** Moderate/hard çiftleriyle denenmedi; OMI'de etki render'dan
  bağımsızdı, burada ölçülmedi.
- **Sütun görünümü PVC'yi geriletiyor** (bölüm 0, madde 4).
- **Ürüne alınırsa Gate 0 kalibrasyonu yeniden fit edilmeli.** Platt katsayıları
  ve eşikler shipped kafanın logitlerine göre (`configs/calibration/ptbxl_fold9_v1.json`).
  Doğrulama fold'ları (9-10) kalibrasyonun fit/audit fold'larıyla çakışıyor; ürün için
  kalibrasyon başka fold'larda ya da iç içe yapılmalı.
- Yalnız `dense` ağırlıkları kaydediliyor (`models/general/...pt`, gitignored);
  backbone shipped checkpoint.

## 4. Sıradaki

1. Sütun görünümü kafasını web app'e **kapalı bayrakla** bağla (kalibrasyon kapalı,
   "araştırma kafası" etiketiyle) → kullanıcının gerçek fotoğraflarında shipped
   kafayla yan yana.
2. PVC gerilemesini ölç: ritim şeridi analiziyle birlikte ürün düzeyinde PVC kararı
   değişiyor mu?
3. Kalibrasyonu yeni kafa için yeniden fit et (fold seçimi düzeltilerek).
4. Moderate/hard PTB-XL çiftleri (korpus derlemesi gerekir; ~2 saat/rejim).

## Tekrar üretmek için

```bash
python scripts/train_general_consistency.py --consistency-weight 0 --epochs 60 \
  --digitised-view columns \
  --output results/general/sweeps/variants_v1/columns_w0_e60.json
python scripts/summarise_consistency_sweep.py results/general/sweeps/ptbxl_v1
```

Bir koşu (60 epoch, 1608 kayıt, backbone frozen) M4'te ~11 dk; sütun görünümü
digitize tarafta 4 kat forward ile ~20 dk. Tiled görünümde eğitim/doğrulama veri
yükleme ~1 dk (2000 WFDB + .npy).
