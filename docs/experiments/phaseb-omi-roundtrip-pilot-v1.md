<!-- Purpose: Phase B pilot — how much of the OMI model survives the paper round-trip? -->

# Aşama B pilotu — OMI modelinin kâğıt round-trip dayanıklılığı (v1)

**Tarih:** 2026-08-01
**Model:** v2a (Gate 2 kazananı, resmî test setinde F1 0.4630)
**Ölçüm:** fold 0'dan dengeli 460 kayıt (230 OMI + 230 negatif)
**Zincir:** temiz WFDB → ECG-Image-Kit render (`clean` zorluk) → Open-ECG-Digitizer → v2a
**Eşik:** 0.6423 (Gate 2'den, değiştirilmedi)

## Sonuç: kayıp ciddi

| Kol | AUROC | AUPRC | **Sens** | Spec | F1 |
|---|---:|---:|---:|---:|---:|
| Temiz sinyal | 0.9100 | 0.8964 | **0.8000** | 0.8739 | 0.8307 |
| Digitize edilmiş | 0.7559 | 0.7579 | **0.5435** | 0.8087 | 0.6266 |
| **Δ** | **−0.154** | **−0.139** | **−0.257** | −0.065 | **−0.204** |

**Duyarlılık 0.800'den 0.544'e düşüyor** — kâğıttan geçen her dört oklüzyondan
birini daha kaybediyoruz. 460 kaydın tamamı başarıyla işlendi, yani bu bir
"başarısız digitizasyon" sorunu değil; sessizce bozulan sinyalin sorunu.

Kayıt-başına skor korelasyonu (temiz vs digitize): **0.644**.

## Bu neden önemli

Gate 2, temiz sinyalde yayımlanmış baseline'ı altı metrikte de geçmişti. Bu
pilot gösteriyor ki **o başarı fotoğraf hattına doğrudan taşınmıyor.** Corio'nun
ürün vaadi "telefonla çekilmiş kâğıt EKG'den OMI riski" olduğuna göre, ölçtüğümüz
kayıp doğrudan ürünün kalbinde.

Aynı zamanda [strateji dokümanının](../research/2026-07-06-specialization-strategy.md)
tezini **doğruluyor**: savunulabilir niş clean-signal OMI değil, fotoğraf
dayanıklılığı — çünkü orada gerçek ve büyük bir problem var.

## Skor korelasyonu 0.644 ne anlatıyor

Bu sayı, düzeltmenin hangi türden olması gerektiğini söylüyor:

- **Korelasyon yüksek olsaydı (≳0.9):** model sıralamayı koruyup yalnızca
  ölçeği kaydırıyor demekti; çözüm ucuz olurdu — digitize edilmiş dağılım
  üzerinde yeniden kalibrasyon ve yeni bir eşik.
- **Gerçekleşen (0.644):** sıralama gerçekten bozuluyor. Yeniden kalibrasyon
  tek başına yetmez; modelin digitize edilmiş sinyali görmesi gerekiyor.

Yani strateji dokümanının Aşama B planı (clean / render / re-digitized
olasılıklarını tutarlı olmaya zorlayan eğitim) gerekli görünüyor — kestirme yok.

## ⚠️ Bu sayılar iyimser

Üç sebepten gerçek dünyada daha kötü olmasını beklemeliyiz:

1. **`clean` zorluk seviyesi kullanıldı** — sentetik, düz, gölgesiz, mükemmel
   hizalı render. Gerçek telefon fotoğrafı `moderate`/`hard` seviyesine daha
   yakın. Bu pilot **en kolay senaryo**.
2. **Segment-ensemble kullanılmadı.** Üretim tanı yolu kâğıt kolonlarını ayrı
   ayrı skorlayıp ortalıyor (PTB-XL'de macro AUROC 0.75→0.87 kazandırmıştı);
   bu pilot digitize edilmiş sinyali tek parça olarak modele verdi. Kaybın bir
   kısmı buradan geri alınabilir — **ilk denenmesi gereken şey bu**.
3. **Digitizasyon uyarıları gözlendi.** Çalışma sırasında birçok kayıtta
   "Einthoven consistency low — lead assignment may be incorrect" ve
   "digitization quality concerns" uyarıları çıktı. Lead ataması bozulduğunda
   kaybın bir kısmı modele değil digitizer'a aittir; bu ayrıştırılmadı.

## Metodoloji notu

Alt küme **dengeli** (%50 OMI), doğal %6.4 değil. Sebep: pilot boyutunda doğal
prevalansla duyarlılık düşüşünü ölçecek kadar pozitif kalmıyor. Prevalans iki
kolda da aynı olduğu için **round-trip deltası geçerli**; ancak mutlak PPV ve
AUPRC değerleri fold 0 veya test seti rakamlarıyla karşılaştırılamaz.

Eşik Gate 2'den olduğu gibi alındı (0.6423). Digitize edilmiş dağılıma göre
yeniden seçilmiş bir eşik duyarlılık kaybının bir kısmını geri alabilir — ama
korelasyon 0.644 olduğu için tamamını değil.

## Sıradaki adımlar (öncelik sırasıyla)

1. **Segment-ensemble'ı OMI yoluna bağla ve pilotu tekrarla.** En ucuz kazanç
   adayı; üretim tanı yolunda zaten kanıtlanmış.
2. **Digitize edilmiş dağılımda eşik yeniden seçimi** — ucuz, ama kısmi.
3. **Zorluk seviyesi taraması** (`moderate`, `hard`) — kaybın gerçek fotoğrafta
   ne kadar büyüdüğünü ölç.
4. **Digitizasyon kalitesine göre ayrıştırma** — Einthoven skoru / layout cost
   düşük olanları ayır; kayıp digitizer hatasından mı yoksa genel bozulmadan mı?
5. **Tutarlılık eğitimi** (strateji dokümanı Aşama B): modeli clean ve
   re-digitized çiftlerinde aynı olasılığı vermeye zorla.
6. **Quality head / abstention** — güvenilmez digitizasyonda tahmin üretme.

## Tekrar üretmek için

```bash
TQDM_DISABLE=1 CORIO_CALIBRATION=0 python scripts/evaluate_omi_roundtrip.py
```

Süre ~55 dk (M4). Render'lar `data/processed/omi-roundtrip/images/` altında
önbelleklenir, tekrar çalıştırmada yeniden üretilmez. Rapor
`results/omi/roundtrip_pilot.json`.
