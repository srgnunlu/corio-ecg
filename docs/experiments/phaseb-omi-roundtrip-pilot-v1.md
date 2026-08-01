<!-- Purpose: Phase B pilot — how much of the OMI model survives the paper round-trip? -->

# Aşama B pilotu — OMI modelinin kâğıt round-trip dayanıklılığı (v1)

**Tarih:** 2026-08-01
**Model:** v2a (Gate 2 kazananı, resmî test setinde F1 0.4630)
**Ölçüm:** fold 0'dan dengeli 460 kayıt (230 OMI + 230 negatif)
**Zincir:** temiz WFDB → ECG-Image-Kit render (`clean` zorluk) → Open-ECG-Digitizer → v2a
**Eşik:** 0.6423 (Gate 2'den, değiştirilmedi)

## Sonuç: kayıp ciddi, ama segment-ensemble üçte ikisini geri alıyor

| Kol | AUROC | AUPRC | **Sens** | Spec | F1 |
|---|---:|---:|---:|---:|---:|
| Temiz sinyal | 0.9100 | 0.8964 | **0.8000** | 0.8739 | 0.8307 |
| Digitize, tek parça | 0.7557 | 0.7553 | **0.5609** | 0.8174 | 0.6434 |
| **Digitize + segment-ensemble** | **0.8612** | **0.8487** | **0.6435** | **0.8957** | **0.7363** |

Temiz sinyale göre kayıp:

| | AUROC | AUPRC | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|
| Tek parça | −0.154 | −0.141 | −0.239 | −0.057 | −0.187 |
| **Segment-ensemble** | **−0.049** | **−0.048** | **−0.157** | **+0.022** | **−0.094** |

**Segment-ensemble AUROC kaybını %68 azaltıyor** (−0.154 → −0.049). 460 kaydın
tamamı işlendi, hepsi 3x4+1R olarak tespit edildi, hiçbirinde ensemble
uygulanamama durumu olmadı.

Kayıt-başına skor korelasyonu: tek parça **0.651**, segment-ensemble **0.854**.

### Bu, ilk okumayı revize ediyor

Bu deneyin ilk turunda (segment-ensemble olmadan) korelasyon 0.644 çıkmış ve
"sıralama gerçekten bozuluyor, yeniden kalibrasyon yetmez" sonucuna varılmıştı.
Segment-ensemble ile korelasyon **0.854**'e çıkıyor — yani kaybın büyük kısmı
modelin digitize edilmiş sinyali anlamamasından değil, **kâğıt kolonlarının
zaman-kaydırmalı yapısının tek parça beslenmesinden** geliyordu.

Bu, PTB-XL'de 150-sınıf tanı yolunda görülen etkinin (macro AUROC 0.75 → 0.87)
OMI'de de geçerli olduğunu doğruluyor.

### Kalan kaybın karakteri: eşik, model değil

Segment-ensemble kolunda **spesifisite temiz sinyalden bile yüksek** (0.896 vs
0.874) ama duyarlılık düşük (0.644 vs 0.800). Bu klasik bir kaymış-eşik
imzasıdır: skorlar aşağı kaymış, Gate 2'den devralınan 0.6423 eşiği artık çok
yukarıda kalıyor.

Yani kalan −0.157 duyarlılık kaybının bir kısmı **eşiği digitize edilmiş
dağılımda yeniden seçerek bedavaya geri alınabilir** — ve korelasyon 0.854
olduğu için bu sefer kalibrasyon gerçekten işe yarayacak bölgedeyiz.

## Bu neden önemli

Gate 2, temiz sinyalde yayımlanmış baseline'ı altı metrikte de geçmişti. Bu
pilot gösteriyor ki **o başarı fotoğraf hattına doğrudan taşınmıyor.** Corio'nun
ürün vaadi "telefonla çekilmiş kâğıt EKG'den OMI riski" olduğuna göre, ölçtüğümüz
kayıp doğrudan ürünün kalbinde.

Aynı zamanda [strateji dokümanının](../research/2026-07-06-specialization-strategy.md)
tezini **doğruluyor**: savunulabilir niş clean-signal OMI değil, fotoğraf
dayanıklılığı — çünkü orada gerçek ve büyük bir problem var.

## Skor korelasyonu ne anlatıyor

Bu sayı düzeltmenin türünü belirliyor:

- **≳0.9:** model sıralamayı koruyup ölçeği kaydırıyor → yeniden kalibrasyon ve
  yeni eşik yeter, ucuz iş.
- **≲0.65:** sıralama bozuluyor → modelin digitize edilmiş sinyali eğitim
  sırasında görmesi gerekir.

Tek parça beslemede 0.651 çıktı (kötü haber), segment-ensemble ile **0.854**
(iyi haber). Yani doğru besleme biçimiyle model **eşik ayarıyla kurtarılabilir
bölgede**. Strateji dokümanının tutarlılık-eğitimi planı hâlâ değerli ama artık
tek çare değil; önce ucuz kazançlar var.

## ⚠️ Bu sayılar iyimser

Üç sebepten gerçek dünyada daha kötü olmasını beklemeliyiz:

1. **`clean` zorluk seviyesi kullanıldı** — sentetik, düz, gölgesiz, mükemmel
   hizalı render. Gerçek telefon fotoğrafı `moderate`/`hard` seviyesine daha
   yakın. Bu pilot **en kolay senaryo**.
2. ~~Segment-ensemble kullanılmadı~~ → **eklendi ve kaybın üçte ikisini geri
   aldı** (yukarıdaki tabloya bakın).
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

1. ~~Segment-ensemble'ı OMI yoluna bağla~~ ✅ **YAPILDI** — AUROC kaybı
   −0.154'ten −0.049'a indi, korelasyon 0.651'den 0.854'e çıktı.
2. **Digitize edilmiş dağılımda eşik yeniden seçimi** — artık en ucuz kazanç.
   Segment kolunun spesifisitesi temiz sinyalden yüksek, duyarlılığı düşük:
   eşik kaymış, düşürmek bedava duyarlılık getirir.
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

Süre ~55 dk ilk çalıştırma, ~40 dk sonrakiler (render'lar
`data/processed/omi-roundtrip/images/` altında önbelleklenir). Raporlar:
`results/omi/roundtrip_pilot.json` (segment-ensemble öncesi) ve
`results/omi/roundtrip_pilot_v2.json` (üç kollu, güncel).
