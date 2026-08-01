<!-- Purpose: Session record — Gate 0 through Phase B pilot in one sitting. -->

# Oturum özeti — 2026-08-01: Gate 0 → Gate 2 → Aşama B pilotu

**Başlangıç durumu:** main `0432e0d` (Phase 4 VT/SVT), son commit 2 Temmuz.
Temmuz'da yapılmış iki iş (strateji dokümanı + website) commit edilmemiş
bekliyordu. Strateji dokümanının kapılarından hiçbiri başlamamıştı.

**Bitiş durumu:** main `790de14`, 10 commit ileride, hepsi pushlandı.
Gate 0 ve Gate 2 kapandı, Aşama B başladı. Suite 475 passed / 1 skipped.

## Yapılanlar

### Temizlik
Bir aylık askıda iş main'e alındı: 6 Temmuz strateji dokümanı (`4a8cc54`) ve
`website/` (`b565cdf`, kendi `.git`'i kaldırılıp monorepo klasörü olarak).

### Gate 0 — sınıf-bazlı kalibrasyon (`ec56faa`)
150 ECGFounder head'i tek bir `0.5` eşiği paylaşıyordu. Fold 9'da Platt scaling
ve F1-optimal eşikler öğrenildi, fold 10'da tek sefer audit edildi.

- micro F1 0.5885 → **0.6465**, macro F1 0.3673 → **0.4129**
- Brier −28%, ECE −75%, AUROC değişmedi (Platt monotonik — beklenen)
- Inferior infarkt F1 **0.029 → 0.423**, 1° AV blok **0.049 → 0.377**

İki zorunlu koruma: negatif Platt slope reddi (bozuk head'i ters çevirip
maskeliyordu) ve 0.30 precision tabanı ("her şeye evet de" dejenerasyonunu
engelliyor; tabansız sürümde micro F1 aslında düşüyordu).

⚠️ Kritik tanıların 7/8'i `research_only` çıktı (PTB-XL'de STEMI/VT/tam blok
pozitifi yok) — tiering körlemesine uygulansaydı tam da kaçırılmaması gereken
etiketler susturulacaktı. Muafiyet eklendi, testle kilitlendi.

### Gate 1 — OMI veri erişimi (`3de3d79`)
6 Temmuz'da 404 dönen Figshare DOI açılmış. Veri indirildi (CC0, 1.42 GB,
MD5'ler doğrulandı), hasta-düzeyi split temiz, format WFDB/500 Hz/Corio'nun lead
sırası.

İki kısıt protokolü şekillendirdi: test etiketleri gizli, ve değerlendirme
platformu yalnızca binary tahmin alıp AUPRC vermiyor. Ayrıca strateji
dokümanındaki OMI kırılımı düzeltildi — OMI'lerin %63'ü NSTEMI etiketli
(doküman tersini yazıyordu).

### Gate 2 — Corio-OMI baseline (`4b6ac22` … `a7a2605`)
Ön kayıtlı protokol, dondurulmuş hasta-gruplu 5-fold split, sonra dört adım:

| Adım | AUROC | AUPRC | NSTEMI sens |
|---|---:|---:|---:|
| Zero-shot (16 head) | 0.782 | 0.205 | 0.590 |
| Lineer prob | 0.844 | 0.340 | 0.565 |
| Fine-tune | 0.908 | 0.425 | 0.571 |
| **v2a (MLP head)** | **0.912** | **0.449** | **0.597** |

Yol boyunca üç önemli bulgu:

1. **Genel AUROC yanıltıcı.** Fold 0'ın %85'i ACS'siz hasta ve içinde tek bir
   OMI var; skor kolay negatiflerden geliyor.
2. **F1 yanlış ölçüttü.** Fine-tuning F1'de baseline'ı geçti ama duyarlılığı
   *düşürdü* (0.600 vs 0.697). F2'ye geçildi → 0.800/0.869, ve iso-karşılaştırmalar
   eşikten bağımsız üstünlük gösterdi.
3. **Backbone'u açmak kazandırmıyor** (+0.003), ama **head kapasitesi
   kazandırıyor** (MLP: AUPRC +0.025). Multi-task etkisiz, NSTEMI
   ağırlıklandırma geri teptdi (hedef alt grupta −0.155 duyarlılık).

**Resmî test seti sonucu (gönderim 1/2):**

| | Sens | Spec | PPV | NPV | Acc | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Corio v2a** | **0.7812** | **0.8907** | **0.3289** | **0.9834** | **0.8837** | **0.4630** |
| Baseline CNN | 0.6970 | 0.8730 | 0.2770 | 0.9760 | 0.8610 | 0.3960 |
| EKG uzmanı 1 / 2 | 0.277 / 0.429 | — | — | — | — | 0.330 / 0.378 |

Altı metriğin altısında da üstün. Fold 0 → test genellemesi güven aralığı
içinde, yani eşiği validation'a bakmadan seçmenin karşılığı alındı.

### Aşama B pilotu (`5cbfe4b`, `4216fbc`)
460 dengeli kayıt render → digitize → skorla:

| Kol | AUROC | Sens |
|---|---:|---:|
| Temiz sinyal | 0.910 | 0.800 |
| Digitize, tek parça | 0.756 | 0.561 |
| **Digitize + segment-ensemble** | **0.861** | **0.644** |

Segment-ensemble AUROC kaybını %68 azalttı ve skor korelasyonunu 0.651'den
**0.854**'e çıkardı — yani kayıp modelin digitize sinyali anlamamasından değil,
kâğıt kolonlarının tek parça beslenmesinden geliyordu. Kalan açık eşik
kaymasına benziyor (spesifisite temiz sinyalden yüksek, duyarlılık düşük).

## Açık kalan asıl sorun

Üç Gate 2 iterasyonunda **NSTEMI duyarlılığı 0.571 → 0.597** oynadı. Genel 0.800
duyarlılık STEMI-OMI'den geliyor (0.901) — klinisyenin zaten gördüğü grup.
Mimari değişiklikleri bu sınırı aşamadı, ki bu sorunun veride veya problem
tanımında olduğuna işaret ediyor (aynı test setinde iki uzman 0.277 ve 0.429
duyarlılık aldı).

Ayrıca resmî test setinde alt grup ölçülemiyor, yani **"gizli oklüzyonu
yakalıyor" iddiası kanıtlanmadı**. Desteklenen iddia: "baseline'dan ve iki
uzmandan iyi bir OMI sınıflayıcı".

## Sıradaki adımlar

1. Digitize dağılımında eşik yeniden seçimi (en ucuz kazanç, eşik kaymış)
2. Zorluk taraması — `moderate`/`hard` render, gerçek fotoğrafa yaklaşma
3. Digitizasyon kalitesine göre ayrıştırma (Einthoven / layout cost)
4. Tutarlılık eğitimi: clean ↔ re-digitized aynı olasılık
5. Quality head / abstention

Test setine ikinci gönderim kotası duruyor; kullanmak için yeni gerekçe gerekir.

## Üretilen kalıcı dosyalar

- `src/calibration/`, `src/omi/` (dataset, split, features, model, training,
  evaluation, threshold, signal_cache)
- `configs/calibration/ptbxl_fold9_v1.json`, `configs/omi/omi_folds_v1.csv` (donmuş)
- 8 deney dokümanı `docs/experiments/`, 1 protokol `docs/plans/`
- Testler: 475 passed / 1 skipped (oturum başında 378)
