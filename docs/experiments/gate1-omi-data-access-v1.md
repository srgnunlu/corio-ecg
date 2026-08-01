<!-- Purpose: Gate 1 verdict — can the Chongqing OMI dataset actually carry the Corio-OMI programme? -->

# Gate 1 — OMI veri erişimi ve kullanılabilirlik denetimi (v1)

**Tarih:** 2026-08-01
**Kaynak plan:** `docs/research/2026-07-06-specialization-strategy.md` § 5, "Gate 1 — OMI veri erişimi"
**Veri:** Chongqing ACS/OMI, DOI [10.6084/m9.figshare.29925314](https://doi.org/10.6084/m9.figshare.29925314)
**Makale:** Du X, Liu Y, Wang L, He W, Bin G, Deng G, Yang J. *Scientific Data* 2026;13:1009.
[10.1038/s41597-026-07278-0](https://doi.org/10.1038/s41597-026-07278-0) — açık erişim, PMID 42082497

## Karar: GEÇTİ, iki ciddi kısıtla

Veri gerçek, temiz, lisansı serbest ve hasta-düzeyi ayrımı sağlam. Gate 2 başlayabilir.
Ancak iki kısıt protokolü doğrudan şekillendiriyor:

1. **Test etiketleri gizli.** Değerlendirme yalnızca yazarların online platformu üzerinden.
2. **Platform sadece binary tahmin kabul ediyor** — olasılık değil. Yani test setinde
   **AUROC/AUPRC hesaplanamıyor.** Strateji dokümanının "birincil metrik OMI AUPRC"
   hedefi test setinde uygulanamaz; sadece kendi validation split'imizde geçerli.

## Strateji dokümanının altı sorusu

| # | Soru | Cevap |
|---|---|---|
| 1 | Veri gerçekten indirilebiliyor mu? | **Evet.** 3 arşiv, 1.42 GB, MD5'ler doğrulandı. |
| 2 | Lisans ne diyor? | **CC0 1.0.** Fine-tuning, ağırlık paylaşımı, ticari kullanım — hepsi serbest. Embargo/gating yok. |
| 3 | Aynı hastanın EKG'leri train/test arasında ayrılmış mı? | **Evet.** Hasta kesişimi **0** (train 17.018 hasta, test 1.891 hasta). Makale sadece "randomly divided" diyor ama veri hasta-düzeyi bölme yapıldığını gösteriyor. |
| 4 | Train içinde validation split kurmak için hasta ID var mı? | **Evet.** `Patient_id` kolonu mevcut. Train'de 855 hastanın >1 EKG'si var (max 5), yani gruplu bölme **zorunlu**. |
| 5 | Gizli test sunucusu erişilebilir mi? | **Evet, ayakta** (`http://39.105.59.221:8080/`, HTTP 200). Ama ⚠️ HTTP (şifresiz), submission limiti belirtilmemiş. |
| 6 | Baseline kodu ve ağırlıklar açık mı? | **HAYIR — kısmen.** Repo [catcatgirl/validation-code](https://github.com/catcatgirl/validation-code) yalnızca 8.7 KB'lık veri-doğrulama scripti içeriyor (sinyal uzunluğu/örnekleme histogramları). **Model kodu ve ağırlıklar yok**, repo **lisanssız**. Makale kendi girişinde de "no open-access OMI model with public weights currently exists" diyor. |

## Veri karakteristikleri (train.csv, n=17.960)

| Etiket | Pozitif | Oran |
|---|---:|---:|
| UA | 6.213 | 34.6% |
| PCI | 4.541 | 25.3% |
| AMI | 2.679 | 14.9% |
| STEMI | 1.442 | 8.0% |
| Prior_PCI | 1.323 | 7.4% |
| NSTEMI | 1.235 | 6.9% |
| **OMI** | **1.151** | **6.4%** |
| CTO | 957 | 5.3% |
| VF_VT | 869 | 4.8% |
| Paced | 338 | 1.9% |

Yaş 64.9 ± 11.0 (19–99), kadın %39.8. Eksik değer yok, tüm etiketler 0/1.

### ⚠️ Strateji dokümanında düzeltilmesi gereken sayı

Doküman "1.274 OMI (807 STEMI-OMI + 467 NSTEMI-OMI)" diyordu. Gerçek kırılım **tersi**:

| | STEMI=1 | NSTEMI=1 | İkisi de 0 |
|---|---:|---:|---:|
| **OMI=1** | 423 | **727** | 1 |
| **OMI=0** | **812** | 715 | 15.282 |

İki yönlü okuma, ve ikisi de projenin tezini güçlendiriyor:

- **OMI'lerin %63'ü NSTEMI etiketli** — yani ST elevasyonu olmayan gizli oklüzyonlar çoğunluk.
- **STEMI'lerin %66'sı OMI değil** (812/1235) — ST elevasyonu oklüzyonun zayıf bir vekili.

Bu tam olarak OMI paradigmasının varlık sebebi ve modelin katma değerinin nerede olduğunu
gösteriyor: kaçırılan oklüzyonu NSTEMI grubunda yakalamak.

### ⚠️ Time_Interval bir confounder

EKG ile anjiyografi arasındaki süre (dakika):

| | Medyan | Q1–Q3 | Maks |
|---|---:|---:|---:|
| Tüm kayıtlar | 1.607 (~27 sa) | 438–3.943 | 10.079 (tam 7 gün) |
| OMI=1 | **474 (~8 sa)** | 221–1.775 | — |
| OMI=0 | 1.664 | 483–4.023 | — |

**Kayıtların %56'sında EKG, anjiyografiden 24 saatten uzun süre önce çekilmiş.** Etiket
anjiyografiden geliyor; o EKG oklüzyonu henüz göstermiyor olabilir. OMI pozitiflerde sürenin
belirgin biçimde kısa olması (akut vakalar hızlı katetere gidiyor) **modelin zamanı bir
kısayol olarak öğrenmesi riskini** doğuruyor — model EKG'den değil, prevalans yapısından
öğrenebilir.

**Gate 2'de zorunlu:** Time_Interval'a göre önceden tanımlanmış alt grup analizi, ve
duyarlılık analizi olarak kısa aralıklı (örn. ≤12 sa) kohortta ayrı eğitim/değerlendirme.

### Culprit damar dağılımı (OMI=1, n=1.151)

PLAD 313 · PRCA 298 · MRCA 177 · PLCX 158 · DRCA 153 · MLAD 128 · MLCX 99 · DLCX 84 ·
DB 38 · DLAD 14 · OM 13 · LM 5. Altı OMI kaydında culprit etiketi yok.

## Aşılması gereken eşik: yayımlanmış baseline

Makalenin baseline'ı **median waveform** girdisiyle çok katmanlı bir CNN; train'de 5-fold CV,
sonra aynı hiperparametrelerle test:

| | Sens | Spec | PPV | NPV | Acc | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline CNN** | 0.697 | 0.873 | 0.277 | 0.976 | 0.861 | **0.396** |
| EKG uzmanı 1 | 0.277 | 0.972 | 0.407 | 0.951 | 0.927 | 0.330 |
| EKG uzmanı 2 | 0.429 | 0.941 | 0.338 | 0.959 | 0.908 | 0.378 |

**Dikkat çekici:** iki uzman da baseline'ın altında kaldı ve duyarlılıkları çok düşük
(0.277 ve 0.429). OMI'yi gözle EKG'den yakalamak gerçekten zor — modelin klinik değer
önerisi bu boşlukta.

## Veri formatı — Corio ile uyum (doğrulandı)

İndirilen 1.42 GB açıldı, üç arşivin de MD5'i doğrulandı. `row_data/` 39.910 dosya
(19.955 × `.dat` + `.hea`), `med_data/` 19.955 dosya.

**Ham kayıt (WFDB):**

| Özellik | Değer | Corio ile uyum |
|---|---|---|
| Örnekleme | 500 Hz | ✅ birebir |
| Süre / örnek | 10 s / 5000 | ✅ ECGFounder'ın beklediği uzunluk |
| Boyut | (5000, 12) | ✅ |
| Birim | mV | ✅ |
| **Lead sırası** | I, II, III, aVR, aVL, aVF, V1–V6 | ✅ **birebir aynı — reorder gerekmez** |

Bu, PTB-XL hattında kanlı canlı yaşadığımız lead-reorder ve resample sorunlarının burada
hiç çıkmayacağı anlamına geliyor: veri doğrudan `ECGDiagnoser`'a verilebilir.

**Median dosyası (`.med`):** WFDB değil, ham binary. 12.000 bayt = 6.000 `int16` (little
endian) = **12 lead × 500 örnek**, yani 1 saniyelik median beat. Düzen **interleaved**
(lead-genlik profili ham sinyalle r=0.910 uyum verirken blok düzeni r=0.024 veriyor).
Ölçek **µV** — mV'ye çevirmek için 1000'e bölünür.

```python
median = np.fromfile(path, dtype="<i2").reshape(-1, 12).T / 1000.0  # (12, 500) mV
```

## Teknik doğrulama (makaleden)

- Etiketler anjiyografi + taburculuk tanısından; kıdemli EKG uzmanı çift-kör gözden geçirmiş,
  **%100 uyum** bildirilmiş.
- Sinyallerin %99.99'u 10 s, tamamı 500 Hz. Format **WFDB** (`.dat` + `.hea`) — Corio
  pipeline'ıyla doğrudan uyumlu.
- V2/V3 genlik aralıkları geniş (−1.72…1.18 mV, −1.49…1.50 mV).

**Makalenin kendi kısıtı:** OMI tanımı TIMI 0–1 ile sınırlı. Anjiyografi öncesi spontan
reperfüze olmuş (TIMI 2–3 + yüksek troponin) vakalar OMI sayılmamış — yani gerçek OMI'lerin
bir kısmı negatif etiketli.

## Küçük tutarsızlıklar (not)

- Makale toplam UA'yı 6.197 diyor; yalnız train.csv'de 6.213 var. STEMI (1.513) ve NSTEMI
  (1.274) toplamları ise train+test ile tutarlı.
- Makale metninde bir yerde "19.995", başka yerde "19.955" kayıt geçiyor.

## Gate 2 için doğrudan çıkarımlar

1. **Kendi validation split'imizi hasta-düzeyinde kurmalıyız** (`Patient_id` ile gruplu),
   çünkü test etiketleri yok. Model seçimi orada yapılacak.
2. **Test setine gönderim sayısını önceden sınırlamalıyız.** Platform limit koymuyor; limiti
   kendimiz koymazsak test seti bir validation setine dönüşür ve sonuç yayımlanamaz hale gelir.
   Öneri: protokol önceden yazılır, **tek gönderim**.
3. **Eşik seçimi kendi validation'ımızda yapılmalı** — platform binary istediği için eşiği
   biz belirliyoruz. Gate 0'daki precision-tabanlı eşik seçimi burada doğrudan işe yarar.
4. **Birincil karşılaştırma F1 üzerinden** olmalı (baseline 0.396), çünkü platform AUPRC
   vermiyor. AUROC/AUPRC yalnızca iç validation'da raporlanabilir.
5. **Median waveform da mevcut** ve baseline onu kullanmış; ham 10 s ile median beat
   karşılaştırması ilk deneylerden biri olmalı.
6. **Fotoğraf hattı için:** veri WFDB/500 Hz, yani mevcut ECG-Image-Kit render → digitize
   döngüsü doğrudan uygulanabilir. Corio'nun asıl farklılaşması burada başlıyor.

## Üretilen dosyalar

- `scripts/download_omi_dataset.py` — MD5-doğrulamalı indirme (`--csv-only` ile 0.3 MB)
- `data/raw/omi-chongqing/` — veri (gitignored)
