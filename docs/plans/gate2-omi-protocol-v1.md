<!-- Purpose: Pre-registered evaluation protocol for the Corio-OMI clean-signal baseline (Gate 2). -->

# Gate 2 — OMI baseline değerlendirme protokolü (v1, ÖN KAYIT)

**Tarih:** 2026-08-01
**Durum:** Model eğitimi başlamadan önce sabitlendi. Sonuçlara bakıp değiştirilmez.
**Önkoşullar:** [Gate 0](../experiments/gate0-calibration-v1.md) ✅ · [Gate 1](../experiments/gate1-omi-data-access-v1.md) ✅

Bu belge, sonuçları gördükten sonra hedefi kaydırmayı (HARKing) engellemek için
yazıldı. Metrikler, alt gruplar ve gönderim kuralları burada; sonuçlar ayrı bir
deney dokümanına yazılacak.

## 1. Split disiplini

| Küme | Kaynak | Kayıt | Kullanım |
|---|---|---:|---|
| Geliştirme | `CSV/train.csv` | 17.960 | 5-fold, hasta-gruplu, OMI-stratifiye |
| **Fold 0** | dondurulmuş | 3.592 | **Birincil validation** — model ve eşik seçimi |
| Fold 1–4 | dondurulmuş | 14.368 | Eğitim |
| Resmî test | `CSV/test.csv` | 1.995 | **Etiketler gizli.** Yalnızca final gönderim. |

Fold ataması `configs/omi/omi_folds_v1.csv` dosyasında **donduruldu**
(seed 20260801, `StratifiedGroupKFold`). Doğrulandı: hiçbir hasta birden fazla
fold'a düşmüyor; fold başına OMI oranı 0.0640–0.0643.

`scripts/build_omi_split.py` mevcut atamanın üzerine `--force` olmadan yazmayı
reddeder — split'i değiştirmek tüm önceki sonuçları karşılaştırılamaz kılar.

## 2. Test setine gönderim kuralı (bağlayıcı)

Yazarların platformu (`http://39.105.59.221:8080/`) **gönderim limiti koymuyor.**
Limiti kendimiz koyuyoruz:

> **Resmî test setine en fazla İKİ gönderim yapılacak.** Biri ana model, biri
> önceden ilan edilmiş bir yedek (örn. ham-sinyal yerine median-beat girdisi).
> Her gönderim öncesi model, eşik ve girdi biçimi bu dokümanda yazılı olacak.

Limit koymazsak test seti fiilen bir validation setine dönüşür ve sonuç
yayımlanamaz hale gelir. Ara denemeler **yalnızca fold 0'da** yapılır.

⚠️ Platform HTTP (şifresiz) üzerinden çalışıyor. Gönderilen dosya yalnızca kayıt
kimlikleri ve 0/1 tahminler içerir — hasta verisi içermez.

## 3. Metrikler

Platform **yalnızca binary tahmin** kabul ediyor ve Sens/Spec/PPV/NPV/Acc/F1
döndürüyor. **AUROC ve AUPRC test setinde hesaplanamaz.** Bu yüzden:

**Birincil sonuç ölçütü (test seti):** OMI için **F1**.
Aşılacak eşik — yayımlanmış baseline:

| | Sens | Spec | PPV | NPV | Acc | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.697 | 0.873 | 0.277 | 0.976 | 0.861 | **0.396** |
| EKG uzmanı 1 | 0.277 | 0.972 | 0.407 | 0.951 | 0.927 | 0.330 |
| EKG uzmanı 2 | 0.429 | 0.941 | 0.338 | 0.959 | 0.908 | 0.378 |

**Go/no-go:** Fold 0'da bootstrap %95 GA'sı baseline F1 0.396'yı içermeyecek
şekilde üstte olmalı. Yalnızca nokta tahmini yeterli değil.

**İç validation'da (fold 0) ayrıca raporlanacak:** AUROC, AUPRC, Brier, ECE ve
kalibrasyon eğrisi. Bunlar test setinde ölçülemediği için yalnızca iç
karşılaştırma ve eşik seçimi amaçlıdır.

**Eşik seçimi:** fold 0'da, Gate 0'daki precision-tabanlı F1 optimizasyonuyla
(`src/calibration/fit.py::select_threshold`). Test setinde eşik aranmaz.

**Belirsizlik:** tüm nokta tahminleri için 2.000 tekrarlı bootstrap %95 GA,
hasta düzeyinde yeniden örnekleme (aynı hastanın kayıtları birlikte).

## 4. Önceden tanımlı alt gruplar

Zero-shot ölçüm ([sonuçlar](../experiments/gate2-omi-zeroshot-v1.md)) kritik bir
yapıyı ortaya çıkardı: fold 0'daki 3.592 kaydın **3.066'sında hiç ACS yok ve
yalnızca 1 OMI var.** Yani genel AUROC büyük ölçüde "ACS var mı yok mu"
ayrımından geliyor; asıl zor soru ACS'li hastalar içinde kimin oklüde olduğu.

Bu yüzden **her sonuç iki katmanda** raporlanacak:

1. **Tüm test seti** — yayımlanmış baseline ile karşılaştırılabilirlik için.
2. **ACS-pozitif alt küme** (STEMI veya NSTEMI etiketli) — modelin gerçek klinik
   katma değeri. Zero-shot'ta bu alt kümede AUROC yalnızca 0.57–0.59'du.

Ek zorunlu alt gruplar (hepsi önceden tanımlı, hiçbiri sonuç görüldükten sonra
eklenmeyecek):

| Alt grup | Gerekçe |
|---|---|
| **NSTEMI-OMI** | OMI'lerin %63'ü burada — gizli oklüzyon, asıl klinik hedef |
| **STEMI-OMI** | Görece kolay grup; kolay/zor ayrımını göstermek için |
| **Time_Interval ≤12 sa / >12 sa** | Kayıtların %56'sında EKG anjiyografiden >24 sa önce; model zamanı kısayol olarak öğrenebilir |
| CTO | Kronik oklüzyon akut OMI'yi taklit edebilir |
| Paced | Uyarılmış ritim ST analizini bozar |
| VF_VT | Ağır hemodinamik bozukluk |
| Prior_PCI | Eski infarkt morfolojisi karıştırır |
| Yaş (<65 / ≥65), cinsiyet | Standart adalet analizi |

## 5. Modelleme planı (Aşama A — temiz sinyal)

Sıra bağlayıcı değil, ama her adım fold 0'da ölçülüp kaydedilecek:

1. **Zero-shot referans** ✅ tamamlandı — eğitimsiz ECGFounder, fold 0'da en iyi
   kombinasyon AUROC 0.782 / AUPRC 0.205 (prevalans 0.064).
2. **Lineer prob:** ECGFounder'ın 1024-boyutlu havuzlanmış feature'ı üzerinde
   lojistik regresyon. Backbone tamamen donuk. Ucuz ve güçlü bir referans.
3. **Head fine-tuning:** yeni binary OMI head, backbone donuk, sonra son
   stage'ler kademeli açılır.
4. **Girdi karşılaştırması:** ham 10 s sinyal **vs** median beat. Yayımlanmış
   baseline median beat kullanmış; ECGFounder 10 s bekliyor. İkisi de ölçülecek.
5. **Küçük 1D ResNet** sıfırdan — foundation model gerçekten gerekli mi?
6. **Multi-task yardımcı head'ler:** OMI + STEMI/NSTEMI + culprit bölge + CTO.
   Yalnızca fold 0'da OMI'yi iyileştirirse tutulur.

Aşama B (fotoğraf dayanıklılığı) ve Aşama C (açıklanabilirlik) ayrı protokolde.

## 6. Yapılmayacaklar

- Test setinde eşik veya model seçimi.
- Alt grupları sonuçları gördükten sonra eklemek/çıkarmak.
- Yalnızca AUROC raporlamak — düşük prevalansta AUPRC, PPV ve kalibrasyon zorunlu.
- Fold ataması değiştirmek (`--force` gerekli ve gerekçesi yazılır).
- "ACS olmayan kolay negatifler" sayesinde şişmiş bir genel skoru klinik başarı
  gibi sunmak.
- Dış geçerlilik iddiası. Bu tek merkezli (Chongqing) bir veri setidir; harici
  doğrulama Gate 4'ün konusudur.

## 7. Bilinen kısıtlar

- Veri seti tek merkezli; OMI tanımı TIMI 0–1 ile sınırlı (spontan reperfüze
  olmuş vakalar negatif etiketli — makalenin kendi kısıtı).
- Test etiketleri gizli olduğu için test setinde kalibrasyon/AUPRC ölçülemez.
- Platformun döndürdüğü metrikler doğrulanamaz; tek doğruluk kaynağı onların
  sunucusu.
