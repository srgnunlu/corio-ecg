# Corio ECG — Master Roadmap v1

**Tarih:** 2026-06-16
**Yazan:** Sergen + Claude (analiz, kod değişikliği yok)
**Durum:** Aktif yol haritası — önceki `master-roadmap.md` (sadece başlık iskeleti) ve
`2026-06-13-profesyonel-urun-yol-haritasi.md` (regülasyon odaklı) belgelerinin
yerine geçen, **mevcut kanıtlara dayalı** birleşik plan.
**Branch:** `feature/phase3-failure-triage`

> Bu belge "ne yapmalıyız" değil, **"elimizdeki sayısal kanıta göre ne yapmalıyız"**
> sorusunu yanıtlar. Her fazın bir çıkış kriteri (gate) ve tahmini süresi vardır.
> Teknik terimler İngilizce, açıklamalar Türkçe.

---

## Vizyon

**PMcardio seviyesinde kağıt EKG analiz uygulaması.** Kullanıcı bir kağıt EKG'nin
telefon fotoğrafını çeker; sistem onu sinyale dönüştürür (digitize), 150+ sınıflı
tanı modeli çalıştırır, kalp hızı ve interval ölçümlerini yapar ve yapılandırılmış
bir rapor üretir. Hedef: **klinik karar destek** seviyesinde güvenilirlik —
güvenmediği görüntüyü reddeden ("yeniden çek"), emin olduğunda net konuşan bir ürün.

**Pipeline:** Foto → ECG-Digitiser (image→signal) → ECGFounder (signal→diagnosis)
→ Measurement (HR/interval) → LLM (structured report).

---

## Mevcut Durum Özeti

### Tamamlananlar (sayısal kanıtlarla)

| Başarı | Kanıt | Kaynak |
|---|---|---|
| Phase 1 pipeline uçtan uca çalışıyor | PTB-XL'de ~61 kayıt/sn, MPS (M4) | MEMORY Phase 1 |
| Phase 2 digitizasyon (merged to main `562b341`) | VPS n=500: %85 agreement, 0.91 cosine sim, sıfır failure | MEMORY Phase 2 |
| Model klinik sinyalde iyi | Temiz 10 s sinyal: **macro AUROC 0.905**, micro F1 0.666 | `diagnosis-roundtrip-accuracy-v1.md` |
| **Segment-ensemble production'a alındı** | Round-trip (n=300 sentetik): macro AUROC **0.754 → 0.871** | `segment-ensemble-production.md`, commit `80459b9` |
| Tanı çöküşünün kök nedeni izole edildi | ~0.12 AUROC kaybı = inference strategy (tiling), ~0.03 = digitizasyon fidelity | `diagnosis-tiling-impact-v1.md` |
| PMcardio holdout protokolü kuruldu | 30 tune + 60 test (locked), pre-registered, seed sabit | `pmcardio_holdout_v1.yaml` |
| Quality gate v1 frozen, feature contract tanımlı | 6 inference feature + 8 planned | `quality_gate_v1.yaml`, `quality_feature_contract_v1.yaml` |

**En önemli son kazanım:** Segment-ensemble. Kağıt EKG her lead'i zaman-kaydırmalı
kolonda basar; eski "tile + tüm sinyali birden besle" yolu cross-lead özellikleri
(aks, dal blokları) bozuyordu. Yeni yol her basılı kolonu ayrı diagnoz edip 150
sınıflık olasılık vektörlerini ortalıyor. SINUS TACHYCARDIA F1 0.09→0.78, AFib
0.11→0.65 kurtarıldı. **Üretim varsayılanı artık bu** (env `CORIO_SEGMENT_ENSEMBLE=0`
ile kapatılabilir, tiled fallback olarak duruyor).

### Nerede takılıyoruz — Kalite kapısı (quality gate)

Quality gate'in amacı: **güvenilmez digitizasyonu üretimden önce reddetmek.**
Frozen v1 gate tune setinde **henüz üretime hazır değil**:

- False accepts: **7/134** hedef-reddi (%5.22) — kötü görüntüleri kaçırıyor
- Reject recall: **%67.91**
- False rejects: **23/76** (%30.26) — kullanılabilir görüntüleri reddediyor
- Layout-scope varyantı (sadece 3x4+1R/3x4+3R kabul) false accept'i 2/134'e düşürüyor
  ama false reject'i **%56.58**'e çıkarıyor — promote edilemez.

**Asıl tıkanma:** İki dirençli false accept (`iphone/26`, `doogee/74`) **beş ayrı
reference-free probe** ile yakalanamadı (shadow stability, geometric perturbation,
Goldberger redundancy, image-space re-projection, calibration/assignment triage).
Triage'ın kesin bulgusu:

- **`iphone/26`** → kalibrasyon-ölçek hatası DEĞİL, lead-atama hatası DEĞİL. Gerçek
  defekt **per-cell horizontal/temporal registration**: precordial hücreler doğru
  lead'in doğru morfolojisini taşıyor ama ~5 s yanlış konumda. Üstelik bu kayma
  **düzensiz** (col 2'de V1,V2 5.1 s ama V3 0.14 s), rhythm-strip lead II'nin
  kendisi de kaymış → reference-free bir detektörün kilitleneceği yapı yok.
- **`doogee/74`** → ne ölçek, ne permütasyon, ne gross misregistration. Sadece
  V4–V6 morfolojisi bozuk (0.53–0.56 korelasyon). Marjinal near-miss.

**Karar (verildi):** Reference-free anchor arayışı **kapandı**. Bu hatalar
"interpretation/extraction-geometry" katmanında yaşıyor; digitizer'ın çıktıları
kendi içinde tutarlı ama yanlış — dış referans olmadan görünmezler. Bu **yayınlanabilir
bir negatif sonuç**. Doğru çözüm **data-first learned gate** (etiketli gerçek-foto↔referans
fidelity seti üzerinde eğitilmiş) — ama bu, 60-grup locked test setini yakmadan
toplanamayacak etiketli veri gerektirdiği için **ertelendi**.

### Bilinen kısıtlamalar

1. **Layout darlığı.** Sadece `3x4+1R` ve `3x4+3R` iyi çalışıyor (median corr 0.88).
   `6x2` → 0.297, `12x1` → 0/21 başarı. Holdout-tune'da genel extraction başarısı
   180/210 (%85.7) ama layout'a göre çok değişiyor.
2. **HR bozuk.** Kök neden: `_expand_canonical_segments` (`digitize.py`) gerçek 10 s
   Lead II rhythm strip'ini bile ~2.5 s kolona kırpıp tile'lıyor → gerçek RR dizisi
   atılıyor. HR fix riskli, validasyon verisi gerektiriyor, **henüz yapılmadı**.
3. **Interval ölçümü YOK.** PR/QT/QTc/QRS için hiç altyapı yok (greenfield).
4. **Rapor üretimi YOK.** LLM rapor modülü yok (`report_template.yaml` şablonu var ama
   kod yok).
5. **VT/SVT modülü boş** (`src/vtsvt/` sadece `__init__.py`).
6. **Sentetik ≠ gerçek.** 0.871 macro AUROC **round-trip (sentetik görüntü)** sayısı;
   gerçek telefon fotoğrafında uçtan uca tanı doğruluğu **henüz ölçülmedi**.
7. **Örneklem gücü.** 60-grup locked test bile, sıfır hata ile dahi <%2 false-accept
   üst sınırını kuramaz (rule-of-three ≈ 150 bağımsız sıfır-hata vaka gerektirir).
   Klinik iddialar için bağımsız toplanmış çok daha büyük veri şart.

---

## PMcardio Veri Seti — Kritik Sorular ve Cevaplar

### PMcardio'da gerçek telefon fotoğrafı var mı, yoksa sentetik mi?

**GERÇEK telefon fotoğrafları var.** Veri seti: PMcardio ECG Image Database
(PM-ECG-ID), Zenodo `10.5281/zenodo.13617673`, lisans **GPL-3.0-or-later**.
Kategoriler doğrudan gerçek çekim koşullarını yansıtıyor:

- `photos_iphone`, `photos_doogee`, `photos_samsung` → **farklı gerçek telefonlarla**
  çekilmiş kağıt EKG fotoğrafları
- `photos_scans` → flatbed tarayıcı (temiz referans)
- `photos_screens` → ekrandan çekim
- `photos_bents`, `photos_crumbles` → bükülmüş/buruşmuş kağıt (gerçek bozulma)
- Ayrıca JPEG-compression vb. augmentation varyantları (`augmentation_*` klasörleri)

**Önemli ayrım:** PMcardio fotoğrafları gerçektir ve her fotoğrafın **matched
ground-truth sinyali** vardır (`leads.npz`, `rhythms.npz`, ECG ID = `LPAE_xxxxx_hr`).
Yani bu seti **fidelity** (sinyal sadakati) doğrulaması için kullanabiliriz. **AMA
tanı etiketi yoktur** — referans, diagnoz değil sinyaldir. Bu yüzden tanı doğruluğu
ölçümü hâlâ sentetik PTB-XL round-trip ile yapılıyor (orada hem görüntü hem etiket var).

> İkinci bir kaynak: `data/real-phone/photos/` (10 vaka, `case001__front.jpg` …) —
> bunlar Sergen'in **kendi** telefon fotoğrafları, ayrı bir küçük set; çoğunun referans
> sinyali yok (10 foto, 3 sinyal, 0 referans). Hızlı duman testi için, metrik için değil.

### Tune (30 EKG) ve test (60 EKG locked) setlerinin durumu

- **Tune (30 ECG grubu, 210 matched görüntü):** AÇIK, araştırmaya serbest. Quality
  gate geliştirme, probe denemeleri, feasibility hep burada yapıldı.
- **Test (60 ECG grubu, locked):** **KAPALI ve KİLİTLİ.** Pre-registered, seed
  `20260615`. Bugüne kadar **hiç açılmadı**. Açılma kuralı (Faz B'de): yeni bir gate
  donduktan ve tune'da regresyonsuz doğrulandıktan **sonra, tek seferlik** çalıştırılır.
  Şu anda açmak için geçerli bir aday yok (reference-free yön kapandı).

### Mevcut kalite kapısının tam durumu

- **Frozen v1, üretime BAĞLI DEĞİL** (UI/diagnoz akışına wire edilmemiş — bilinçli).
- 6 aktif feature: `layout_cost`, `detected_leads_count`, `nonzero_leads_count`,
  `einthoven_score`, `avg_pixel_per_mm`, `raw_lines_count`.
- 8 "planned but unavailable" feature (blur, glare, shadow, occlusion,
  `calibration_pulse_confidence`, `reconstruction_disagreement`,
  `diagnosis_disagreement`, `page_completeness`) — hiçbiri henüz üretilmiyor.
- Tune metrikleri yukarıda (false accept %5.22, false reject %30.26). **Üretime hazır
  değil.**

### Segment-ensemble entegrasyonunun durumu

- **TAMAMLANDI ve production default** (commit `80459b9`, `4a76528`; branch'te,
  **push edilmedi**). Web app (`src/web/app.py`) varsayılan olarak per-column
  segment-ensemble çalıştırıyor; tiled fallback duruyor.
- `evaluate_roundtrip.py` artık aynı paylaşılan helper'ları (`build_paper_column_signals`,
  `aggregate_probability_vectors`) kullanıyor → eval ve üretim **birebir aynı matematik**.
- Testler: 273 passed, 1 skipped; ruff + mypy temiz.
- **Eksik:** Gerçek PMcardio fotoğraflarında uçtan uca doğrulama (round-trip 0.871
  sentetik üst-sınırdır, gerçek-foto sayısı değil).

### Klinik özellikler (HR, interval) için mevcut altyapı var mı?

- **HR: KISMİ altyapı var.** `src/utils/rhythm.py` → `estimate_heart_rate_bpm`,
  Pan-Tompkins autocorrelation, Lead II tercihli, fonksiyon-içi 5–30 Hz bandpass.
  **Ama** upstream'de `_expand_canonical_segments` rhythm strip'i kırptığı için
  gerçek 10 s RR dizisi modele/HR'a ulaşmıyor → HR güvenilmez.
- **Interval (PR/QT/QTc/QRS): altyapı YOK** (greenfield).
- **Ritim şeridi:** digitize ediliyor (`rhythms.npz` referansta var) ama üretim
  pipeline'ı 2.5 s'ye kırpıp tile'lıyor — tam 10 s şerit henüz tanı/HR'a beslenmiyor.

---

## Faz A: Kalite Kapısı Sertleştirme (ŞİMDİ)

**Amaç:** Güvenilmez digitizasyonu üretimden önce reddeden, dürüst metriklerle
donmuş bir gate. Bu faz **araştırma tıkanıklığını dürüstçe kapatmak** ile başlıyor.

### A.0 — Phase 0 triage sonucu (TAMAMLANDI)

Reference-free yön **bitti**. Beş probe de başarısız; `iphone/26` temporal
registration, `doogee/74` morfoloji near-miss (yukarıda detaylı). Bu, planın §9'da
öngörülen "documented rejection" çıktısı. **Bunu yeniden denemeyin** —
`phase3-failure-triage-v1.md`'de gerekçeler kayıtlı.

### A.1 — Kalibrasyon ankoru stratejisi (yeniden konumlandırma)

Orijinal plan (`phase3-calibration-anchor-plan.md`) iki fiziksel anchor öneriyordu;
triage **ikisini de mevcut survivor'lar için çürüttü**. Dolayısıyla:

- **Track A (calibration-pulse amplitude):** SUPERSEDED. `iphone/26`'nın kalibrasyonu
  aynı EKG'nin temiz taramasıyla özdeş (gain 0.948 vs 0.946); 1 mV pulse-deviation
  feature'ının ayıracak ayrımı yok. **Yapma.**
- **Track B (lead-label assignment):** SUPERSEDED (bu survivor'lar için). `iphone/26`
  etiketleri/layout'u doğru. Track B *farklı* bir false-accept sınıfını locked test'te
  yakalayabilir ama elimizdeki survivor'ları yakalamıyor → tek başına gerekçelendirilemez.

**Yeni strateji = iki yönlü:**

1. **Ship & document (düşük risk, ŞİMDİ):** Frozen gate'i mevcut haliyle, üretime
   **bağlamadan**, `iphone/26` + `doogee/74`'ü "bilinen residual false accept" olarak
   belgele. Negatif sonucu yayına hazırla (bkz. Faz E / publication).
2. **Data-first learned gate (asıl çözüm, ertelendi → Faz B'ye bağlı):** Reference-free
   detektör yerine, etiketli gerçek-foto↔referans fidelity seti üzerinde **öğrenilmiş**
   bir gate. Bu, locked test'i yakmayan **yeni veri** gerektirir (Faz B veri toplama).

### A.2 — Track A / Track B detayları (referans için, ASKIDA)

Orijinal planın modül tasarımı (`src/quality/calibration_pulse.py`,
`src/quality/lead_assignment_check.py`, `*_one.py`/`*_pilot.py` izole worker pattern)
teknik olarak sağlam ve **gelecekte farklı bir false-accept sınıfı için** yeniden
kullanılabilir. Ancak mevcut survivor'lara karşı ölü; bu yüzden **yalnızca data-first
gate'in feature mühendisliğinde aday** olarak saklanıyor, öncelikli iş değil.

### A.3 — Başarı kriteri

- [ ] Frozen v1 gate, üretime bağlanmadan, residual'lar belgelenmiş halde dondurulmuş.
- [ ] `iphone/26` + `doogee/74` "known limitation" olarak `docs/` altında kayıtlı (zaten).
- [ ] Reference-free yönün kapandığı ve data-first'e geçişin gerekçesi tek belgede
      toplanmış (bu roadmap + triage).
- [ ] Locked 60-grup test **kapalı** kaldı.
- **Karar noktası:** Yeni gate adayı YOK ise, Faz A burada "dürüstçe kapalı" sayılır
  ve enerji Faz C (klinik özellikler) ile Faz B (veri toplama) arasında bölünür.

### A.4 — Tahmini süre

**0.5–1 hafta** (çoğu analiz zaten yapıldı; kalan iş dokümantasyon + dondurma +
karar). Data-first gate'in kendisi Faz B'ye bağımlı, ayrı ölçülüyor.

---

## Faz B: PMcardio Test Seti Doğrulaması

**Amaç:** İlk dürüst, dış-referanslı performans okuması. Bu faz **veri** ve **kilit
açma disiplini** ile ilgili.

### B.1 — 60 EKG locked test setinin açılma koşulları

Test seti **yalnızca** şu zincir tamamlanınca, **tek seferlik** açılır:

1. Donmuş bir gate (veya tanı) aday politikası mevcut.
2. Tune'da tam değerlendirilmiş, regresyon yok (false-reject artmadı).
3. Açılış öncesi beklenen sonuç ve kabul kriteri **yazılı olarak** pre-register
   edilmiş (analiz sonrası kriter değiştirme yasak).
4. Tek koşu; sonuç ne olursa olsun set "tüketilmiş" sayılır, ikinci kez aynı amaçla
   kullanılmaz.

> **⚠️ KRİTİK:** Şu an geçerli bir aday yok. Test'i merakla açmak, en değerli iç
> kanıtı yakar. Açma.

### B.2 — Gerçek fotoğraf durumu

PMcardio'nun gerçek fotoğrafları (iphone/doogee/samsung/scans/screens/bents/crumbles)
fidelity ground-truth'a (matched sinyal) sahip ama **tanı etiketine sahip değil**.
Bu yüzden test setinde ölçebileceğimiz **birincil metrik fidelity'dir** (korelasyon,
RMSE, SNR), tanı doğruluğu değil. Tanı doğruluğu için ayrı, etiketli veri gerekiyor
(B.4).

### B.3 — Hedef metrikler

| Metrik | Mevcut (tune) | Faz B hedefi (test, in-scope layout) |
|---|---|---|
| Extraction success (3x4+1R/3x4+3R) | %85.7 genel; in-scope median corr 0.88 | ≥ %90 in-scope |
| Median waveform correlation | 0.526 genel, 0.88 in-scope | ≥ 0.85 in-scope |
| Gate false accept (in-scope) | %1.49 (layout-scope) | dürüst rapor (CI ile) |
| Gate false reject | %30–56 | < %25 hedef, ama tek-koşu okuması |

> Bu hedefler **iç mühendislik** kanıtıdır. Klinik/dış iddia için yeterli **değildir**
> (örneklem gücü kısıtı — bkz. Risk).

### B.4 — Ek veri toplama gereksinimi (asıl darboğaz)

İki ayrı veri ihtiyacı var, karıştırılmamalı:

1. **Fidelity için (gate'in data-first çözümü):** Daha büyük, etiketli gerçek-foto↔referans
   seti. PMcardio'nun kalan kategorileri + Sergen'in kendi `data/real-phone` setinin
   referanslı genişletilmesi. Hedef: locked test'i yakmadan, gate'i **öğrenecek** kadar
   örnek (yüzlerce bağımsız EKG).
2. **Tanı doğruluğu için (gerçek-foto, gerçek-etiket):** Bugün **hiç yok**. Sentetik
   round-trip üst sınır veriyor (0.871) ama gerçek-foto tanı doğruluğu ölçülemiyor.
   Çözüm: bilinen tanılı gerçek kağıt EKG'lerin fotoğrafları (klinik arşiv, etik kurul
   ile de-identified). Bu, prospektif/retrospektif veri toplama → etik kurul başvurusu
   gerektirir.

### B.5 — Tahmini süre

- B.1–B.3 (test okuması hazırlığı + tek koşu): **1–2 hafta** *aday hazır olduğunda*.
- B.4 (veri toplama): **6–12 hafta+** (etik kurul süresine bağlı; Sergen'in EM arşiv
  erişimi avantaj). Bu, paralel başlatılması gereken **en uzun lead-time** iş.

---

## Faz C: Klinik Özellikler

**Amaç:** Tanı olasılıklarının yanında klinisyenin beklediği **ölçümler ve net
kararlar**. Bu faz büyük ölçüde **software**, GPU gerektirmez — Mac M4 yeterli.

### C.1 — HR (kalp hızı) hesaplama

- **Kök neden (biliniyor):** `_expand_canonical_segments` 10 s rhythm strip'i 2.5 s'ye
  kırpıp tile'lıyor.
- **Doğru fix:** Tam-genişlik rhythm lead'ini (Lead II, 10 s) kırpmadan koru ve
  `estimate_heart_rate_bpm`'e ham 10 s olarak besle. `rhythm.py` zaten Pan-Tompkins
  autocorrelation ile hazır.
- **⚠️ Risk:** Bu değişiklik core digitization davranışını hem HR hem diagnosis sinyali
  için değiştirir → **validasyon verisiyle** yapılmalı (segment-ensemble kararı zaten
  bunu blind değiştirmekten kaçındı). PMcardio rhythm referansı (`rhythms.npz`) burada
  ground-truth olarak kullanılabilir: digitize edilen 10 s şerit vs referans şerit HR'ı.
- **Başarı kriteri:** Tune setinde digitize-HR vs referans-HR mutlak hata medyanı
  < 5 bpm; >20 bpm sapma oranı < %10 (mevcut tile'lı yolda %32).

### C.2 — Interval ölçümleri (PR, QT, QRS, QTc)

- **Durum:** Greenfield, hiç altyapı yok.
- **Yaklaşım:** Delineation kütüphanesi (örn. NeuroKit2 — Python, açık kaynak; Context7
  ile API doğrula) ile P/QRS/T sınırlarını çıkar; PR, QRS süresi, QT ölç; QTc için
  Bazett + Fridericia. Referans: 10 s rhythm strip + en temiz precordial lead.
- **Modül:** `src/measurement/intervals.py` (yeni), pure + testable, sentetik known-interval
  sinyalleriyle unit test.
- **⚠️ Yeni bağımlılık:** NeuroKit2 → eklenmeden önce maintained mı, lisans (MIT),
  alternatifler (ecg-segmentation modelleri) değerlendir; Sergen'e bildir.
- **Başarı kriteri:** PTB-XL'in makine ölçümleriyle (varsa) veya uzman okumasıyla
  concordance; QTc için ±20 ms tolerans hedefi.

### C.3 — Ritim şeridi analizi

- Tam 10 s Lead II şeridini koruyup (C.1 fix'iyle birlikte) ritim sınıflandırmaya
  (AFib, sinüs, ektopi) ve RR-değişkenliğine besle.
- Segment-ensemble tanı doğruluğunu burada gerçek 10 s şeritle test et (NORMAL ECG
  0.31, PVC 0.45 residual'larının asıl iyileşme kapısı).

### C.4 — Normal EKG tespiti

- **Sorun:** Segment-ensemble'da NORMAL ECG F1 sadece 0.31 (tiled'da 0.00, baseline
  0.76). Kök neden: NORMAL kararı 12 lead'in **birlikte** değerlendirilmesini ister;
  kolon-bazlı bölme bunu zorlaştırır.
- **Yaklaşım:** Whole-12-lead bir "NORMAL doğrulama" pass'i ekle (segment-ensemble
  pozitif olduğunda full-aligned sinyalle ikinci kontrol) veya rhythm-strip-aware
  hibrit. Bu, C.1/C.3 ile birlikte gelir.

### C.5 — PVC ve aritmiler

- **Sorun:** PVC F1 0.45 (baseline 0.93). Ektopi tek atımlık olay; 2.5 s kolonda
  kaybolur, tam 10 s şerit ister.
- **Yaklaşım:** Tam 10 s rhythm strip'i ayrı bir "ectopy/rhythm" başına besle (C.1/C.3'e
  bağlı); RR düzensizliği + tek-atım morfoloji sapması ile PVC/PAC işaretle.

### C.6 — Tahmini süre

**6–10 hafta** (C.1 önce, 1–2 hafta + validasyon; C.2 intervaller 2–3 hafta; C.3–C.5
ritim/normal/ektopi 3–4 hafta, C.1'e bağımlı). GPU gerekmez.

---

## Faz D: Ürün Olgunlaştırma

**Amaç:** Araştırma prototipinden klinisyenin günlük kullanabileceği ürüne.

### D.1 — Web arayüzü iyileştirme

- Mevcut: Gradio test UI (`src/web/app.py`, localhost:7860, 6x2+1R görselleştirme).
- İşler: digitize overlay (sinyal vs orijinal foto), per-lead güven göstergesi,
  quality-gate "yeniden çek" davranışı (gate üretime bağlandığında), HR/interval/tanı
  panelleri, einthoven debug paneli korunur.
- **Abstention UX'i kritik:** Güvenmediğinde net "bu görüntüyü okuyamadım, şu açıdan
  yeniden çekin" mesajı (program ilkesi: kritik sınıfta "bilmiyorum" geçerli davranış).

### D.2 — PDF rapor çıktısı (LLM, Phase 5)

- **Durum:** Şablon var (`configs/report_template.yaml`), kod YOK.
- **Yaklaşım:** Deterministic ölçüm bloğu (HR/interval/aks/tanı listesi) + LLM ile
  doğal-dil özet. **LLM klinik karar motoru değildir** (program ilkesi) — sadece
  yapılandırılmış bulguları okunur metne çevirir, yeni tanı üretmez.
- **Model:** En güncel Claude (Opus 4.8 / `claude-opus-4-8`); API key `.env`'de,
  hiçbir hardcode yok.
- **Çıktı:** `docx`/`pdf` skill'leri ile profesyonel rapor (başlık, ölçümler, tanı
  olasılıkları, disclaimer).
- **⚠️ GÜVENLİK:** Hasta verisi loglanmaz; rapor üretiminde PHI/PII URL'lere/loglara
  sızmaz.

### D.3 — Mobil uygulama (opsiyonel)

- React Native (Expo) — Sergen'in stack'i. Telefonla çek → backend digitize/diagnoz
  → rapor. **V1 için opsiyonel/ertelenebilir**; önce web olgunlaşsın.
- Backend ağır model çalıştırdığı için server-side inference (Mac M4 self-host veya
  vast.ai burst) gerekir; mobil sadece capture + UI.

### D.4 — Dağıtım stratejisi

- **V1:** Self-hosted (Mac M4 inference, web UI lokal/Hostinger VPS frontend).
- Model ağırlıkları/dataset asla repo'ya commit edilmez (gitignored; `scripts/` ile
  indirilir).
- Heavy inference burst gerekirse vast.ai RTX 4090 (~$0.28/saat) — sadece açık/de-identified
  veri ile.

### D.5 — Tahmini süre

**8–14 hafta** (D.1 web 3–4 hafta; D.2 rapor 3–4 hafta; D.3 mobil opsiyonel +4–8 hafta).

---

## Risk ve Kısıtlamalar

### Hesaplama gücü (vast.ai ne zaman gerekir?)

- **Mac M4 (24 GB) yeterli:** Tüm inference (digitize ~7 s, ECGFounder ~1 GB),
  segment-ensemble, HR/interval, hafif fine-tune, Faz A/C/D'nin neredeyse tamamı.
- **vast.ai gerekir:** (a) Büyük fidelity/tanı setinde batch digitizasyon eval (n=500+),
  (b) data-first learned gate eğitimi, (c) ECGFounder/digitizer fine-tune. Her seferinde
  **yalnızca açık veya güvenli de-identified veri** (program ilkesi). Önceki VPS kapatıldı;
  gerektiğinde yeniden kurulur.

### Veri toplama gereksinimleri (en kritik darboğaz)

- **Örneklem gücü gerçeği:** <%2 false-accept üst sınırı için ~150 bağımsız sıfır-hata
  vaka gerekir. 60-grup locked test bile tek başına klinik iddia için yetmez.
- **İki ayrı eksik:** (1) etiketli gerçek-foto↔referans **fidelity** seti (gate için),
  (2) gerçek-foto↔gerçek-**tanı** seti (tanı doğruluğu için, bugün sıfır).
- **Lead-time uzun:** Etik kurul + de-identification haftalar-aylar. **Şimdi başlatılmalı**
  (Faz B.4), diğer fazlara paralel.

### Yasal / regülatif konular

- **KVKK / hasta verisi:** Repo'da PHI/PII yok (sadece açık dataset). Gerçek klinik veri
  girince RLS, de-identification, etik kurul, veri paylaşım sözleşmesi zorunlu.
- **Lisanslar:** PMcardio **GPL-3.0** (türev çalışma yükümlülüğüne dikkat — ürünleştirmede
  kritik), ECGFounder (NEJM AI 2024 kullanım koşulları), ECG-Digitiser BSD-2,
  Open-ECG-Digitizer. Ürünleşmeden önce **lisans envanteri** (Faz 0, profesyonel
  roadmap'te tanımlı) tamamlanmalı. **⚠️ KRİTİK:** GPL-3.0 dataset türevleri ticari
  kapalı-kaynak ürünü kısıtlayabilir — hukuki görüş gerekli.
- **Medikal cihaz:** Klinik karar destek = potansiyel tıbbi cihaz (CE/Türkiye TİTCK).
  V1 dar kapsamlı, "araştırma/pilot" sınırında tutulur; pazara sunum ayrı program
  (profesyonel roadmap Faz 7).
- **Sorumluluk:** Her raporda "karar destek, nihai sorumluluk hekimde" disclaimer'ı.

---

## Kilometre Taşları ve Zamanlama

| Faz | Ana çıktı | Çıkış kriteri (gate) | Bağımlılık | Süre |
|---|---|---|---|---|
| **A. Kalite kapısı sertleştirme** | Frozen gate + belgelenmiş residual, reference-free yön dürüstçe kapalı | Gate donmuş & üretime bağlı değil; residual'lar kayıtlı; locked test kapalı | — | **0.5–1 hafta** |
| **B. PMcardio test doğrulaması** | İlk dış-referanslı fidelity okuması (tek koşu) + veri toplama başlatma | Aday hazır + pre-registered kriter → tek koşu; B.4 veri pipeline'ı açık | A, etik kurul | **1–2 hafta** (+ B.4: 6–12 hafta paralel) |
| **C. Klinik özellikler** | HR + PR/QT/QTc/QRS + ritim + NORMAL/PVC | HR hata medyanı <5 bpm; interval concordance; gerçek 10 s şerit beslemesi doğrulandı | C.1 fix validasyonu (PMcardio rhythm ref) | **6–10 hafta** |
| **D. Ürün olgunlaştırma** | Web UX + LLM/PDF rapor (+ opsiyonel mobil) | Abstention UX çalışıyor; deterministic+LLM rapor üretiliyor; PHI sızıntısı yok | C (ölçümler), A (gate UX) | **8–14 hafta** |

**Paralellik:** B.4 (veri toplama) en uzun lead-time → **hemen başlat**, A/C ile
paralel ilerle. C büyük ölçüde software → GPU beklemeden başlayabilir. D, C'nin
ölçüm çıktılarına bağlı.

**Kritik yol:** A (kapat) → C.1 HR fix (PMcardio rhythm ref ile valide et) → C.3/C.5
ritim → D.2 rapor. Veri toplama (B.4) bunlara paralel arka planda.

### Bir sonraki oturum için somut ilk adımlar

1. **Faz A'yı dürüstçe kapat:** Branch'i (`feature/phase3-failure-triage`) commit'le,
   residual'ları bu roadmap + triage ile belgele (zaten yazıldı). Push **yapma** (Sergen
   isteyince).
2. **Segment-ensemble'ı gerçek fotoğrafta doğrula:** PMcardio tune fotoğraflarında
   uçtan uca digitize→diagnoz, round-trip 0.871'in gerçek-foto karşılığını ölç.
3. **C.1 HR fix'ini PMcardio `rhythms.npz` ile valide et** (en yüksek değerli, software-only,
   GPU gerektirmez).
4. **B.4 veri toplamayı başlat:** Etik kurul / de-identified arşiv erişimi için süreci aç
   (en uzun lead-time).

---

## Ek: Bu Roadmap'in Dayandığı Kanıt Dosyaları

- `docs/experiments/diagnosis-roundtrip-accuracy-v1.md` — tanı doğruluğu (0.754→0.871→0.905)
- `docs/experiments/phase3-failure-triage-v1.md` — reference-free yönün kapanışı
- `docs/experiments/phase3-pmcardio-holdout-tune-v1.md` — gate tune metrikleri, layout darlığı
- `docs/plans/phase3-calibration-anchor-plan.md` — Track A/B tasarımı (askıda)
- `docs/sessions/2026-06-16-segment-ensemble-production.md` — segment-ensemble entegrasyonu
- `configs/quality_gate_v1.yaml`, `quality_feature_contract_v1.yaml`, `pmcardio_holdout_v1.yaml`
- `data/reference/pmcardio*/source.json` — PMcardio provenance (Zenodo, GPL-3.0)
