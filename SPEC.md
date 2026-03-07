# SPEC: Corio ECG - Paper ECG Image Interpretation AI Pipeline

## Proje Ozeti

Kagit ECG fotograflarindan yapilan taramalardan 150+ tani koyabilen, yapilandirilmis rapor ureten bir AI pipeline'i. Mevcut acik kaynak modelleri (ECGFounder + ECG-Digitiser) temel alarak fine-tuning ile gelistirilecek. Ozellikle VT/SVT ayirimi icin Brugada ve Vereckei kriterleri gibi klinik morfoloji kurallari modele ogretilecek.

**Hedef:** Akademik yayinlar + ileride mobil uygulama
**Yaklasim:** Sifirdan model egitmek degil, mevcut guclu modelleri fine-tune ederek gelistirmek

---

## 1. Sistem Mimarisi

### 1.1 Genel Pipeline

```
[Kagit ECG Fotografi / Tarama]
        |  (PNG/JPG)
        v
[ECG-Digitiser]  ──  nnU-Net 2D segmentasyon + Hough Transform
        |  (WFDB formati: 12-lead, 500 Hz, mV)
        v
[ECGFounder]  ──  Net1D CNN, 76.3M parametre
        |  (150 sinif icin olasilik skorlari)
        v
[Rapor Motoru]  ──  LLM entegrasyonu (acik kaynak veya Claude/GPT-4)
        |
        v
[Yapilandirilmis ECG Raporu]
```

### 1.2 Bilesen Detaylari

#### ECG-Digitiser (Goruntu → Sinyal)
- **Kaynak:** https://github.com/felixkrones/ECG-Digitiser
- **Model:** nnU-Net 2D segmentasyon (~475 MB)
- **Girdi:** PNG/JPG kagit ECG goruntusu (standart 3x4 + ritim stribi formati)
- **Cikti:** WFDB formati (.dat + .hea), 12-lead, 500 Hz, mV
- **GPU:** ~4 GB VRAM (CPU'da da calisir, yavas)
- **Lisans:** BSD-2

#### ECGFounder (Sinyal → Tani)
- **Kaynak:** https://github.com/PKUDigitalHealth/ECGFounder
- **HuggingFace:** https://huggingface.co/PKUDigitalHealth/ECGFounder
- **Model:** Net1D (RegNet tabanli 1D CNN), 76.3M parametre (~370 MB)
- **Girdi:** 12-lead ECG sinyali (numpy array, 12 x 5000, z-score normalized)
- **Cikti:** 150 sinifli multi-label classification (sigmoid olasiliklari)
- **GPU:** ~1 GB VRAM
- **Yayin:** NEJM AI (2024)

#### Rapor Motoru (Tani → Yapilandirilmis Rapor)
- **Girdi:** ECGFounder'dan gelen 150 etiket olasiliklari
- **LLM secimi:** Proje ilerledikce karar verilecek (acik kaynak oncelikli, alternatif Claude/GPT-4)
- **Cikti formati:**

```
Ritim: [Sinus ritmi / AF / Atriyal flutter / SVT / VT / ...]
Hiz: [XX bpm]
Aks: [Normal / Sol aks deviasyonu / Sag aks deviasyonu]
PR Intervali: [Normal / Uzamis / Kisa]
QRS Suresi: [Normal / Genis]
QT/QTc: [Normal / Uzamis]
ST Segmenti: [Normal / Elevasyon (derivasyonlar) / Depresyon (derivasyonlar)]
T Dalgasi: [Normal / Inversiyon (derivasyonlar)]
Iletim Bozuklugu: [Yok / RBBB / LBBB / AV Blok tipi]
Hipertrofi: [Yok / LVH / RVH]
Ozel Bulgular: [WPW / Brugada paterni / Perikardit / ...]
Genel Degerlendirme: [Ozet yorum]
```

- **Rapor dili:** Turkce ve Ingilizce secenegi

---

## 2. Tani Kapsami

### 2.1 Temel Ritim Tanilari
- Sinus ritmi (normal, bradikarsi, tasikardi)
- Atriyal fibrilasyon (AF)
- Atriyal flutter
- Supraventrikuler tasikardi (SVT)
- Ventrikuler tasikardi (VT)
- Ventrikuler fibrilasyon (VF)

### 2.2 Iletim Bozukluklari
- Sag dal blogu (RBBB)
- Sol dal blogu (LBBB)
- AV Blok (1., 2. derece Tip I/II, 3. derece)
- Fascikuler bloklar

### 2.3 Iskemik Degisiklikler
- ST elevasyonu (STEMI, derivasyon bazli lokalizasyon)
- ST depresyonu
- T dalga inversiyonu
- Patolojik Q dalgalari

### 2.4 Hipertrofi
- Sol ventrikul hipertrofisi (LVH)
- Sag ventrikul hipertrofisi (RVH)
- Atriyal genisleme

### 2.5 Diger Bulgular
- QT uzamasi
- WPW (Wolff-Parkinson-White)
- Brugada paterni
- Perikardit
- Elektrolit bozukluklari (hiperkalemi, hipokalemi)
- Pacemaker ritimleri

### 2.6 Ozel Odak: VT vs SVT Ayirimi
- Brugada kriterleri (RS intervali, morfoloji uyumu)
- Vereckei algoritmasi (aVR derivasyonu analizi)
- Diger morfolojik kriterler (AV disosiyasyon, capture/fusion beats)
- Bu alan projenin en onemli akademik katkisi olacak

---

## 3. Veri Stratejisi

### 3.1 Temel Veri Setleri

| Veri Seti | Kayit Sayisi | Kullanim Amaci |
|-----------|-------------|----------------|
| **PTB-XL** | 21,799 (12-lead) | Ana benchmark, fine-tuning baslangici, 71 etiket |
| **MIMIC-IV-ECG** | 800,000+ | Ek egitim verisi, ICU hasta cesitliligi |

### 3.2 Sentetik Goruntu Uretimi
- **ECG-Image-Kit** (https://github.com/alphanumericslab/ecg-image-kit) kullanilarak PTB-XL sinyallerinden gercekci kagit ECG goruntuleri uretilecek
- Augmentasyon: grid varyasyonlari, kagit kirisimliklari, aci bozukluklari, isik degisimleri, farkli ECG cihaz formatlari
- **PTB-XL-Image-17K** veri seti de kullanilabilir (hazir render edilmis)

### 3.3 VT/SVT Ozel Veri
- PTB-XL ve MIMIC-IV-ECG'den VT ve SVT ornekleri filtrelenecek
- Brugada/Vereckei kriterlerine gore ek etiketleme yapilacak
- Gerekirse diger acik kaynak veri setlerinden (Chapman-Shaoxing, CPSC 2018) ek VT/SVT ornekleri alinacak

---

## 4. Fine-tuning Stratejisi

### 4.1 ECGFounder Fine-tuning Hedefleri

#### Hedef A: Dijitizasyon Artefaktlarina Dayaniklilik
- **Problem:** ECGFounder temiz dijital sinyallerle egitilmis. Kagittan dijitize edilen sinyallerde kalite kaybi var.
- **Yontem:** Temiz sinyallere yapay digitizasyon gurultusu ekleyerek (noise injection) fine-tune
- **Basari metrigi:** Temiz vs dijitize sinyal arasindaki performans farkini minimize etmek

#### Hedef B: Genel Dogruluk Artisi
- **Yontem:** MIMIC-IV-ECG verisiyle ek egitim
- **Basari metrigi:** PTB-XL benchmark'ta AUROC iyilesmesi

#### Hedef C: VT/SVT Uzmanlastirma
- **Yontem:** Morfoloji bazli kriterleri modele ogretmek (en uygun teknik arastirilacak)
  - Aday yontemler: Rule-augmented training, multi-task learning, curriculum learning, veya kombinasyonu
  - Brugada kriterleri: V1-V2 morfolojisi, RS intervali >100ms, AV disosiyasyon
  - Vereckei algoritmasi: aVR derivasyonu analizi
- **Basari metrigi:** VT vs SVT siniflandirma AUROC, sensitivity, specificity

#### Hedef D: Yeni Tani Ekleme
- ECGFounder'in kapsamadigi ek durumlar icin yeni siniflar ekleme
- Transfer learning ile mevcut agirliklar korunarak yeni baslik (head) egitimi

### 4.2 ECG-Digitiser Fine-tuning
- Farkli ECG cihaz formatlari icin segmentasyon iyilestirme
- Dusuk kaliteli goruntu (bulanik, egik, dusuk cozunurluk) toleransini artirma
- Farkli kagit renkleri/grid stilleri icin adaptasyon

---

## 5. Proje Fazlari

### Faz 1: Temel Pipeline Kurulumu
- **Hedef:** ECG-Digitiser + ECGFounder calisir hale getirmek
- **Isler:**
  - Gelistirme ortamini hazirla (Python, PyTorch, bagimliliklar)
  - ECG-Digitiser'i kur ve test et (ornek goruntularle)
  - ECGFounder'i kur ve test et (PTB-XL sinyalleriyle)
  - Iki modeli birlestir: goruntu → sinyal → tani pipeline'i
  - PTB-XL uzerinde baseline performans olc
- **Cikti:** Calisan pipeline + baseline metrikler
- **Potansiyel makale:** -

### Faz 2: Dijitizasyon Dayanikliligi
- **Hedef:** Kagittan dijitize edilen sinyallerde performans kaybini olcmek ve azaltmak
- **Isler:**
  - ECG-Image-Kit ile PTB-XL'den sentetik kagit ECG goruntuleri uret
  - Bu goruntuleri ECG-Digitiser ile dijitize et
  - Dijitize sinyallerdeki ECGFounder performansini olc (temiz sinyale kiyasla)
  - Performans farkini belgele
  - Dijitizasyon artefaktlarina karsi fine-tuning yap
  - Fine-tuning sonrasi performansi olc
- **Cikti:** Dijitizasyona dayanikli model + karsilastirma metrikleri
- **Potansiyel makale:** "Impact of ECG Digitization Artifacts on AI Diagnostic Accuracy: Evaluation and Mitigation Using Foundation Models"
  - Arastirma sorusu: Kagittan dijitize edilen ECG'lerde AI performansi ne kadar dusuyor ve fine-tuning ile bu ne kadar telafi edilebilir?
  - Hedef dergi: Computers in Biology and Medicine (IF ~7), JMIR (IF ~7)

### Faz 3: Ek Veri ile Genel Iyilestirme
- **Hedef:** MIMIC-IV-ECG ekleyerek genel dogrulubu artirmak
- **Isler:**
  - MIMIC-IV-ECG verisini indir ve isle
  - ECGFounder'i MIMIC-IV-ECG ile fine-tune et
  - PTB-XL benchmark'ta performans karsilastirmasi yap
  - Ablasyon calismalari (hangi veri ne kadar katkida bulunuyor)
- **Cikti:** Gelistirilmis model + ablasyon sonuclari
- **Potansiyel makale:** "Improving ECG Foundation Model Generalization with Multi-Source ICU Data"
  - Arastirma sorusu: ICU hasta verisi (MIMIC-IV) eklemek, genel populasyon icin egitilmis modelin performansini nasil etkiler?
  - Hedef dergi: npj Digital Medicine (IF ~15), Heart (IF ~6)

### Faz 4: VT/SVT Uzmanlastirma
- **Hedef:** Brugada ve Vereckei kriterleriyle VT/SVT ayirimi icin model egitmek
- **Isler:**
  - VT ve SVT orneklerini tum veri setlerinden topla ve filtrele
  - Morfolojik kriterleri modele ogretmek icin en uygun yontemi arastir
  - Brugada kriterleri (V1-V2 R dalgasi genisligi, RS intervali, morfoloji uyumu)
  - Vereckei algoritmasi (aVR analizi)
  - Fine-tuning yap
  - Klasik kural bazli algoritmalar vs AI vs AI+kurallar karsilastirmasi
- **Cikti:** VT/SVT uzman modeli + karsilastirma calisması
- **Potansiyel makale:** "Teaching AI Classical ECG Morphology Criteria: A Novel Approach to VT vs SVT Differentiation Using Brugada and Vereckei Algorithms"
  - Arastirma sorusu: Klinik morfoloji kurallarini AI'ya ogretmek, saf veri odakli egitimden daha iyi sonuc verir mi?
  - Hedef dergi: AJEM (IF ~4), Circulation (IF ~35), European Heart Journal (IF ~35)
  - **NOT:** Bu projenin en guclu akademik katkisi. Kimse henuz klasik tani kriterlerini AI modeline sistematik olarak ogretme yaklasimini yayinlamadi.

### Faz 5: Rapor Uretimi
- **Hedef:** LLM entegrasyonu ile yapilandirilmis rapor ciktisi
- **Isler:**
  - LLM secimi (acik kaynak oncelikli, alternatif Claude/GPT-4)
  - Prompt engineering: 150 etiket olasiligini yapilandirilmis rapora donusturme
  - Rapor sablonu olusturma (Turkce + Ingilizce)
  - Rapor kalitesini degerlendirme (varsa kardiyolog validasyonu)
- **Cikti:** Otomatik rapor uretim sistemi
- **Potansiyel makale:** "Automated Structured ECG Reporting Using Foundation Models and Large Language Models"
  - Arastirma sorusu: AI uretimli yapilandirilmis ECG raporlari klinik kullanima uygun mu?
  - Hedef dergi: JAMIA (IF ~7), European Heart Journal - Digital Health

### Faz 6: Test Arayuzu
- **Hedef:** Basit web arayuzu ile pipeline'i test etmek
- **Isler:**
  - ECG fotografi yukleme
  - Pipeline'i calistirma (digitize → tani → rapor)
  - Sonucu gosterme
  - Minimal, fonksiyonel tasarim (test amacli)
- **Teknoloji:** Next.js veya basit bir Python (Gradio/Streamlit) arayuzu
- **Cikti:** Calisan web demo

### Faz 7: Akademik Yayin
- Her fazin sonuclarini ayri makale olarak yazma
- Tam pipeline'i kapsayan buyuk bir makale
- **Potansiyel buyuk makale:** "End-to-End Paper ECG Interpretation Pipeline: From Photograph to Structured Report Using AI Foundation Models"
  - Hedef dergi: Lancet Digital Health (IF ~30), Nature Medicine (IF ~80)

---

## 6. Teknik Gereksinimler

### 6.1 Donanim

| Bilesen | Minimum | Onerilen |
|---------|---------|----------|
| **Gelistirme/Test** | Mac Mini M4, 24GB RAM | Ayni (yeterli) |
| **Fine-tuning** | Vast.ai RTX 4090 ($0.28/saat) | Vast.ai A100 40GB ($0.29-0.87/saat) |
| **Inference** | Mac Mini M4 veya Google Colab (ucretsiz T4) | Ayni |

### 6.2 Tahmini GPU Maliyeti

| Faz | Tahmini GPU Saati | Tahmini Maliyet |
|-----|-------------------|-----------------|
| Faz 1 (kurulum/test) | ~5 saat | ~$1.5 (veya ucretsiz Colab) |
| Faz 2 (digitizasyon fine-tune) | ~20-40 saat | ~$6-12 |
| Faz 3 (MIMIC-IV fine-tune) | ~40-80 saat | ~$12-24 |
| Faz 4 (VT/SVT fine-tune) | ~20-40 saat | ~$6-12 |
| Faz 5 (LLM testleri) | ~5-10 saat | ~$1.5-3 |
| **Toplam** | **~90-175 saat** | **~$27-53** |

**Not:** Bu cok kaba tahminler. Gercek maliyet hiperparametre arama, deneme sayisi ve veri boyutuna gore degisir. En kotu senaryoda bile $100'u gecmesi beklenmez.

### 6.3 Yazilim Bagimliliklari

```
Python 3.11
PyTorch >= 2.4.0
torchvision >= 0.19.0
nnUNet (ECG-Digitiser icin)
wfdb >= 4.2.0
numpy, pandas, scipy, scikit-learn
matplotlib (gorsellestirme)
opencv-python (goruntu isleme)
tensorflow 2.14.0 (ECG-Digitiser icin)
```

### 6.4 Veri Depolama

| Veri Seti | Boyut (tahmini) |
|-----------|----------------|
| PTB-XL (sinyaller) | ~3 GB |
| PTB-XL sentetik goruntuler | ~10-20 GB |
| MIMIC-IV-ECG | ~100+ GB |
| Model agirliklari | ~1 GB |
| **Toplam** | **~115-125 GB** |

---

## 7. Proje Yapisi (Kod Organizasyonu)

```
corio-ecg/
├── SPEC.md                          # Bu dosya
├── CLAUDE.md                        # Proje ozel Claude talimatları
├── README.md
├── pyproject.toml                   # Python proje konfigurasyonu
│
├── data/
│   ├── raw/                         # Ham veri setleri (gitignore)
│   │   ├── ptb-xl/
│   │   └── mimic-iv-ecg/
│   ├── processed/                   # Islenmis veri
│   │   ├── signals/                 # Dijitize edilmis sinyaller
│   │   └── images/                  # Sentetik ECG goruntuleri
│   └── splits/                      # Train/val/test bolumleri
│
├── models/
│   ├── digitiser/                   # ECG-Digitiser agirliklari
│   ├── ecgfounder/                  # ECGFounder agirliklari
│   │   ├── base/                    # Orijinal agirliklar
│   │   └── finetuned/              # Fine-tune edilmis versiyonlar
│   └── llm/                         # Rapor motoru modeli (ileride)
│
├── src/
│   ├── pipeline/
│   │   ├── digitize.py              # Goruntu → sinyal donusumu
│   │   ├── diagnose.py              # Sinyal → tani
│   │   ├── report.py                # Tani → yapilandirilmis rapor
│   │   └── run.py                   # Tam pipeline orchestration
│   │
│   ├── training/
│   │   ├── fine_tune.py             # ECGFounder fine-tuning
│   │   ├── data_loader.py           # Veri yukleme ve augmentasyon
│   │   ├── augmentation.py          # Dijitizasyon noise injection
│   │   └── evaluate.py              # Performans degerlendirme
│   │
│   ├── vtsvt/                       # VT/SVT ozel modul
│   │   ├── criteria.py              # Brugada, Vereckei implementasyonu
│   │   ├── features.py              # Morfolojik feature cikarimi
│   │   └── train.py                 # VT/SVT fine-tuning
│   │
│   ├── utils/
│   │   ├── ecg_render.py            # Sinyal → sentetik goruntu uretimi
│   │   ├── wfdb_helpers.py          # WFDB format islemleri
│   │   └── metrics.py               # AUROC, sensitivity, specificity
│   │
│   └── web/                         # Test arayuzu (Faz 6)
│       └── app.py                   # Gradio/Streamlit arayuzu
│
├── notebooks/                       # Jupyter notebook'lar (deneme/analiz)
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_digitization_impact.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_vtsvt_analysis.ipynb
│
├── configs/
│   ├── training.yaml                # Egitim hiperparametreleri
│   └── report_template.yaml         # Rapor sablonu
│
├── scripts/
│   ├── download_ptbxl.sh            # PTB-XL indirme
│   ├── download_mimic.sh            # MIMIC-IV-ECG indirme
│   └── setup_environment.sh         # Ortam kurulumu
│
└── results/
    ├── metrics/                     # Performans metrikleri
    ├── figures/                     # Makale icin grafikler
    └── reports/                     # Ornek uretilmis raporlar
```

---

## 8. Basari Metrikleri

### 8.1 Dijitizasyon Kalitesi
- **SNR (Signal-to-Noise Ratio):** Dijitize sinyal vs orijinal sinyal
- **Korelasyon katsayisi:** Lead bazinda Pearson korelasyonu
- **Hedef:** SNR > 15 dB

### 8.2 Tani Performansi
- **AUROC:** Her tani sinifi icin (birincil metrik)
- **Sensitivity (Duyarlilik):** Ozellikle kritik tanilar icin (STEMI, VT)
- **Specificity (Ozgulluk):** Yanlis pozitif oranini minimize etmek
- **F1-Score:** Dengesiz siniflar icin
- **Hedef:** Mevcut ECGFounder baseline'in uzerinde istatistiksel olarak anlamli iyilesme

### 8.3 VT/SVT Ozel Metrikler
- VT sensitivity > %95 (kacirilan VT olumcul olabilir)
- SVT specificity > %90
- Brugada/Vereckei kriterleri ile uyum orani

### 8.4 Rapor Kalitesi
- Kardiyolog degerlendirmesi (varsa): Likert olcegi (1-5)
- Tani-rapor tutarliligi: Rapordaki bulgular ile model ciktisi arasindaki uyum

---

## 9. Riskler ve Azaltma Stratejileri

| Risk | Olasilik | Etki | Azaltma |
|------|----------|------|---------|
| Dijitizasyon kalitesi dusuk | Orta | Yuksek | Augmentasyon + fine-tuning ile tolerans artirma |
| VT/SVT veri yetersizligi | Yuksek | Yuksek | Birden fazla veri setinden toplama + sentetik veri |
| Fine-tuning catastrophic forgetting | Orta | Yuksek | Dusuk learning rate, LoRA/adapter yontemleri |
| MIMIC-IV-ECG veri boyutu (100+ GB) | Dusuk | Orta | Subset ile basla, gerekirse tamamini kullan |
| LLM rapor hallusinasyonu | Orta | Orta | Strict prompt engineering, sadece model ciktisina dayali rapor |
| GPU maliyeti beklenenden yuksek | Dusuk | Dusuk | Google Colab ucretsiz tier ile baslama |

---

## 10. Kisitlamalar ve Kapsam Disi

### Kapsam ICINDE
- ECG-Digitiser + ECGFounder pipeline kurulumu
- Fine-tuning (dijitizasyon dayanikliligi, genel iyilestirme, VT/SVT)
- LLM ile yapilandirilmis rapor uretimi
- Basit test arayuzu
- Akademik yayin hazirligi

### Kapsam DISINDA (su an icin)
- Mobil uygulama gelistirme (ileride ayri proje)
- Gercek hasta verisi toplama (etik kurul sureci ayri)
- FDA/CE sertifikasyon sureci
- Ticari urun gelistirme
- Gerçek zamanli (real-time) monitoring
- Holter ECG analizi
- Tek-lead ECG (Apple Watch vb.) destegi

---

## 11. Karar Kaydi

| Karar | Secilen | Neden |
|-------|---------|-------|
| Model yaklasimi | Fine-tuning (sifirdan degil) | Mevcut modeller cok guclu, kaynak tasarrufu |
| Base model | ECGFounder (76.3M param) | 150 tani, NEJM AI yayini, hafif, AUROC >0.95 |
| Dijitizasyon araci | ECG-Digitiser | PhysioNet 2024 birincisi, BSD-2 lisans |
| Girdi formati | Kagit ECG fotografi/taramasi | Klinik ortamda en yaygin senaryo |
| Cikti formati | Yapilandirilmis rapor (LLM ile) | Klinik kullanima uygun |
| Veri setleri | PTB-XL + MIMIC-IV-ECG | Acik erisim, buyuk, iyi etiketlenmis |
| Ozel odak | VT/SVT ayirimi (Brugada/Vereckei) | Hayat kurtaran karar, akademik bosluk |
| Gelistirme ortami | Mac Mini M4 + Vast.ai/Colab | Dusuk maliyet, yeterli performans |
| LLM secimi | Sonra karar verilecek | Acik kaynak oncelikli |
| Test arayuzu | Gradio veya Streamlit | Basit, hizli kurulum, sadece test amacli |

---

## 12. Referanslar

- **ECGFounder:** Li et al., "ECGFounder: A Foundation Model for ECG Analysis," NEJM AI, 2024
- **ECG-Digitiser:** Krones et al., PhysioNet Challenge 2024 Winner
- **ECG-Image-Kit:** Shaker et al., "ECG-Image-Kit: A Toolkit for Synthesis, Analysis, and Digitization of ECG Images," Physiological Measurement, 2024
- **PTB-XL:** Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," Scientific Data, 2020
- **MIMIC-IV-ECG:** Gow et al., PhysioNet, open access
- **Brugada Criteria:** Brugada et al., "A New Approach to the Differential Diagnosis of a Regular Tachycardia with a Wide QRS Complex," Circulation, 1991
- **Vereckei Algorithm:** Vereckei et al., "New Algorithm Using Only Lead aVR for Differential Diagnosis of Wide QRS Complex Tachycardia," Heart Rhythm, 2008
