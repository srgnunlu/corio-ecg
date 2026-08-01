<!-- Purpose: Evidence-based strategy for choosing Corio ECG's first specialist diagnostic model. -->

# Corio ECG uzmanlaşma stratejisi

**Tarih:** 2026-07-06  
**Karar sorusu:** Corio ECG hangi dar tanı alanında klinik olarak değerli, teknik olarak savunulabilir ve yayınlanabilir bir üstünlük kurabilir?

## Yönetici özeti

Proje tıkanmış değil; iki farklı problem tek bir “tanı başarısız” hissinde birleşmiş:

1. ECGFounder temiz dijital sinyalde bazı yaygın ritimleri iyi ayırıyor, fakat 150 etiketin önemli kısmında destek az, etiket ontolojisi gürültülü ve herkese uygulanan `0.5` eşiği kalibrasyonsuz.
2. Fotoğraf → sinyal geçişindeki zaman yerleşimi sorunu segment-ensemble ile büyük ölçüde azaltılmış olsa da gerçek fotoğraf, cihaz ve kalibrasyon değişkenliği hâlâ klinik bir kalite kapısı gerektiriyor.

Bu nedenle “150 tanıyı biraz daha iyi yapan genel model” doğru hedef değil. Önerilen ana yön:

> **Corio-OMI: kâğıt/fotoğraf girdisine doğal olarak dayanıklı, anjiyografiyle temellendirilmiş, belirsizlikte karar vermeyen bir akut koroner oklüzyon risk modeli.**

Bu öneri **veri erişimine bağlıdır**. Mayıs 2026'da 18.909 hastadan 19.955 EKG ve 1.274 anjiyografik OMI içeren yeni bir veri seti yayımlandı. Yayımlanan basit CNN baseline'ı testte duyarlılık `0.697`, özgüllük `0.873`, F1 `0.396` verdi; yani anlamlı iyileştirme alanı var. Ancak makaledeki Figshare DOI'si 2026-07-06 itibarıyla 404 dönüyor. Erişim ve lisans doğrulanmadan ana geliştirme başlamamalı.

**VT–SVT ikinci adaydır, fakat şu an veri problemi yüzünden ilerlememelidir.** Repodaki 53 kayıtlık audit yalnızca negatif kontrol içeriyor; gerçek VT duyarlılığı ölçülmedi. Üstelik 2024'te 3.330 WCT EKG ile eğitilmiş bir CNN %93 doğruluk bildirdi. Corio'nun farklılaşması ancak EP-doğrulanmış WCT verisi ve fotoğraf dayanıklılığıyla mümkün olur.

## 1. Projenin bugünkü gerçek durumu

### 1.1 Tanı modeli tamamen başarısız değil

Tam PTB-XL test fold'unda yerel ECGFounder entegrasyonu upstream modelle logit düzeyinde eşleştirildi:

- 2.198 etiketli kayıtta macro AUROC: `0.8679`
- Macro average precision: `0.4454`
- Micro F1: `0.5885`
- Macro F1: `0.3673`

Bazı klinik sınıflar güçlü:

| Sınıf | AUROC | F1 |
|---|---:|---:|
| Atriyal fibrilasyon | 0.973 | 0.868 |
| Sinüs taşikardisi | 0.986 | 0.771 |
| PVC | 0.980 | 0.764 |
| LBBB | 0.985 | 0.744 |

Bazı infarct/iletim başlıkları ise zayıf veya kötü eşiklenmiş:

| Sınıf | AUROC | F1 (`0.5` eşik) |
|---|---:|---:|
| Anterior infarct | 0.548 | 0.000 |
| İnferior infarct | 0.805 | 0.029 |
| QRS widening | 0.521 | 0.028 |
| 1. derece AV blok | 0.877 | 0.049 |

Bu tablo iki şeyi söylüyor: genel model bazı ritimler için zaten iyi; tek global eşik ise birçok sınıf için yanlış. İnferior infarct'ta AUROC ile F1 arasındaki uçurum özellikle kalibrasyon/eşik sorununa işaret ediyor. Bu, OMI etiketi değildir ve OMI yerine kullanılamaz.

### 1.2 Fotoğraf problemi artık daha iyi tanımlı

İlk 50 kayıtlık sentetik round-trip değerlendirmesinde fotoğraf yolu yaklaşık `-0.10` macro AUROC ve `-0.19` micro-F1 kaybediyordu. Kök neden yalnızca çizgi çıkarma değil, 3×4 kâğıtta farklı lead'lerin farklı 2,5 saniyelik zaman pencerelerinden gelmesiydi.

Segment-ensemble yaklaşımı her eşzamanlı kâğıt sütununu ayrı değerlendirerek sentetik round-trip macro AUROC'yi yaklaşık `0.75 → 0.87` taşıdı. PMcardio'nun 70 eşleşmiş görüntüsünde medyan waveform-shape korelasyonu `0.823`, başarı `70/70` oldu. Bunlar güçlü mühendislik kazanımlarıdır; yine de dış veri üzerinde klinik tanı doğruluğu anlamına gelmez.

2026'da yayımlanan açık kaynak Ahus digitizer'ı gerçek telefon fotoğraflarında ortalama 10–12 dB SNR bildirdi ve gerçek fotoğrafın hâlâ taramadan belirgin zor olduğunu gösterdi. Corio'nun skorları farklı veri/uyum protokolüyle hesaplandığı için doğrudan sayısal karşılaştırma yapılamaz; aynı benchmark'ta head-to-head test gerekir. [Ahus çalışması](https://www.nature.com/articles/s41746-025-02327-1)

### 1.3 VT/SVT modülü şu an bir model değil, güvenli bir kriter prototipi

Mevcut audit:

- 53/53 başarılı negatif kontrol
- Yalnızca 4 kayıt gerçek WCT scope'una girdi
- 0 VT-pozitif örnek
- 0 false positive, görünen özgüllük `1.0`
- Duyarlılık hesaplanamaz

Bu sonuç “VT/SVT çözüldü” değil, “motor negatif kontrollerde aşırı çağrı yapmadı” demektir. EP çalışması, cihaz içi elektrogram veya tartışmasız takip EKG'siyle doğrulanmış pozitifler olmadan burada model eğitilemez.

## 2. Aday alanların karşılaştırması

Skorlar stratejik karar yardımıdır; bilimsel ölçüm değildir. `5` daha iyi durumu gösterir.

| Aday | Klinik değer | Altın standart veri | Fotoğraf sinerjisi | Farklılaşma | Bugün uygulanabilirlik | Karar |
|---|---:|---:|---:|---:|---:|---|
| Fotoğraf-native OMI riski | 5 | 4* | 5 | 4 | 4* | **Birinci tercih** |
| VT vs SVT-aberransi | 5 | 1 | 4 | 3 | 2 | Veri ortaklığına bağlı |
| AF/PVC/yaygın aritmi | 4 | 5 | 3 | 1 | 5 | Benchmark için iyi, amiral gemisi değil |
| 150 etiket genel yorum | 4 | 3 | 4 | 1 | 3 | Kapsam fazla geniş |
| Tanıyı koruyan digitizasyon + abstention | 4 | 4 | 5 | 4 | 5 | Paralel yayın/altyapı hattı |

`*` Yeni OMI veri setinin fiilî erişimi ve lisansı doğrulanırsa.

### Neden yaygın aritmi değil?

AF, PVC, RBBB/LBBB gibi alanlarda açık veri bol ve mevcut model zaten iyi. Chapman–Shaoxing verisi 45.152 uzman etiketli EKG içeriyor; MIMIC-IV-ECG yaklaşık 800.000 EKG sunuyor. Bu veri bolluğu deneyi kolaylaştırıyor ama “rakipsiz” farklılaşmayı zorlaştırıyor. [Chapman–Shaoxing](https://www.physionet.org/content/ecg-arrhythmia/1.0.0/), [MIMIC-IV-ECG](https://www.physionet.org/content/mimic-iv-ecg/1.0/)

### Neden bugün VT/SVT değil?

Alan klinik olarak çok güçlü, fakat açık veriler çoğunlukla “SVT” veya “VT” rapor etiketi verir; aynı WCT epizodunda VT vs SVT-aberransi için EP-temelli altın standart sağlamaz. Klasik algoritmaların bağımsız karşılaştırmasında doğruluklar orta düzeyde kalmıştı; Brugada doğruluğu %77,5 ve özgüllüğü %59,2 idi. [Europace karşılaştırması](https://academic.oup.com/europace/article/14/8/1165/464643)

Ayrıca doğrudan rekabet var: 3.330 WCT EKG'de eğitilen 2024 CNN çalışması testte %93 doğruluk ve %91,9 VT duyarlılığı bildirdi. [Canadian Journal of Cardiology](https://www.sciencedirect.com/science/article/pii/S0828282X24002964)

Corio bu alanı ancak şu veriyle yeniden açmalı:

- En az yüzlerce **gerçek WCT epizodu**
- VT/SVT kararı EP study, intrakardiyak kayıt veya kesin takip EKG'siyle doğrulanmış
- Hasta düzeyinde ayrılmış dış merkez testi
- Kâğıt/fotoğraf eşleri veya güvenilir sentetik baskı-fotoğraf protokolü

### Neden OMI?

OMI, yalnızca “STEMI etiketi”nden daha klinik bir hedeftir; hedef acil revaskülarizasyon gerektiren akut oklüzyonu öngörmektir. Güçlü yayımlanmış modeller zaten vardır:

- 18.616 EKG ile geliştirilen PMcardio OMI modeli dış testte AUROC `0.938`, duyarlılık `%80,6`, özgüllük `%93,7` bildirdi. Etiketler anjiyografi ve klinik adjudikasyona dayanıyordu. [EHJ Digital Health](https://academic.oup.com/ehjdh/article/5/2/123/7453297)
- 2026 SwED çalışması 540.372 acil EKG'si ve kateterizasyon sonuçlarıyla OMI için C-statistic `≥0.95` ve culprit damar lokalizasyonu bildirdi; ana veri açık değil. [Nature Communications](https://www.nature.com/articles/s41467-026-73023-1)

Dolayısıyla “dünyanın en iyi clean-signal OMI modeli” iddiası gerçekçi değildir. Açık kalan savunulabilir niş şudur:

> **Telefonla çekilmiş kâğıt EKG'de, görüntü kalitesini hesaba katan ve güvenilmez durumda abstain eden OMI risk modeli.**

Bu niş Corio'nun digitizasyon birikimini doğrudan klinik hedefe bağlar.

## 3. Kritik fırsat: 2026 açık OMI veri seti

Yeni Chongqing veri seti şu özellikleri bildiriyor:

- 19.955 adet 10 saniye, 12-lead, 500 Hz EKG
- 18.909 hasta; tümü koroner anjiyografi geçirmiş
- 1.274 OMI (`807 STEMI-OMI + 467 NSTEMI-OMI`)
- OMI tanımı: PCI öncesi culprit damarda akut oklüzyon ve TIMI 0–1
- Ham waveform, median beat, EKG–anjiyografi zaman aralığı, CTO, prior PCI, pacing, VF/VT ve segment düzeyinde culprit damar alanları
- 90/10 train/test; test etiketleri gizli, çevrimiçi değerlendirme sunucusu bildirilmiş
- Baseline: sensitivity `0.697`, specificity `0.873`, PPV `0.277`, NPV `0.976`, F1 `0.396`
- İki EKG uzmanı testte F1 `0.330` ve `0.378` bildirmiş

Kaynak: [Scientific Data 2026 veri tanımı](https://www.nature.com/articles/s41597-026-07278-0)

### Erişim riski

Makale Figshare DOI `10.6084/m9.figshare.29925314` veriyor, fakat DOI ve Figshare API kaydı 2026-07-06 tarihinde bulunamadı. Makale de “article in press” sürümüdür. İlk karar kapısı şunları doğrulamalı:

1. Veri gerçekten indirilebiliyor mu?
2. Veri lisansı akademik fine-tuning, ağırlık paylaşımı ve ileride ticari kullanım için ne diyor?
3. Aynı hastanın birden fazla EKG'si train/test arasında ayrılmış mı?
4. Train içinde yayınlanabilir bir validation split oluşturmak için hasta ID var mı?
5. Gizli test sunucusu erişilebilir ve sonuçlar saklanabilir mi?
6. Baseline kodu, split ID'leri ve model ağırlıkları gerçekten açık mı?

Bu altı sorudan ilk dördü çözülmeden eğitim yapılmamalı.

## 4. Önerilen model: Corio-OMI

### 4.1 Ürün/araştırma iddiası

Çıktı “OMI tanısı” değil:

- `yüksek / orta / düşük akut koroner oklüzyon riski`
- `acil uzman değerlendirmesi gerekir`
- `görüntü/sinyal güvenilmez — karar verilemedi`
- Olası culprit bölgesi yalnızca ayrı, doğrulanmış yardımcı çıktı

Model kateterizasyonun veya klinik değerlendirmenin yerine geçmez.

### 4.2 Basit başlayacak mimari

**Aşama A — clean signal specialist**

1. ECGFounder backbone + yeni binary OMI head, önce backbone frozen.
2. Son katmanlar kademeli açılarak fine-tune.
3. Aynı veride güçlü ama küçük bir 1D ResNet baseline.
4. Ham 10 saniyelik sinyal ile median-beat girdisini karşılaştır.
5. Multi-task yardımcı head'ler: OMI, STEMI/NSTEMI, culprit territory, CTO.

İlk turda yeni foundation model peşinde koşmamak daha doğru; veri hattı ve benchmark güvenilir olmadan mimari araması sonuç üretmez.

**Aşama B — photo-native robustness**

1. Her clean EKG'den 3×4, 6×2 ve farklı cihaz/gain/speed görselleri üret.
2. Mevcut digitizer ve en az bir bağımsız open-source digitizer ile sinyal çıkar.
3. Clean sinyal, render görüntü ve re-digitized sinyalin OMI olasılıklarını tutarlı olmaya zorla.
4. Doğrudan image branch ile digitized-signal branch'i karşılaştır; yalnızca doğrulanırsa ensemble yap.
5. Ayrı quality head, başarısız girdide sınıf tahmini yerine abstain üretsin.

**Aşama C — açıklanabilirlik**

- Lead/time saliency tek başına “klinik açıklama” sayılmamalı.
- ST/T ve reciprocal-change ölçümleri yardımcı feature hedefleri olarak kullanılabilir.
- Açıklama, model kararından türetilmiş kanıt + ölçüm güvenilirliği + bilinen kısıt şeklinde verilmelidir.

## 5. Deney tasarımı ve başarı kapıları

### Gate 0 — Mevcut tanı katmanını dürüstleştir (1 hafta)

- PTB-XL fold 9 üzerinde sınıf-bazlı threshold ve calibration öğren.
- Fold 10'u yalnız final audit için kullan.
- Destek yetersiz 150 head'i UI'da “araştırma çıktısı” veya gizli sınıfa taşı.
- AUROC yanında AUPRC, F1, Brier/ECE ve decision curve raporla.
- Modelin doğrulanmış güçlü sınıfları ile zayıf sınıfları ayrı sun.

Bu çalışma yeni model değildir; mevcut “tanı kötü” hissinin kalibrasyon kaynaklı kısmını temizler.

### Gate 1 — OMI veri erişimi (en fazla 1 hafta aktif çalışma)

- DOI/lisans/split/baseline erişimini doğrula.
- Yazarlara kısa teknik erişim e-postası gönder.
- Erişim yoksa PTB-XL “acute MI” ile OMI sonucu iddia etme.
- Erişim açılana kadar yalnız veri loader ve sentetik test fixture geliştir.

### Gate 2 — Clean-signal baseline (2–3 hafta)

- Hasta düzeyinde train/validation ayrımı.
- Hidden test'e yalnız önceden yazılmış protokolle gönderim.
- Birincil metrik: OMI AUPRC ve validation'da önceden seçilmiş özgüllükte duyarlılık.
- Ayrı rapor: tüm OMI ve özellikle NSTEMI-OMI.
- CTO, pacing, VF/VT, prior PCI, yaş ve cinsiyet alt grupları.
- Bootstrap %95 güven aralıkları ve kalibrasyon.

**İlk go/no-go hedefi:** yayımlanmış baseline'ın `F1 0.396 / sensitivity 0.697 / specificity 0.873` dengesini anlamlı güven aralığıyla aşmak. Yalnız AUROC artışı yeterli değildir.

### Gate 3 — Photo robustness (3–5 hafta)

- Clean sinyal performansını dondur.
- Aynı hastanın clean/render/digitized örneklerini aynı split'te tut.
- Her fiziksel capture tipi için performans düşüşünü ayrı ölç.
- Hedef: kabul edilen fotoğraflarda clean-signal duyarlılığını korumak; bozuk fotoğrafta zorla tahmin yerine abstain.
- Gerçek prevalence ile PPV/NPV ve false-alert burden raporla.

### Gate 4 — Dış doğrulama

PTB-XL acute MI, OMI için gerçek dış doğrulama değildir; anjiyografik label yoktur. Dış merkez OMI verisi bulunmadan klinik üstünlük iddia edilmemeli. En gerçekçi seçenekler:

1. Veri seti yazarlarıyla dış değerlendirme işbirliği.
2. Türkiye'de kardiyoloji/acil tıp + girişimsel kardiyoloji ortaklığı ve etik kurul.
3. Yayımlanmış küçük fotoğraf-OMI kohortlarının yazarlarıyla blinded external test.

## 6. Paralel ama kontrollü ikinci hat

OMI veri erişimi beklenirken ekip boşta kalmamalı. Paralel çalışma:

> **Diagnosis-preserving digitization:** Fotoğraf bozulmasının hangi klinik kararları bozduğunu ölçen, kaliteye göre abstain eden benchmark.

Bu hat için:

- Corio digitizer, Open-ECG-Digitizer ve mümkünse ECGtizer aynı matched set'te karşılaştırılır.
- SNR/korelasyonun yanında downstream tanı delta'sı, interval/axis/ST ölçüm hatası raporlanır.
- “İyi görünen waveform” ile “tanıyı koruyan waveform” ayrımı ana araştırma sorusu yapılır.

PhysioNet 2024 challenge sonuçları da SNR'nin klinik yorumla tam örtüşmediğini açıkça göstermiştir; sınıflandırma lideri extended hidden set'te macro-F1 `0.73` elde ederken digitizasyon ve sınıflandırma sıralamaları birebir aynı değildi. [Challenge raporu](https://moody-challenge.physionet.org/2024/papers/cinc_paper.pdf)

Bu çalışma amiral gemisi OMI modelini besler ve veri erişimi gecikse bile yayınlanabilir çıktı üretir.

## 7. Yapılmaması gerekenler

- 150 sınıfı aynı anda fine-tune edip “genel doğruluk arttı” demek.
- PTB-XL infarct etiketlerini OMI ground truth gibi kullanmak.
- Test setinde threshold seçmek.
- Aynı hastanın EKG veya fotoğraf varyantlarını farklı split'lere koymak.
- Sentetik fotoğraf başarısını gerçek telefon fotoğrafı başarısı diye sunmak.
- AUROC tek başına raporlamak; düşük prevalansta AUPRC, PPV ve calibration zorunlu.
- Kalitesiz görüntüde her koşulda tanı üretmek.
- Negatif-kontrol VT audit'ini sensitivite kanıtı olarak yorumlamak.
- “Rakipsiz” veya klinik kullanım iddiasını dış merkez ve prospektif değerlendirmeden önce kullanmak.

## 8. Net karar

1. **Amiral gemisi hedef:** Corio-OMI.
2. **Moat:** OMI'nin kendisi değil; fotoğraftan OMI riskine giden kalite-farkındalıklı, abstention'lı uçtan uca sistem.
3. **İlk engel:** veri erişimi ve lisans; model mimarisi değil.
4. **VT/SVT:** kod korunmalı, fakat EP-doğrulanmış pozitif kohort gelene kadar araştırma prototipi olarak dondurulmalı.
5. **Yaygın aritmi:** kalite kontrol ve baseline amaçlı kullanılmalı; ana farklılaşma alanı yapılmamalı.
6. **Hemen yapılacak teknik iş:** mevcut 150-head çıktısını sınıf-bazlı kalibrasyon ve destek filtresiyle dürüstleştirmek.

## Metodoloji ve temel kaynaklar

Araştırma; proje SPEC'i, 22 oturum özeti, yerel benchmark artefaktları ve güncel birincil/kurumsal kaynaklar üzerinden yürütüldü. Web taramasında OMI, WCT/VT-SVT, açık ECG veri setleri, ECG foundation model eğilimleri ve paper-ECG digitizasyonu ayrı alt sorular olarak incelendi. Ana karar için en yüksek ağırlık anjiyografik/EP-temelli altın standarda, hasta-düzeyi ayrımlara ve dış doğrulama imkânına verildi.

Temel kaynaklar:

1. [Açık anjiyografi-temelli ACS/OMI veri seti, Scientific Data 2026](https://www.nature.com/articles/s41597-026-07278-0)
2. [540k EKG'de OMI ve culprit lokalizasyonu, Nature Communications 2026](https://www.nature.com/articles/s41467-026-73023-1)
3. [Uluslararası OMI AI değerlendirmesi, EHJ Digital Health 2024](https://academic.oup.com/ehjdh/article/5/2/123/7453297)
4. [Fotoğrafla toplanmış küçük ACOMI modeli, Archivos de Cardiología de México 2025](https://pubmed.ncbi.nlm.nih.gov/40020200)
5. [WCT CNN karşılaştırması, Canadian Journal of Cardiology 2024](https://www.sciencedirect.com/science/article/pii/S0828282X24002964)
6. [WCT kriterlerinin bağımsız karşılaştırması, Europace 2012](https://academic.oup.com/europace/article/14/8/1165/464643)
7. [Open paper ECG digitizer, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-025-02327-1)
8. [PhysioNet Challenge 2024 resmî raporu](https://moody-challenge.physionet.org/2024/papers/cinc_paper.pdf)
9. [PTB-XL](https://physionet.org/content/ptb-xl/1.0.2/)
10. [MIMIC-IV-ECG](https://www.physionet.org/content/mimic-iv-ecg/1.0/)
11. [Chapman–Shaoxing 45k arrhythmia seti](https://www.physionet.org/content/ecg-arrhythmia/1.0.0/)
12. [2025 ACC/AHA ACS guideline ana sayfası](https://professional.heart.org/en/science-news/2025-guideline-for-the-management-of-patients-with-acute-coronary-syndromes)

