# Corio ECG Profesyonel Ürün Yol Haritası

**Tarih:** 13 Haziran 2026
**Hedef:** Corio ECG'yi kontrollü araştırma prototipinden klinik karar destek ürününe taşımak
**İlişkili değerlendirme:** [Profesyonelleşme değerlendirmesi](../reports/2026-06-13-profesyonellesme-degerlendirmesi.md)

## 1. Program İlkeleri

1. Her faz ölçülebilir bir gate ile kapanır; gate geçmeden sonraki klinik iddia yapılmaz.
2. Model performansı, digitizasyon fidelity, giriş kalitesi ve klinik fayda ayrı değerlendirilir.
3. İlk ürün dar kapsamlı ve locked modeldir; kapsam ve model yalnız kontrollü değişiklik süreciyle büyür.
4. Hasta/site/device sızıntısı engellenir; dış test seti geliştirme sırasında açılmaz.
5. Kritik sınıflarda “bilmiyorum / yeniden çek / uzmana gönder” ürün davranışıdır.
6. LLM klinik karar motoru değildir.
7. Vast.ai yalnız açık veya güvenli biçimde de-identified araştırma verisi için kullanılır.

## 2. Fazlar ve Gate'ler

| Faz | Süre tahmini | Ana çıktı | Çıkış gate'i |
|---|---:|---|---|
| 0. Ürün amacı ve yönetişim | 2-4 hafta | Intended use, risk kaydı, klinik liderlik | Kapsam ve başarı kriterleri onaylı |
| 1. Mühendislik temeli | 4-6 hafta | CI, lockfile, container, modüler kod, experiment registry | Tek komutla tekrarlanabilir build/test |
| 2. Veri ve kalite reddi | 6-10 hafta | Dış doğrulanmış quality gate, eşleşmiş gerçek fotoğraf seti | Kötü girdiler güvenle reddediliyor |
| 3. Digitizasyon v2 | 8-14 hafta | Layout-aware, amplitüd ve morfoloji koruyan pipeline | Dış matched set'te hedef fidelity |
| 4. Tanı motoru v1 | 10-16 hafta | 12-20 sınıf, kalibrasyon, OOD, abstention | Locked external diagnostic test geçildi |
| 5. Ölçüm ve rapor | 8-12 hafta | PR/QRS/QT/QTc/aks/ST + deterministic rapor | Uzman ölçüm/rapor concordance gate |
| 6. Klinik ürün ve pilot | 12-24 hafta | Güvenli API, mobil/web UX, silent pilot | Prospektif güvenlik ve workflow gate |
| 7. Regülasyon ve pazara hazırlık | 9-18 ay paralel | QMS, teknik dosya, klinik değerlendirme | CE/Türkiye başvuru hazırlığı |
| 8. VT/SVT ve akut bakım | 6-18 ay, ayrı program | Gold-standard özel model | Ayrı klinik validasyon ve risk dosyası |

## 3. Faz 0: Ürün Amacı ve Klinik Yönetişim

### İşler

- Kardiyoloji klinik lideri, ML lideri ve regülasyon/QA sorumlusu belirle.
- V1 intended use, kullanıcı, hasta popülasyonu, desteklenen layout ve çekim koşullarını yaz.
- “Araştırma”, “klinik pilot” ve “pazara sunulan ürün” sınırlarını ayır.
- İlk 12-20 bulguyu klinik önem, veri desteği ve doğrulanabilirliğe göre seç.
- Her bulgu için reference standard, minimum destek ve kabul kriteri tanımla.
- ISO 14971 formatında ilk hazard log'u aç.
- Lisans envanteri çıkar: ECG-Digitiser, Open-ECG-Digitizer, PMcardio dataset,
  ECGFounder, ECG-Image-Kit ve tüm model/data türev kullanım koşulları.
- Etik kurul ve veri paylaşım sözleşmesi taslaklarını başlat.

### Gate 0

- İmzalı intended use taslağı.
- V1 sınıf listesi ve kapsam dışı durumlar.
- İlk risk matrisi ve klinik kanıt planı.
- Regülasyon uzmanından ön sınıflandırma görüşü.

## 4. Faz 1: Mühendislik Temeli

### 4.1 Repo ve Kalite

- `uv.lock` veya eşdeğer lockfile oluştur; Python/CUDA/PyTorch kombinasyonunu pinle.
- Linux CUDA ve macOS geliştirme için iki doğrulanmış environment profili tanımla.
- Dockerfile ve digest ile pinlenmiş training/inference image oluştur.
- GitHub Actions: unit tests, Ruff, mypy, dependency audit, secret scan, license scan.
- `digitize.py`, `evaluate_roundtrip.py`, `web/app.py` ve büyük scriptleri 250 satır hedefiyle böl.
- Public API'leri type-safe dataclass/Pydantic modellerine geçir.
- Config'leri tek schema altında versiyonla; hardcoded path ve threshold'ları kaldır.
- Model, dataset, code commit ve config hash'ini her artifact'e yaz.
- Test kapsamını raporla; kritik dönüşüm ve safety rule'larda branch coverage ekle.

### 4.2 Hedef Modül Sınırları

```text
src/
  capture/          # image validation, PHI redaction, page/layout detection
  digitization/     # segmentation, trace extraction, calibration, reconstruction
  measurements/     # beats, intervals, axis, ST/T and morphology
  diagnosis/        # models, ensembles, calibration, thresholds, uncertainty
  quality/          # capture, signal and diagnosis gates
  reporting/        # structured schema, rules, language rendering
  clinical_rules/   # contradiction checks, later VT/SVT criteria
  serving/          # authenticated API and job orchestration
  evaluation/       # locked benchmark runners and statistical reports
```

### 4.3 MLOps

- Deney takibi için MLflow veya eşdeğer registry kur.
- Dataset manifest, DVC/lakeFS veya içerik hash tabanlı veri registry kur.
- Her eğitimde seed, commit, container digest, data manifest, split ve metric artifacts sakla.
- Model card, data card ve release note şablonları ekle.
- Checkpoint resume, early stopping ve başarısız job recovery testleri yaz.

### Gate 1

- Temiz clone'da tek komutla smoke build ve test.
- CI yeşil; yeni kodda Ruff ve mypy hatası yok.
- Reproducibility testi aynı artifact hash veya tanımlı toleransla aynı sonuç veriyor.
- Üçüncü taraf lisans ve SBOM raporu mevcut.

## 5. Faz 2: Eşleşmiş Veri ve Quality Gate

### 5.1 Veri Toplama Seviyeleri

| Seviye | Hedef | İçerik |
|---|---:|---|
| A. Operasyonel | 100+ farklı ECG | Fotoğraf, metadata; yalnız extraction davranışı |
| B. Mühendislik matched | 200+ farklı ECG, 1.000+ görüntü | Dijital WFDB/PDF/scan + telefon varyantları |
| C. Ürün matched | 2+ merkez, 1.000+ ECG | Ardışık vakalar, cihaz/printer/telefon çeşitliliği |
| D. Klinik external | Ayrı merkezler, binlerce vaka | Adjudicated diagnosis, kilitli dış test |

### 5.2 Capture Protokolü

- Her vaka için düz scan/PDF veya orijinal digital waveform sakla.
- Telefon modeli, kamera çözünürlüğü, flash, açı, ışık, mesafe, layout, kağıt hızı/gain kaydet.
- Front, 20-35 derece açı, düşük ışık, gölge, ekran ve kontrollü bozulma varyantları çek.
- PHI redaction kalitesini ayrı doğrula; lead etiketi ve kalibrasyon pulse korunmalı.
- Split'i ECG kimliğine göre yap; aynı ECG'nin varyantları farklı split'e geçmemeli.

### 5.3 Üç Ayrı Gate

1. **Capture gate:** sayfa tam mı, okunabilir mi, glare/blur/occlusion var mı?
2. **Signal gate:** lead mapping, grid calibration, amplitüd, morphology ve timing güvenilir mi?
3. **Diagnosis gate:** model kalibre mi, OOD mi, ensemble uyumlu mu?

### Quality Gate Aday Özellikleri

- Page boundary ve perspective confidence.
- Blur, glare, shadow, occlusion ve compression ölçümleri.
- Layout confidence ve lead-label confidence.
- Active lead sayısı, calibration pulse confidence, px/mm.
- Einthoven ve Goldberger ilişkileri; lead polarity/energy.
- Reconstruction ensemble disagreement.
- Direct-image ile signal-model diagnosis disagreement.
- OOD embedding distance ve diagnosis uncertainty.

### Gate 2

- Threshold'lar geliştirme setinde kilitlenmiş, dış matched sette değerlendirilmiş olmalı.
- Kötü fidelity kayıtları için reject recall hedefi başlangıçta `>=95%`.
- False accept hedefi başlangıçta `<2%`; güven aralığı ayrıca raporlanmalı.
- Kabul edilen koşullarda kategori bazlı fidelity alt sınırları karşılanmalı.
- UI reject vakada tanı göstermemeli, yeniden çekim nedeni vermeli.

Bu eşikler mühendislik başlangıç hedefidir; klinik lider ve risk dosyasıyla kesinleştirilmelidir.

## 6. Faz 3: Digitizasyon v2

### 6.1 Deney Sırası

1. Mevcut 70 görüntüyü frozen baseline olarak koru.
2. Bent/crumpled için page mesh/dewarping ve shadow-aware normalization deneyleri yap.
3. Layout ve lead segmentasyonunu ayrı modeller/başlıklar olarak değerlendir.
4. Full-width rhythm strip ile kısa lead segmentlerini farklı veri tipleri olarak koru.
5. 2.5 saniyelik segmenti sahte 10 saniyeye tile etmek yerine:
   - segment-aware classifier,
   - variable-length/masked signal model,
   - paper-column ensemble,
   - direct image branch
   seçeneklerini karşılaştır.
6. Global z-score yanında calibrated mV sinyalini tanı ve ölçüm motoruna sun.
7. Digitizer uncertainty ve trace overlay üret; hekim görünür şekilde karşılaştırabilsin.

### 6.2 Eğitim

- ECG-Image-Kit ve PhysioNet ECG-Image-Database ile geniş domain randomization.
- Fiziksel matched veriyle fine-tune; sentetikten gerçeğe domain adaptation.
- Loss yalnız pixel/segmentation değil; waveform SNR, interval, morphology ve lead consistency içersin.
- Hard-negative ve failure mining yap; kötü girdileri “başarılı” göstermeyi cezalandır.

### Gate 3

- Dış matched sette scan ve desteklenen iyi telefon fotoğrafında hedef SNR/fidelity geçilmeli.
- QRS/QT/ST gibi klinik ölçümlerin digitizasyon kaynaklı hatası uzmanlar arası değişkenlikle
  karşılaştırılmalı.
- Kategori bazlı regression olmamalı.
- Desteklenmeyen bent/crumpled koşullar kalite gate tarafından reddedilmeli.

## 7. Faz 4: Tanı Motoru v1

### 7.1 Model Stratejisi

- ECGFounder baseline'ını koru; önce veri ve değerlendirme hatalarını kapat.
- 12-20 V1 bulgu için class-specific head veya fine-tuning karşılaştır.
- Signal model + direct-image model + deterministic measurements ensemble geliştir.
- Paired clean/digitized consistency loss ile artifact robustness eğit.
- Sınıf hiyerarşisi ve contradiction rules uygula: normal vs kritik anormallik, bradi vs taki,
  RBBB/LBBB, ritim ve hız tutarlılığı.
- Sınıf başına threshold ve calibration kullan; tek global `0.5` ürün kararı olamaz.
- Temperature/isotonic calibration, deep ensemble ve OOD yöntemlerini locked validation'da kıyasla.

### 7.2 MIMIC-IV Kullanım Sırası

1. 659 kayıtlık demo ile loader/split/label pipeline'ını doğrula.
2. Erişim eğitimleri ve data use koşullarını tamamla.
3. Küçük, hasta-bazlı subset ile label taxonomy ve domain shift analizi yap.
4. Multi-source fine-tuning ablation'ı çalıştır.
5. PTB-XL, MIMIC ve bağımsız site performansını ayrı raporla.

### Gate 4

- Her V1 sınıfı için önceden tanımlı minimum support ve CI.
- Kritik sınıflar için klinik riskle belirlenen sensitivity/NPV hedefleri.
- Sınıf bazlı calibration ve threshold'lar kilitli.
- External test, cihaz/site/kalite alt grupları ve abstention coverage-risk raporu tamam.
- Model card ve intended use ile uyumlu limitation listesi mevcut.

## 8. Faz 5: Ölçüm ve Rapor Motoru

### 8.1 Ölçüm Motoru

- R-peak ve beat segmentation.
- Heart rate ve rhythm regularity.
- PR, QRS, QT, QTc; kullanılan QTc formülü açıkça belirtilmeli.
- Frontal aks.
- Lead bazlı ST elevation/depression ve T-wave polarity.
- Voltage ve hypertrophy kriterleri.
- Pacemaker ve artifact işaretleri.
- Her ölçüm için değer, birim, confidence, kullanılan lead ve kalite durumu.

### 8.2 Rapor Şeması

```json
{
  "input_quality": {"outcome": "accept|warn|reject", "reasons": []},
  "measurements": [],
  "findings": [{"code": "", "probability": 0.0, "calibrated": true}],
  "critical_findings": [],
  "uncertainty": {},
  "limitations": [],
  "model_version": "",
  "audit_id": ""
}
```

- Önce deterministic rules ve template ile Türkçe/İngilizce rapor üret.
- LLM kullanılırsa yalnız bu JSON'u anlatmalı; yeni finding/ölçüm eklemesi schema ile engellenmeli.
- Her rapor model, data, config ve rule sürümüyle audit edilebilir olmalı.

### Gate 5

- Ölçüm doğruluğu uzman ve cihaz referansıyla doğrulandı.
- Rapor-finding tutarlılığı otomatik testlerde `%100`.
- Kritik uyarı, kalite reddi ve uncertainty hiçbir dilde kaybolmuyor.
- Kardiyolog kör değerlendirmesi ve hata taxonomy'si tamamlandı.

## 9. Faz 6: Klinik Ürün Mimarisi

### Bileşenler

- Mobil capture SDK: page guide, blur/glare kontrolü, offline PHI redaction, retake yönlendirmesi.
- Authenticated API: job submission, status, result, audit, deletion.
- Worker queue: digitizer GPU job ve diagnosis job izolasyonu.
- Encrypted object storage ve metadata database.
- RBAC, MFA, organization tenancy, consent/retention politikası.
- FHIR `DiagnosticReport`/`Observation` ve uygun olduğunda DICOM waveform export.
- Observability: latency, failure, quality-gate distribution, model drift, security events.
- Feature flag ve canary release; model rollback.

### Klinik Doğrulama Basamakları

1. Çok merkezli retrospektif external validation.
2. Prospektif silent deployment; sonuç klinisyene gösterilmez.
3. Reader study: klinisyen yalnız, AI yalnız, klinisyen+AI.
4. Sınırlı prospektif assistive pilot.
5. Post-market performance ve vigilance.

### Gate 6

- Silent pilot'ta beklenmeyen safety signal yok.
- İnsan faktörleri/use-error testi tamam.
- Latency, uptime, audit ve rollback hedefleri karşılandı.
- Klinik çalışma raporu STARD-AI/DECIDE-AI ile uyumlu.

## 10. Faz 7: QMS ve Regülasyon Workstream

Bu workstream Faz 0'da başlar ve paralel ilerler.

- ISO 13485 QMS: document control, training, supplier control, CAPA, complaint, change control.
- ISO 14971 risk management file ve benefit-risk.
- IEC 62304 planları: requirements, architecture, SOUP, V&V, maintenance, problem resolution.
- Cybersecurity threat model, SBOM, vulnerability management, penetration test, incident response.
- Clinical Evaluation Plan/Report ve Post-Market Clinical Follow-up.
- Usability engineering file.
- Data protection impact assessment, DPA/BAA ve retention politikaları.
- Model değişiklikleri için Predetermined Change Control yaklaşımı.
- Notified body/TİTCK/FDA strateji görüşmeleri ve gerekiyorsa pre-submission.

## 11. Faz 8: VT/SVT Özel Programı

VT/SVT, V1 genel tanı motoruna küçük bir head ekleyerek çözülecek görev değildir.

- Wide-complex tachycardia intended use ve hasta durumu ayrı tanımlanmalı.
- Gerçek VT, SVT with aberrancy, pre-excited tachycardia ve paced rhythm kohortu kurulmalı.
- Gold standard klinik/EP adjudication olmalı.
- Brugada, Vereckei/aVR ve diğer morfoloji kriterleri deterministic feature motoru olarak uygulanmalı.
- Model; raw signal, ölçümler ve klinik kuralları multi-task/ensemble biçiminde kullanmalı.
- Safety hedefi, yanlış VT reddi ile aşırı VT çağrısı arasındaki klinik maliyete göre belirlenmeli.
- Ayrı retrospektif ve prospektif çalışma yapılmadan ürün kapsamına alınmamalı.

## 12. Operasyon, Vast.ai, Takım ve Bütçe

Mac Mini M4 geliştirme sınırları, Vast.ai kiralama ve kapatma adımları, artifact depolama,
çekirdek ekip ve planlama bütçesi ayrı runbook'ta tanımlanmıştır:
[GPU, geliştirme ve operasyon runbook'u](../operations/2026-06-13-gpu-gelistirme-operasyon-runbook.md).

## 13. İlk 90 Günlük Uygulama Planı

### Gün 1-30

- Intended use ve V1 sınıf listesini klinik liderle kilitle.
- Mevcut quality-gate işini review edip commit'le; sonuçları “development-only” olarak işaretle.
- CI, lockfile, container, lint/type baseline ve license/SBOM ekle.
- Dataset registry ve experiment manifest schema oluştur.
- Level B matched gerçek fotoğraf protokolünü etik/mahremiyet açısından onaylat.
- İlk 50 farklı ECG için matched veri toplama başlat.

### Gün 31-60

- Capture/signal/diagnosis gate'lerini kodda ayır.
- Quality gate'i dış holdout ile yeniden değerlendir.
- `digitize.py`yi modüllere böl; frozen baseline regression suite kur.
- Segment-aware production adayını ve direct-image baseline'ını karşılaştır.
- Ölçüm motoru için R-peak, HR, QRS/QT teknik spike'ları yap.
- Vast için pinlenmiş container ve resume-capable smoke training job hazırla.

### Gün 61-90

- İlk controlled digitization v2 deneylerini bent/crumpled ve iyi telefon kategorilerinde çalıştır.
- Matched veri setini 200 ECG / 1.000 görüntü yönüne büyüt.
- V1 sınıfları için class support ve external validation fizibilite tablosu çıkar.
- Class-specific calibration/threshold baseline kur.
- İlk deterministic structured report schema ve contradiction rules ekle.
- Faz 0-2 gate review yap; eğitim ve klinik pilot kararını kanıta göre ver.

## 14. Hemen Sonraki Sprint Backlog'u

1. `docs/product/intended-use-v1.md` ve `docs/risk/initial-hazard-log.md` oluştur.
2. Quality gate'i bağımsız holdout için train/tune/test ayrımıyla yeniden tasarla.
3. Untracked quality-gate dosyalarını review edip kontrollü commit'e hazırla.
4. GitHub Actions, lockfile, Dockerfile ve SBOM pipeline ekle.
5. `digitize.py` için davranış değiştirmeyen modülerleştirme planı çıkar.
6. Level B matched gerçek fotoğraf manifest ve veri doğrulama scripti yaz.
7. Her output'a code/data/model/config hash ekleyen ortak run manifest geliştir.
8. Segment-aware inference'ı UI'dan bağımsız production-candidate modüle taşı.
9. Capture QC için blur/glare/page-completeness baseline geliştir.
10. İlk ölçüm motoru doğrulama setini ve uzman referans formatını tanımla.

## 15. Programın Başarı Tanımı

Corio ECG profesyonel seviyeye, yalnız yüksek AUROC gördüğümüzde değil, şu soruların tamamına
kanıtla cevap verebildiğimizde ulaşır:

- Hangi görüntüleri güvenle kabul ediyoruz ve hangilerini neden reddediyoruz?
- Kabul edilen görüntüde ECG morfolojisini ne kadar doğru geri kazanıyoruz?
- Her desteklenen tanı için dış merkez performansı ve kalibrasyon nedir?
- Model ne zaman abstain ediyor ve bu davranış klinik riski azaltıyor mu?
- Hekim+AI iş akışı yalnız hekime göre daha güvenli veya faydalı mı?
- Her sonuç, model ve veri sürümüne kadar audit edilebilir mi?
- Değişiklik, güvenlik olayı ve post-market drift yönetilebiliyor mu?
