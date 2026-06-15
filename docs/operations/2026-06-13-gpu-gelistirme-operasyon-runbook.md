# Corio ECG GPU, Geliştirme ve Operasyon Runbook'u

**Tarih:** 13 Haziran 2026
**Amaç:** Mac Mini M4 geliştirme sınırları, Vast.ai GPU iş akışı, ekip ve bütçe planı
**İlişkili yol haritası:** [Profesyonel ürün yol haritası](../plans/2026-06-13-profesyonel-urun-yol-haritasi.md)

## 1. Mac Mini M4 Çalışma Planı

### Mac Üzerinde Yapılacaklar

- Kod geliştirme, unit/integration test, küçük benchmark ve Gradio araştırma demosu.
- 20-200 kayıtlık smoke değerlendirmeler.
- Dataset manifest, rapor ve görselleştirme.
- Küçük head fine-tune veya CPU/MPS uygun deneyler.

### Mac Üzerinde Yapılmayacaklar

- Tam MIMIC-IV eğitimi veya geniş hyperparameter search.
- Bellek kontrolü olmayan dewarping batch'i.
- PHI içeren üretim/pilot veri işleme.
- Public Gradio tunnel ile klinik vaka paylaşımı.

### Önerilen Yerel Kurulum

- Şifreli 2-4 TB harici NVMe; açık veri, cache ve artifact ayrımı.
- `mise` ile Python, lockfile ile dependency, pre-commit ve CI ile aynı komutlar.
- Geliştirme verisi açık/de-identified; hassas veri için ayrı şifreli kontrollü ortam.
- Günlük kaynak kod Git, büyük artifact object storage/registry, yerel cache silinebilir olmalı.

## 2. Vast.ai GPU Runbook

Vast instance'ları Docker container olarak çalışır, saniye bazlı ücretlenir ve SSH/Jupyter ile
erişilebilir. SSH public key instance oluşturulmadan önce hesaba eklenmelidir
([Vast SSH belgeleri](https://docs.vast.ai/guides/instances/connect/ssh)).

### 2.1 Ne Zaman Kiralanır?

Yalnız şu checklist geçince:

- Küçük subset job'u Mac/Linux container'da başarıyla tamamlandı.
- Tek komut entrypoint, lockfile, container digest ve git commit hazır.
- Checkpoint resume ve artifact upload test edildi.
- Tahmini süre, disk, GPU RAM ve bütçe üst sınırı belirlendi.
- Job yalnız açık/de-identified veri kullanıyor.

### 2.2 Instance Seçimi

| İş | Başlangıç GPU | Disk | Not |
|---|---|---:|---|
| Büyük digitizasyon batch | RTX 4090 24 GB | 250-500 GB | İyi fiyat/performans |
| ECGFounder fine-tune | RTX 4090 24 GB | 300-800 GB | Mixed precision ve checkpoint |
| Büyük batch/çoklu deney | A100 40/80 GB | 500 GB+ | Yalnız ölçülmüş darboğaz varsa |

- Verified/reliable host, direct SSH, yeterli ağ hızı ve CUDA uyumlu image seç.
- Tek seferlik root disk yerine kalıcı volume veya dış object storage kullan.
- Vast volume'ların fiziksel host'a bağlı olduğunu unutma; instance silmeden yedek al
  ([Vast volume belgeleri](https://docs.vast.ai/guides/instances/storage/volumes)).
- Marketplace fiyatı dinamik olduğundan saatlik değil **tam deney maliyet tavanı** belirle.

### 2.3 Güvenli Başlangıç Akışı

```bash
# Mac: Vast için ayrı anahtar
ssh-keygen -t ed25519 -f ~/.ssh/corio_vast -C "corio-vast"

# Sunucuda, private repo erişimi ve secret'lar kontrollü sağlandıktan sonra
git clone <repo-url> /workspace/corio-ecg
cd /workspace/corio-ecg
git checkout <exact-commit>

# Yol haritasında geliştirilecek standart komutlar
make bootstrap-gpu
make smoke-gpu
make train EXPERIMENT=<versioned-config>
make evaluate-locked MODEL=<artifact-id>
make upload-artifacts RUN=<run-id>
```

Mevcut `setup_vps.sh` ve `vps_setup_and_run.sh` araştırma scriptleridir. Profesyonel kullanım
öncesi `git pull`, floating pip paketleri, Kaggle secret environment'ı ve kalıcı artifact eksikleri
giderilmelidir.

### 2.4 Job Sırasında

- `tmux` veya job runner kullan; stdout yanında structured log ve heartbeat üret.
- GPU utilization, VRAM, RAM, disk, loss, validation metric ve ETA izle.
- Her 15-30 dakikada veya her epoch'ta resume checkpoint al.
- İlk saat sonunda metric ilerlemiyorsa job'u durdur.
- Dataset ve locked test setini training container'ında yazılabilir tutma.

### 2.5 Kapatma Checklist'i

- Model, optimizer state, config, metrics, log, SBOM ve environment manifest upload edildi mi?
- Artifact hash doğrulandı mı?
- Locked test sonucu ayrı ve salt okunur mu?
- Volume yedeği alındı mı?
- Instance gerçekten destroy edildi mi?

## 3. Veri ve Artifact Depolama

### Önerilen Katmanlar

| Katman | İçerik | Politika |
|---|---|---|
| Git | Kod, küçük config, schema, doküman | Review ve sürümleme |
| Dataset registry | Dataset manifest, split ve provenance | Değişmez sürüm ve checksum |
| Object storage | Model, log, büyük sonuç ve ara artifact | Encryption, retention ve lifecycle |
| Model registry | Onaylı model, metric, intended use | Promotion gate ve rollback |
| Locked test store | Nihai test seti ve sonucu | Salt okunur, sınırlı erişim |

Her deney en az şu kimliği taşımalıdır: code commit, container digest, dataset/split hash,
model/config hash, seed, hardware, başlangıç-bitiş zamanı ve üretilen artifact hash'leri.

## 4. Minimum Çekirdek Takım

| Rol | Gereksinim |
|---|---|
| Kardiyoloji klinik lideri | Intended use, label adjudication, safety ve çalışma tasarımı |
| ML/ECG lideri | Signal/image modelleri, calibration, validation |
| Data/ML engineer | Dataset registry, training, experiment ve GPU işleri |
| Backend/MLOps/security engineer | API, deployment, audit, monitoring, güvenlik |
| Mobil/frontend engineer | Capture UX ve clinician workflow |
| QA/RA uzmanı | QMS, risk, teknik dosya, regülasyon |
| Biostatistician | Güç analizi, protokol, güven aralığı ve klinik raporlama |

Erken araştırma aşamasında bazı roller aynı kişide olabilir. Klinik lider, QA/RA ve
biostatistician sorumlulukları buna rağmen açıkça atanmalıdır.

## 5. Planlama Bütçesi

| Aşama | Tahmini aralık | Ana maliyet |
|---|---:|---|
| 0-3 araştırma güvenilirliği | `$5k-$25k` | Depolama, GPU, veri operasyonu, uzman zamanı |
| 4-6 klinisyen pilotu | `$25k-$150k+` | Etiketleme, güvenli altyapı, pilot ve istatistik |
| Regülasyon/klinik validasyon | `$150k-$750k+` | QMS, danışmanlık, çalışmalar, denetim/başvuru |

Bu aralıklar karar bütçesidir; ülke, kurum, örneklem ve regülasyon yoluna göre ciddi değişir.
SPEC'teki `$27-$53` toplam GPU tahmini yalnız küçük araştırma deneyi için düşünülebilir;
profesyonel ürün bütçesini temsil etmez.

### GPU Deney Bütçesi Formülü

Her job öncesinde şu hesap kaydedilmelidir:

```text
toplam_tavan = (gpu_saat_ucreti + disk_saat_ucreti) * tahmini_saat * 1.5
```

`1.5` çarpanı ilk planlama tamponudur. Smoke job ölçümü sonrasında gerçek süreyle güncellenir.
Takipsiz ve limitsiz hyperparameter search çalıştırılmaz.

## 6. Güvenlik Kuralları

- Vast ve benzeri marketplace GPU'larda PHI işleme; yalnız açık veya de-identified veri kullan.
- Repo erişimini kısa ömürlü ve minimum yetkili credential ile sınırla.
- `.env`, API anahtarı, dataset credential veya private SSH key'i image/artifact içine koyma.
- SSH password login açma; ayrı anahtar kullan ve iş bitiminde erişimi kaldır.
- Training image'larını digest ile pinle; dependency ve image taraması yap.
- Artifact'ları yüklemeden önce doğrula, şifrele ve hash kaydet.
- Üretim/pilot klinik veri için marketplace yerine sözleşmeli, kontrollü ve audit edilebilir ortam kullan.

## 7. İlk Altyapı Teslimatları

1. Pinlenmiş Python lockfile ve GPU container.
2. `make bootstrap-gpu`, `make smoke-gpu`, `make train` ve `make upload-artifacts`.
3. Resume-capable training ve kontrollü interrupt testi.
4. Dataset/model/run manifest schema ve checksum doğrulaması.
5. Object storage ile minimum yetkili artifact upload.
6. Bütçe tavanı, heartbeat ve başarısız job uyarısı.
7. Temiz instance üzerinde tekrarlanabilir smoke job kanıtı.
