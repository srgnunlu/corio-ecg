<!-- Purpose: Zero-shot reference — what stock ECGFounder already knows about OMI, before any training. -->

# Gate 2 — ECGFounder zero-shot OMI referansı (v1)

**Tarih:** 2026-08-01
**Protokol:** [gate2-omi-protocol-v1.md](../plans/gate2-omi-protocol-v1.md)
**Ölçüm kümesi:** fold 0 (n=3.592, 230 OMI, prevalans %6.40) — resmî test setine dokunulmadı
**Model:** eğitilmemiş ECGFounder, kalibrasyon **kapalı** (ham sigmoid çıktıları)

## Neden bu ölçüm

Fine-tuning'e başlamadan önce bilmemiz gereken şey: ECGFounder OMI hakkında
zaten ne biliyor? Bu, sonraki her adımın kazancını ölçeceğimiz sıfır noktası —
ve bedava, çünkü hiçbir eğitim gerektirmiyor.

150 head arasından post-hoc en iyisini seçmek bir arama yanlılığı olurdu. Bu
yüzden iskemi/infarkt taşıyabilecek **16 head önceden ilan edildi**
(`scripts/evaluate_omi_zeroshot.py::CANDIDATE_HEADS`) ve hepsi raporlanıyor.

## Sonuçlar

Prevalans 0.0640 — AUPRC için taban değer budur.

| Head | AUROC | AUPRC |
|---|---:|---:|
| LATERAL INJURY PATTERN | 0.7498 | 0.2058 |
| ACUTE MI | 0.7305 | 0.1985 |
| INFERIOR INJURY PATTERN | 0.7479 | 0.1956 |
| ACUTE MI / STEMI | 0.7270 | 0.1930 |
| ANTEROLATERAL INJURY PATTERN | 0.7269 | 0.1777 |
| INFEROLATERAL INJURY PATTERN | 0.6737 | 0.1594 |
| ANTEROLATERAL INFARCT | 0.7117 | 0.1451 |
| POSTERIOR INFARCT | 0.5934 | 0.1426 |
| INFERIOR INFARCT | 0.6670 | 0.1420 |
| INFERIOR-POSTERIOR INFARCT | 0.6026 | 0.1415 |
| ANTERIOR INJURY PATTERN | 0.7015 | 0.1398 |
| ANTEROSEPTAL INFARCT | 0.6473 | 0.1382 |
| ST ELEVATION NOW PRESENT IN | 0.6713 | 0.1338 |
| SEPTAL INFARCT | 0.6404 | 0.1274 |
| ANTERIOR INFARCT | 0.6761 | 0.1246 |
| LATERAL INFARCT | 0.6793 | 0.1182 |

**Basit kombinasyonlar:**

| | AUROC | AUPRC |
|---|---:|---:|
| max(16 head) | 0.7499 | 0.1573 |
| **mean(16 head)** | **0.7818** | 0.2048 |

Ortalama, en iyi tekil head'i AUROC'de geçiyor (0.782 vs 0.750) — yani sinyal
tek bir head'de değil, infarkt ailesine dağılmış durumda.

## ⚠️ Asıl bulgu: genel skor yanıltıcı

En iyi head ile alt grup kırılımı:

| Alt grup | n | OMI | AUROC |
|---|---:|---:|---:|
| STEMI etiketli | 295 | 152 | 0.566 |
| NSTEMI etiketli | 231 | 77 | 0.590 |
| **Ne STEMI ne NSTEMI** | **3.066** | **1** | 0.988* |

`*` tek pozitifle hesaplandığı için anlamsız.

Fold 0'ın **%85'i** hiç ACS olmayan hastalardan oluşuyor ve bu 3.066 kayıtta
yalnızca **1** OMI var. Yani AUROC 0.78'in büyük kısmı "bu hastada akut koroner
olay var mı?" ayrımından geliyor — ki bu görece kolay bir soru.

Asıl zor soru — **ACS'li hastalar içinde kim oklüde?** — cevaplandığında model
neredeyse rastgele: AUROC 0.566 (STEMI grubunda) ve 0.590 (NSTEMI grubunda).

Bu, Gate 2'nin bütün hedefini netleştiriyor. İyileştirilecek yer, kolay
negatifleri daha iyi elemek değil; ACS-pozitif hastalar içinde oklüzyonu ayırt
etmek. Protokol bu yüzden her sonucun **iki katmanda** raporlanmasını zorunlu
kılıyor: tüm sette (baseline karşılaştırması) ve ACS-pozitif alt kümede (gerçek
klinik değer).

## Yorum

1. **ECGFounder OMI hakkında bir şey biliyor ama yeterli değil.** AUPRC 0.205,
   prevalansın (0.064) 3.2 katı — yani rastgele değil. Ancak yayımlanmış
   baseline'ın F1 0.396'sına eğitimsiz ulaşmak mümkün görünmüyor.
2. **Sinyal "injury pattern" head'lerinde yoğunlaşıyor**, saf infarkt lokalizasyon
   head'lerinde değil. Bu beklenen: OMI akut bir olaydır, eski infarkt değil.
3. **Fine-tuning'in işi belli:** ACS-pozitif alt kümedeki 0.57–0.59'u yükseltmek.
   Genel AUROC'yi yükseltmek kolay ama klinik olarak boş bir kazanç olur.

## Tekrar üretmek için

```bash
python scripts/build_omi_split.py
CORIO_CALIBRATION=0 python scripts/evaluate_omi_zeroshot.py
```

Çıktı: `results/omi/zeroshot_fold0.json`. Süre ~37 saniye (M4, MPS).
