# Phase D.1 + D.2 — Professional Web UI & PDF Report (v1)

**Tarih:** 2026-06-17
**Branch:** `feature/phase3-failure-triage` (push edilmedi)
**Durum:** Tamamlandı — UI yeniden tasarlandı, PDF üretimi çalışıyor, testler geçti.
**Master roadmap:** Faz D.1 (web arayüzü) + D.2 (PDF rapor) (`docs/plans/master-roadmap-v1.md`)

## Özet (TLDR)

Araştırma prototipi Gradio UI'ı **klinisyenin günlük kullanabileceği bir ürün
arayüzüne** dönüştürdük ve **tek tıkla profesyonel PDF rapor** çıktısı ekledik.

- **D.1 Web arayüzü** (`src/web/app.py`): modern tıbbi görünüm (gradient başlık,
  kart düzeni, özel CSS, responsive), drag&drop + pano + kamera yükleme, analiz
  sırasında progress indicator, renkli karar kartı (Normal/Anormal/Belirsiz),
  HR/interval/ritim/AI-tanı panelleri korundu ve düzenlendi.
- **D.2 PDF rapor** (`src/report/pdf_report.py`, yeni): `ECGReport`'tan reportlab
  ile A4 PDF üretir — başlık, tarih, renkli karar özeti, ritim/hız tablosu,
  interval tablosu (bayrak renkli), AI top-5 tanı, orijinal foto thumbnail +
  dijitalleştirilmiş 12-derivasyon grafiği, her sayfada disclaimer. Web app'te
  "⬇ Download PDF Report" butonu.

**Test:** 7 yeni test (`test_pdf_report.py` 4, `test_web_app.py` 3), tüm suite
**320 passed, 1 skipped**. Yeni dosyalar ruff temiz; mypy yeni modüllerde
fonksiyonel hata yok (kalan uyarılar reportlab/gradio stub eksikliği — codebase'in
mevcut scipy/torchvision deseniyle aynı, `type: ignore[import-untyped]`).

## Tasarım kararları

### Web arayüzü (D.1)

- **Durum aktarımı için `gr.State`.** Analiz sonucu (`AnalysisContext`: report +
  intervals + orijinal görüntü + sinyal görüntüsü) `gr.State`'te tutulur; PDF
  butonu bu state'ten üretir → kullanıcının **baktığı** sonucun PDF'i, yeniden
  çalıştırma yok.
- **Progress indicator** `gr.Progress` ile pipeline aşamalarına bağlandı
  (hazırlık → digitize → tanı → render → done). Çekirdek mantık değişmedi.
- **Yükleme kaynakları** `sources=["upload", "clipboard", "webcam"]` — drag&drop,
  pano yapıştırma ve kamera. Mobilde kamera doğrudan çalışır.
- **CSS responsive.** `@media (max-width: 768px)` ile mobilde başlık/padding
  küçülür; kartlar tek kolona iner (Gradio'nun kendi akışı).
- **⚠️ Gradio 6 değişikliği:** `theme` ve `css` artık `Blocks()` constructor'ından
  `launch()`'a taşındı. `create_app()` sade Blocks döndürür; tema/CSS `main()`'de
  `launch()`'a verilir. (Aksi halde her çağrıda UserWarning.)

### PDF rapor (D.2)

- **Bağımlılık: reportlab** (`>=4.0.0`, pyproject'e eklendi). Hafif, saf-Python,
  BSD lisans, bakımlı. fpdf2'ye tercih edildi: tablo/akış (platypus) desteği daha
  olgun, görsel gömme daha temiz.
- **Katman ilkesi korundu:** PDF modülü **hiçbir klinik bulgu hesaplamaz**; sadece
  `ECGReport`'taki hazır değerleri dizgiler. Roadmap'in "rapor katmanı tanı motoru
  değildir" ilkesiyle uyumlu.
- **Interval bayrak hücreleri kısaltıldı.** `interpret_intervals` zengin metin
  döndürüyor ("prolonged (>200 ms) — consider first-degree AV block"); dar tabloda
  taşıyordu → hücrede tek kelime ("prolonged"/"wide"), tam açıklama renkli karar
  kartında. Anormal bayraklar kırmızı, normal yeşil.
- **Görüntüler** aspect-ratio korunarak kutuya ölçeklenir; orijinal foto küçük
  (sol), dijitalleştirilmiş grafik büyük (sağ). 12-derivasyon grafiği genelde 2.
  sayfaya akar — reportlab sayfa kırılımını otomatik yönetir.
- **Çıktı yeri:** `/tmp/corio-reports/corio-ecg-report-<timestamp>.pdf` (repo
  dışı, .gitignore gerekmez). Gradio `gr.File` ile indirme linki sunar.

## Doğrulama

- Gerçekçi bir Abnormal ECG senaryosu (AFib + LVH, HR 128, PR 210/QRS 130/QTc 470
  hepsi anormal) ile PDF üretildi ve görsel incelendi: karar kartı, tablolar,
  renkli bayraklar, 2-sayfa görüntü yerleşimi ve disclaimer beklendiği gibi.
- Normal, Indeterminate ve görüntüsüz/tanısız yollar testlerle kapsandı.
- **Yalnızca render/üretim doğrulaması** — bu faz sayısal model doğruluğu
  ölçmez; D.1/D.2 sunum katmanıdır.

## Bilinen sınırlamalar / sonraki adımlar

- **LLM doğal-dil özeti henüz yok.** Roadmap D.2 nihai hedefi deterministik
  ölçüm bloğu + LLM (Claude Opus 4.8) doğal-dil özet. Şu an rapor tamamen
  deterministik; LLM narrative ayrı adım (API key `.env`, PHI loglanmaz).
- **Abstention UX (D.1) bekliyor:** quality gate üretime bağlanınca "bu görüntüyü
  okuyamadım, şu açıdan yeniden çekin" mesajı. Gate Faz A'da bilinçli olarak
  üretime bağlanmadı → şimdilik debug paneli uyarıları mevcut.
- "ECG Images" başlığı bazen 1. sayfa sonunda kalıp görüntüler 2. sayfaya akıyor;
  kozmetik, `KeepTogether` ile ileride bağlanabilir.

## Dosyalar

- `src/report/pdf_report.py` — yeni, reportlab PDF üreteci.
- `src/web/app.py` — UI yeniden tasarımı, progress, PDF butonu, `AnalysisContext`.
- `tests/test_pdf_report.py` — yeni (4 test).
- `tests/test_web_app.py` — +3 test (no-image context, generate_pdf yolları).
- `pyproject.toml` — `reportlab>=4.0.0` eklendi.
- `README.md` — yeni özellikler.
