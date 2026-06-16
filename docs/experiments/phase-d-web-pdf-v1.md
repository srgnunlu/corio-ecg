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

## LLM doğal-dil özeti (D.2 nihai hedef — TAMAMLANDI 2026-06-17)

Deterministik raporun yanına **Claude Opus 4.8** ile doğal-dil özet eklendi.

- **Modül** `src/report/llm_narrative.py`: `generate_narrative(report, language)`.
  Girdi = `report_to_dict(report)` JSON. Model `claude-opus-4-8`, thinking yok
  (kısa/grounded özet → hız+maliyet için adaptive gereksiz), max_tokens 1024,
  non-streaming.
- **Halüsinasyon koruması (sıkı system prompt):** yalnızca verilen bulguları
  özetler; listede olmayan tanı/ritim/ölçüm eklemez; null değeri "ölçülemedi"
  der; tedavi/karar vermez; sonunda "bu AI özetidir, hekim doğrulamalı" cümlesi.
  3 paragraf: karar+HR/ritim, intervaller, top tanılar.
- **Best-effort:** `ANTHROPIC_API_KEY` yoksa veya API hata verirse `None` döner;
  rapor yine deterministik çalışır. Key `.env`'den (`python-dotenv`).
- **Dil:** TR/EN seçici (web'de dropdown). **Tetikleme:** her analizde otomatik
  (key varsa). Web'de mor "🧠 AI Summary" kartı; PDF'e "AI Summary" bölümü.
- **⚠️ Font bug yakalandı + çözüldü:** reportlab varsayılan Helvetica Türkçe
  glyph (ş/ı/ğ/İ) içermiyordu → PDF'te kutu (□) çıkıyordu. matplotlib ile gelen
  **DejaVuSans** TTF kaydedildi (`registerFont`+`registerFontFamily`), tüm
  stiller bu fonta geçirildi. Kayıt başarısız olursa Helvetica'ya düşer.
- **⚠️ GÜVENLİK:** API'ye sadece yapılandırılmış bulgular (HR, interval, tanı
  etiketleri) gider — hasta kimliği/PHI gitmez. Gerçek hasta verisi devreye
  girince dış-servis veri aktarımı sayılır; KVKK/onam ayrıca değerlendirilmeli.
- **Maliyet:** rapor başına ~$0.013 (Opus 4.8, ~600 giriş + ~400 çıkış token).
- **Bağımlılık:** `anthropic>=0.40.0`. **Test:** 6 yeni (`test_llm_narrative.py`,
  fake client — gerçek API çağrısı yok) + PDF narrative testi. Suite **327 passed**.

## Bilinen sınırlamalar / sonraki adımlar

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
