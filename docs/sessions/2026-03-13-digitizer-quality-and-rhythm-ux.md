# Oturum: Digitizer Quality Recovery + Rhythm-Aware Diagnosis UX
**Tarih:** 2026-03-13
**Sure:** ~3 saat (tahmin)
**Faz:** Phase 6 — Test Web UI (digitizer quality + diagnosis UX)

## Ozet
Open-ECG-Digitizer vendor patch'lerinden gelen kalite regresyonu geri alindi, sessiz sifir-sinyal ciktilari fail-fast hale getirildi, synthetic/UI layout secimi `3x4+1R` ile hizalandi, diagnosis tarafina kalp hizi farkindaligi eklendi ve zorlu tasikardi ornekleri icin UI/timing/debug akisinda iyilestirmeler yapildi. Son asamada, kotu geometri veren gercek fotograf ornekleri icin otomatik dewarping retry fallback'i eklendi.

## Yapilan Isler
- [x] Vendor digitizer kalite regresyonu A/B test ile izole edildi
- [x] Crop ve signal extraction downscale patch'leri kaliteyi bozmayacak sekilde geri cekildi
- [x] Sessiz layout fallback mantigi duzeltildi
- [x] Digitize pipeline'a fail-fast kalite kapilari eklendi (`raw_lines`, `nonzero leads`, `all-NaN canonical`)
- [x] Synthetic digitization script'ine exact `3x4+1R` layout hint eklendi
- [x] Render subprocess'i aktif interpreter (`sys.executable`) ile calisacak sekilde duzeltildi
- [x] Gradio layout secenekleri gercek layout adlariyla hizalandi ve default `3x4+1R` yapildi
- [x] Diagnosis threshold varsayilani `0.7` yapildi
- [x] Kalp hizi tahmini eklendi ve ritimle celisen tanilar rate-aware post-processing ile asagi cekildi
- [x] Tachyarrhythmia alt uyarilari icin ayrik UI kutusu eklendi
- [x] Timing panelindeki cift toplam bug'i duzeltildi
- [x] Diagnosis tarafindaki gereksiz cift inference kaldirildi
- [x] Kotu geometri / bozuk orta kolon vakalari icin dewarping retry fallback'i eklendi
- [x] Unit testler yeni davranislar icin guncellendi/genisletildi

## Alinan Kararlar
| Karar | Secilen | Neden |
|-------|---------|-------|
| Layout default | `3x4+1R` | Hedef ECG kagit formatlari ve synthetic pipeline ile en uyumlu varsayilan bu |
| Diagnosis threshold default | `0.7` | Digitized fotograflarda `0.5` fazla gurultulu bulgular veriyordu |
| Kalite stratejisi | Fail-fast + retry | Bozuk sinyali sessizce kabul etmek yerine once reject, zor vakada tek seferlik fallback |
| Tachy UX | Ayrik "Tachyarrhythmia Considerations" kutusu | ECGFounder multi-label skorlarini threshold altinda da gorunur kilmak gerekiyor |

## Degistirilen/Olusturulan Dosyalar
```
scripts/digitize_synthetic_images.py                   — default/exact layout hint (`3x4+1R`) ve CLI secenegi
src/pipeline/diagnose.py                               — HR tahmini entegrasyonu, rate-consistency adjustments
src/pipeline/digitize.py                               — kalite kapilari, nonzero lead sayimi, retry/dewarping secimi
src/utils/ecg_render.py                                — subprocess interpreter fix (`sys.executable`)
src/utils/rhythm.py                                    — yeni; kalp hizi tahmini yardimcilari
src/web/app.py                                         — layout UI, threshold default, timing panel fix, tachy considerations UI
tests/test_digitize.py                                 — kalite kapilari ve retry mantigi testleri
tests/test_ecg_render.py                               — interpreter secimi testi
tests/test_rhythm.py                                   — yeni; HR estimation ve rate-aware diagnosis testleri
external/open-ecg-digitizer/src/config/lead_layouts_george-moody-2024.yml — layout sirasi/isim hizasi
external/open-ecg-digitizer/src/model/inference_wrapper.py                 — kaliteyi bozan downscale patch'lerinin geri alinmasi
external/open-ecg-digitizer/src/model/lead_identifier.py                  — sessiz layout fallback mantigi duzeltildi
```

## Karsilasilan Sorunlar
- **Sorun:** Hiz icin eklenen vendor downscale patch'leri bazi goruntulerde `raw_lines=0` ve tum lead'lerde sifir sinyal uretiyordu.
  **Cozum:** Crop/extraction downscale varsayilanlari geri alindi; segmentation cap korundu.
- **Sorun:** Extraction basarisiz olsa bile pipeline bunu gecerli cikti gibi kabul edip sifira yakin `.npy` kaydedebiliyordu.
  **Cozum:** `all-NaN canonical`, dusuk `raw_lines`, dusuk aktif lead sayisi icin fail-fast guard eklendi.
- **Sorun:** UI'da `Total (end-to-end)` ile alttaki `Total` iki farkli toplammis gibi gorunuyordu; gercekte cift sayim vardi.
  **Cozum:** Timing paneli yeniden duzenlendi; toplam yalnizca gercek end-to-end sureyi gosteriyor.
- **Sorun:** Ilk istek kullaniciya "digitizer cok yavas" gibi gorunuyordu, cunku model load sureleri inference ile karisiyordu.
  **Cozum:** `Load digitizer`, `Load diagnoser`, `Digitization`, `Diagnosis inference`, `Plot rendering` gibi ayri satirlar eklendi.
- **Sorun:** Bir gercek tasikardi fotografinda V2/V3/V4 orta kolon extraction'i bozuldu, kalite `POOR` ve `layout_cost=1.55` oldu.
  **Cozum:** Kotu geometri durumunda tek seferlik `apply_dewarping=True` retry ve en iyi sonucu secen skorlayici eklendi.
- **Sorun:** Gradio interaktif oturum kapaninca baglanti kopuyordu.
  **Cozum:** Uygulama `launchctl` ile arka planda calistirildi; loglar `/tmp/corio-gradio.log` altindan takip edildi.

## Teknik Notlar
- Upstream `open-ecg-digitizer` HEAD (`963387f`) ile A/B karsilastirma yapildi. Ayni `clean/162.png` orneginde mevcut patched surum `raw_lines=0` verirken upstream saglikli cikti uretti; bu kalite regresyonunun vendor patch'lerden geldigini dogruladi.
- Fix sonrasi ayni sentetik ornek exact `3x4+1R` ile `raw_lines=4`, `nonzero_leads=12`, `layout=3x4+1R`, `cost~=0.101` verdi.
- HR tabanli UI/diagnosis iyilestirmeleri iki zorlu gercek fotografla dogrulandi: bir ornekte `Estimated HR=221 bpm`, digerinde `Estimated HR=250 bpm`.
- Tachy etiketleri (ornegin `VENTRICULAR TACHYCARDIA`) ana threshold altinda kalsa bile artik ayrik tachy kutusunda gosteriliyor.
- Test ilerleyisi: once `42 passed, 1 skipped`, ritim ekleri sonrasi `45 passed, 1 skipped`, dewarping retry sonrasi `47 passed, 1 skipped`.
- `data/processed/signals/*` altindaki eski sinyaller onceki bozuk digitizer ciktilarini iceriyor olabilir; yeni kodla taze klasore yeniden uretmek gerekli.

## Siradaki
- [ ] Kalan zor gercek foto vakalari icin yalnizca gerekirse full-resolution segmentation retry ekle
- [ ] Eski `data/processed/signals/*` artefaktlarini yeni pipeline ile yeniden uret
- [ ] Tachy/wide-complex vakalar icin heuristic alarm veya ek re-ranking katmani dusun
- [ ] Vendor degisikliklerini patch/fork olarak kalici hale getir
- [ ] Gercek telefon fotograf dataset'i ile digitizer fine-tune planini baslat
