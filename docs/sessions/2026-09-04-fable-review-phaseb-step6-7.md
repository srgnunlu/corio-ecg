<!-- Purpose: Session record — Fable 5.1 project review, repo hygiene, Phase B steps 6-7 across three render regimes, and the digitiser memory investigation. -->

# Oturum özeti — 2026-09-04/05: inceleme → hijyen → Aşama B adım 6-7 (üç rejim)

**Başlangıç durumu:** branch `feature/phaseb-consistency-training` main'in 2
commit önünde (`2a10d88`), 14 Ağustos'tan beri dokunulmamış, push yok.
Adım 6 (PTB-XL etiketsiz ölçekleme) yarım: NaN-etiket değişikliği uncommitted,
korpus scripti untracked, korpus 12 kayıtlık smoke test. Suite 499 passed.

**Bitiş durumu:** main = origin/main = `5b17457`, 16 yeni commit, hepsi
pushlandı. Suite 504 passed / 1 skipped. Aşama B adım 6 ve 7 kapandı.

## Yapılanlar

### Proje incelemesi (Fable 5.1 taze göz)
Puanlama: araştırma metodolojisi 9/10, kanıt gücü 6/10, kod/test 7/10,
tekrarlanabilirlik 4/10, ürün olgunluğu 4/10, hijyen 5/10 → genel 6.5/10.
Ana bulgular: adım 6 yarım ve tasarım riski taşıyor, push/merge yok, OMI ürün
yolunda değil, `run.py` ölü, Python sürümü dört yerde tutarsız, lockfile yok.

### Hijyen (`1fd5fc2`, `5ca8a4d`, `8106caf`)
Python 3.12 her yerde; `requirements.lock`; `memory_test.py`, `AGENTS.md`,
`.agents/`, `results-vps-500/` silindi; yetim `run.py` ve `override_lead_assignment()`
kaldırıldı; `evaluation.py` `Callable` tip düzeltmesi.

### Aşama B adım 6 — etiketsiz ölçekleme (`1d987f6`, `8f7b5da`, `3439f11`, `b723d4a`)
Doküman: `docs/experiments/phaseb-unlabelled-scale-v1.md`.
Etiketsiz satırlar **ayrı loader'dan sabit sayıda** her batch'e ekleniyor (tek
loader'da 21.8K satır 460 etiketliyi boğardı); teacher hedefi satır bazlı; 5 test.
Sweep koşturucu ve eşleştirilmiş özetleyici (`--reference-dir` ile çapraz dizin).

| Kol (w=1, 6 hücre) | AUROC | AUPRC | Ayrım |
|---|---:|---:|---:|
| yalnız etiketli | 0.8272 | 0.8131 | 0.575 |
| + 2000 PTB-XL çifti | **0.8382** | **0.8231** | 0.605 |

Eşleştirilmiş ΔAUROC **+0.0110 (6/6, p=0.049)**; kazanç en zayıf hücrede
birikiyor, kol içi sapma yarıya iniyor. PTB-XL teacher hedeflerinin %86'sı
negatif → ayrım kıpırdamıyor. 24 epoch tekrarı etkisiz (bütçe darboğaz değil).

### Aşama B adım 7 — zorluk taraması, üç rejim (`74f529a`, `d8b5d76`, `f1d2d18`, `5b17457`)
Doküman: `docs/experiments/phaseb-difficulty-sweep-v1.md`. Aynı 920 kayıt,
aynı fold'lar, aynı seed'ler, `--records-from` ile eşleşmiş korpuslar.

| Kol | clean | moderate | hard |
|---|---:|---:|---:|
| başlangıç | 0.7590 | 0.7565 | **0.7268** |
| w=0 | 0.8033 | 0.7993 | 0.7853 |
| w=1 | **0.8272** | **0.8251** | **0.8097** |
| Δ(w=1 − w=0) | +0.024 (5/6) | +0.026 (6/6) | +0.024 (6/6) |

18/18 hücre lehte, etki büyüklüğü render'dan bağımsız. Hard ilk kez
digitizer'ı zorlayan rejim (düzen hatası 75/920, Einthoven p10 0.56); tutarlılık
kolu hard cezasının yarısını geri alıyor. Ayrımı yalnız tutarlılık terimi açıyor
(+0.16…0.21), w=0 üç rejimde de yerinde bırakıyor. Eşleşmemiş ilk moderate
denemesindeki "w=0 daraltıyor" bulgusu geri çekildi.

### Digitizer bellek soruşturması (`9752be3`, `56349c2`, `d4cd0b7`, `062dad7`)
Hard derlemesi traceback'siz ölüyordu. Bulgular:
- Tek kayıt digitizasyonu her rejimde **~17.5 GB bellek izi** (RSS 8.6, gerisi
  Metal); 24 GB makinede sınırda. Eşzamanlı ağır iş → swap → ölüm.
- Hard render'da **dewarping retry** yolu ~%12 kayıtta **100 GB** izi ile
  patlıyor (altı kayıt yalıtılmış ölçümle doğrulandı; retry kapalı 8 s'de bitiyor).
  Bir seferinde swap 62 GB'ın 61 GB'ına çıktı.
- Derleyici: manifest 5 kayıtta flush, `--stop-after` (süreç yeniden başlatma),
  `--render-only` (0.3 GB render geçişi), `torch.mps.empty_cache()` her kayıtta,
  zehirli kayıt mekanizması (`in_flight.txt` → `poisoned.txt`), `--no-dewarping-retry`.
- Hard korpus retry kapalı, bekçili süreç yeniden başlatmalarıyla 920/920,
  94 dakika, sıfır ölüm.
- ⚠️ Aynı dewarp yolu web app'te gerçek fotoğrafla tetiklenebilir → ayrı iş
  çipi bırakıldı ("Guard the digitiser's dewarping retry against runaway memory").

## Metodolojik dersler
1. Rejimler arası karşılaştırma **aynı kayıtlarda** yapılmalı; negatif örneklemi
   hem seviyeyi hem "w=0 ne yapar" cevabını değiştirebiliyor.
2. Tek koşuya/tek hücreye bakma; belirsizlik seed'den değil veriden geliyor.
3. Korpus derlerken makinede başka ağır iş koşma; swap'ı izle.
4. Bekçi, ölümden hemen sonra taban alırsa yanlış pozitif verir (08014);
   yalıtılmış yeniden doğrulama şart.

## Sıradaki adımlar
1. Kalan açık (hard w=1 0.81 vs temiz ~0.91) digitizer tarafında: düzen tanıma,
   dewarp yolunun onarımı, ya da head'e görüntü kalitesi girdisi.
2. Etiketsiz ölçekleme için: batch-boyutu kontrolü, PTB-XL MI-zenginleştirilmiş
   etiketsiz küme, sonra 500/1000/5000 eğrisi.
3. Aşama B kapanınca OMI'yi web app'e kapalı bayrakla bağla (bilinçli ertelendi).
4. `digitize.py` (1320) ve `app.py` (1100) yalnız dokunulduğunda bölünsün.

## Üretilen kalıcı dosyalar
- `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py`,
  `scripts/build_ptbxl_consistency_corpus.py`, `requirements.lock`
- `results/omi/sweeps/{clean,clean_e24,moderate,moderate_matched,hard,ptbxl_unlabelled,ptbxl_unlabelled_e24}/`
- `docs/experiments/phaseb-difficulty-sweep-v1.md`, `phaseb-unlabelled-scale-v1.md`
- Korpuslar (gitignored): `omi-corpus/{moderate,hard}` eşleşmiş 920, `ptbxl-corpus/clean` 2000
- Testler: 504 passed / 1 skipped (oturum başında 499)
