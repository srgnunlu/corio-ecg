<!-- Purpose: Session record — Fable 5.1 project review, repo hygiene, Phase B steps 6-7, overnight sweep chains. -->

# Oturum özeti — 2026-09-04: proje incelemesi → hijyen → Aşama B adım 6-7

**Başlangıç durumu:** branch `feature/phaseb-consistency-training` main'in 2
commit önünde (`2a10d88`), 14 Ağustos'tan beri dokunulmamış, push yok.
Adım 6 (PTB-XL etiketsiz ölçekleme) yarım: NaN-etiket değişikliği uncommitted,
korpus scripti untracked, korpus 12 kayıtlık smoke test. Suite 499 passed.

**Bitiş durumu (bu kayıt yazılırken):** main = origin/main = `d8b5d76`, 8 yeni
commit. Suite 504 passed / 1 skipped. İki gece zinciri arka planda koşuyor.

## Yapılanlar

### Proje incelemesi (Fable 5.1 taze göz)
Puanlama: araştırma metodolojisi 9/10, kanıt gücü 6/10, kod/test 7/10,
tekrarlanabilirlik 4/10, ürün olgunluğu 4/10, hijyen 5/10 → genel 6.5/10.
Ana bulgular: adım 6 yarım ve tasarım riski taşıyor (etiketsiz satırlar
denetimi boğar), push/merge yok, OMI ürün yolunda değil, `run.py` ölü,
Python sürümü dört yerde tutarsız, lockfile yok, iki dev dosya limitin 3 katı.

### Hijyen (`1fd5fc2`, `5ca8a4d`, `8106caf`)
- Python 3.12 her yerde (pyproject, ruff, mypy, CLAUDE.md, setup script);
  `requirements.lock` eklendi; `memory_test.py`, `AGENTS.md`, `.agents/`,
  `results-vps-500/` silindi; March plan dokümanı `docs/plans/` altına.
- `src/pipeline/run.py` (yetim CLI) ve `override_lead_assignment()` (hiç
  çağrılmayan fonksiyon) kaldırıldı; CLAUDE.md/README'deki eski komut düzeltildi.
- `evaluation.py`'de `callable` tip hatası → `Callable`.

### Aşama B adım 6 — etiketsiz ölçekleme (`1d987f6`, `8f7b5da`)
Tutarlılık terimi etiket gerektirmediği için PTB-XL çiftleri karıştırılabilir.
Kritik tasarım kararı: etiketsiz satırlar **ayrı loader'dan sabit sayıda**
(varsayılan 16) her etiketli batch'e ekleniyor. Tek loader'da havuzlansaydı
21.8K etiketsiz satır 460 etiketliyi boğar, batch'lerin çoğunda BCE terimi
sıfırlanırdı. Teacher hedefi artık satır bazlı (teacher varsa o, yoksa
detached clean logit). 5 yeni test. Sweep koşturucu + eşleştirilmiş özetleyici
yazıldı; özetleyici v1'in hücre farklarını birebir üretiyor.

⚠️ Smoke bulgusu: PTB-XL temiz sinyalde v2a teacher logitlerinin yalnız %8'i
pozitif (medyan −3.8). Etiketsiz çiftler ağırlıkla negatif bölgede invariance
öğretecek; pozitif bölgedeki sıkışmayı açması beklenmemeli.

### Aşama B adım 7 — moderate zorluk taraması (`74f529a`, `d8b5d76`)
Doküman: `docs/experiments/phaseb-difficulty-sweep-v1.md`.

| Kol | clean AUROC | moderate AUROC | clean ayrım | moderate ayrım |
|---|---:|---:|---:|---:|
| başlangıç | 0.7590 | 0.7537 | 0.387 | 0.367 |
| w=0 | 0.8033 | 0.7773 | 0.413 | **0.286** |
| w=1 | **0.8272** | **0.7942** | 0.575 | **0.484** |

Eşleştirilmiş w=1 − w=0 moderate'ta: ΔAUROC **+0.0169 (6/6, p=0.012)**,
Δayrım **+0.197 (6/6, p<0.001)**. Kazanç render artefaktı değil. Yeni bulgu:
düz digitize eğitimi ayrımı **daraltıyor** (AUROC'u iyileştirirken), tutarlılık
terimi geri açıyor. Sürpriz: moderate render digitizer'ı neredeyse hiç
zorlamıyor (başlangıç −0.005 vs clean) → gerçek fotoğraf için `hard` asıl sınav.

Kusur ve düzeltme: moderate korpus iki fold'u birlikte örnekleyerek kurulduğu
için negatifler clean'den farklı (484/920 ortak). Derleyiciye `--records-from`
eklendi; eşleşmiş moderate + hard gece zincirinde.

## Devam eden (arka plan, nohup)

- **Zincir 1:** PTB-XL 2000 kayıt (`ptbxl-corpus/clean`, ~4 sa) → etiketsiz
  sweep 2 fold × 3 seed, w=1, +16/batch → `results/omi/sweeps/ptbxl_unlabelled/`
- **Zincir 2 (zincir 1 bitince):** moderate fill (436 kayıt) → hard korpus
  (920) → `manifest_matched.csv` → `results/omi/sweeps/moderate_matched/` +
  `results/omi/sweeps/hard/`
- Loglar scratchpad'de (`phaseb_chain.log`, `phaseb_chain2.log`).
  Özet: `python scripts/summarise_consistency_sweep.py <dir>`.

⚠️ Korpus derlerken eşzamanlı pytest + eğitim koşmak makineyi swap'a soktu
(8 → 100 s/kayıt). Derleme sırasında başka ağır iş koşma.

## Sıradaki adımlar

1. Zincir sonuçlarını `phaseb-unlabelled-scale-v1.md` + difficulty doc
   güncellemesi olarak yaz.
2. Kalan açık digitizer tarafındaysa (moderate'ta %26 toparlanma) sonraki
   müdahale head'de değil, digitize sinyalde veya image-branch'te.
3. Aşama B kapanınca OMI'yi web app'e kapalı bayrakla bağla (öneri 4, bilinçli
   ertelendi).
4. `digitize.py` (1320) ve `app.py` (1100) yalnız dokunulduğunda bölünsün.

## Üretilen kalıcı dosyalar

- `scripts/run_consistency_sweep.sh`, `scripts/summarise_consistency_sweep.py`,
  `scripts/build_ptbxl_consistency_corpus.py`, `requirements.lock`
- `results/omi/sweeps/{clean,moderate}/` (24 JSON)
- `docs/experiments/phaseb-difficulty-sweep-v1.md`
- Testler: 504 passed / 1 skipped (oturum başında 499)
