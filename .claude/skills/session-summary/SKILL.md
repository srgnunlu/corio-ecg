---
name: session-summary
description: Oturum (sohbet) sonunda yapılanların özetini docs/sessions/ klasörüne kaydet. Use when the user says "oturumu özetle", "session summary", "oturumu kaydet", "epikriz yaz", or at the end of a significant work session. Also use when switching phases, finishing a feature, or before closing a conversation where meaningful work was done.
---

# Session Summary (Oturum Epikriz)

Write a clinical-style discharge summary (epikriz) for the current coding session. This creates institutional memory so future sessions can pick up exactly where this one left off.

## What To Do

1. **Scan the session context** — review what was discussed, what files were created/modified, what decisions were made
2. **Check git log** — run `git log --oneline` to see commits made during this session
3. **Check git diff --stat** against the starting commit (if known) to see all changes
4. **Write the summary** following the template below
5. **Save to** `docs/sessions/YYYY-MM-DD-<kısa-konu>.md`
6. **Update MEMORY.md** in the project memory directory with any new stable patterns
7. **Commit** the session summary

## Template

Use this exact structure. Fill in each section based on the session. Skip sections that don't apply (e.g., no issues encountered = skip that section). Write in Turkish.

```markdown
# Oturum: [Kısa Başlık]
**Tarih:** YYYY-MM-DD
**Süre:** ~X saat (tahmin)
**Faz:** [Hangi proje fazında çalışıldı]

## Özet
[2-3 cümle: Ne yapıldı, neden, sonuç ne oldu]

## Yapılan İşler
- [ ] veya [x] formatında, her iş bir satır
- Commit hash'leriyle birlikte

## Alınan Kararlar
| Karar | Seçilen | Neden |
|-------|---------|-------|
| ... | ... | ... |

## Değiştirilen/Oluşturulan Dosyalar
```
dosya/yolu.py  — kısa açıklama
dosya/yolu2.py — kısa açıklama
```

## Karşılaşılan Sorunlar
- **Sorun:** Kısa açıklama
  **Çözüm:** Ne yapıldı

## Teknik Notlar
[Gelecek oturumların bilmesi gereken teknik detaylar — model parametreleri, API davranışları, edge case'ler vs.]

## Sıradaki
- [ ] Bir sonraki oturumda yapılacak ilk iş
- [ ] Sonraki adımlar
```

## Rules

- Keep it concise — a future Claude session should be able to read this in under 30 seconds and know exactly where to pick up
- Include specific file paths, commit hashes, and version numbers — vague summaries are useless
- If a decision was made, record WHY — the reasoning matters more than the choice
- Technical notes should include things that surprised you or that differ from documentation
- The "Sıradaki" section is the most important — it's the handoff to the next session
