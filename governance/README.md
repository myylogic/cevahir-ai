# governance/

Tracked data of the roof package (`cevahir_ai`). No Python code lives here.
Explanation (Turkish, with an English summary): `docs/architecture/CATI_VE_ANAHTAR_SISTEMI.md`.

| File | What it is | Changed by |
|---|---|---|
| `freeze/cev11-closure.json` | The product modules the frozen CEV-11 study depends on (33 modules), each pinned by its utf8-lf digest, plus the history of re-pins with their reasons. | `python -m cevahir_ai.evidence.closure record --reason "…"` after the change was reviewed and the study's verification re-run. Never by hand. |
| `registry/tokenizers.json` | Tracked tokenizers: canonical key (`tok:bpe-sem-v1`), file keys of `vocab.json` and `merges.txt`, and the legacy `tokenizer_digest` v3 values that name the same tokenizer. | `cevahir_ai.adapters.tokenizers.tracked_entry(...)` after a reviewed tokenizer change. Aliases only grow. |

Checks:

```
python -m cevahir_ai.evidence.closure check     # exit 5 names the CEV-11 dependency that changed
python -m cevahir_ai.keys.schemas --check       # golden weight layouts still match the constructor
```

A drift reported by the closure check is not an error by itself. It means: a module that CEV-11
imports changed, so re-run `tests/evolution/test_cev11_*.py` and the study's `verify.py --check`,
then re-pin with the reason.

Files here are LF on every platform (`.gitattributes`), so the digests in them are stable.
