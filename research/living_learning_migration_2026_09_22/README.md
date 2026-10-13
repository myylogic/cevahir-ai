# Living learning — Tur 16

Doğru öğrenilmiş bir programı devam eden bit akışına yerleştirmek: kural edinimi, mevcut durum bilgisi ve eski yetkinliği koruma ayrı ölçülür. 24 yaşam × iki geçmiş uzunluğu × iki yeni görev × altı taşıma yöntemi. Kalibrasyon kanalı ve aday diller verilidir; yeni genel öğrenme ilkesi veya AGI iddiası yoktur.

- [Rapor](../../docs/research/living_learning_migration_2026_09_22/REPORT_TR.md)
- [Sonuç öncesi protokol](../../docs/research/living_learning_migration_2026_09_22/PROTOCOL.md)
- [Matematik ve kaynaklar](../../docs/research/living_learning_migration_2026_09_22/MATHEMATICAL_AND_LITERATURE_TR.md)
- [Maliyet sayacı düzeltmesi](../../docs/research/living_learning_migration_2026_09_22/CORRECTIONS.md)
- [Ana problem ve sonraki ayırıcı sınama](../../docs/research/living_learning_migration_2026_09_22/WHOLE_PROBLEM_AND_NEXT_TEST_TR.md)
- [Ham sonuç](results/migration_learning.json), [özet](results/summary.json), [doğrulama](results/verification.json), [son denetim](results/final_audit.json)

Depo kökünde, Python standart kütüphanesiyle:

```powershell
python research/living_learning_migration_2026_09_22/migration_learning.py
python research/living_learning_migration_2026_09_22/analyze_results.py
python research/living_learning_migration_2026_09_22/verify_migration.py --rerun
python research/living_learning_migration_2026_09_22/final_audit.py
```

İlk komut kayıtlı sonucu yeniden hesaplayıp tam eşitliği denetler. `--write` ana sonuç üretimine ayrılmıştır. İlk maliyet eksiğini içeren kayıtlar yalnız tarihsel denetim için korunur. Önceki 977 dosya ve HEAD aynı kalır. Commit veya push yapılmaz.
