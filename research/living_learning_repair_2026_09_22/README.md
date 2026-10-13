# Living learning — Tur 17

Deneyimden çıkarılan ağaç düzeltme şemalarının sonraki edinime etkisi: 24 yaşam, üç rejim, beş yöntem, 2.880 hedef denemesi. Verilen F17 ifade dili ve yorumlayıcı içinde kontrollü transfer deneyi; yeni temel öğrenme ilkesi iddiası yoktur. Güçlü toplu düzenleme kontrolü şemalı yöntemle aynı kabul programlarını üretti.

- [Rapor](../../docs/research/living_learning_repair_2026_09_22/REPORT_TR.md)
- [Sonuç öncesi protokol](../../docs/research/living_learning_repair_2026_09_22/PROTOCOL.md)
- [Matematik ve literatür](../../docs/research/living_learning_repair_2026_09_22/MATHEMATICS_AND_LITERATURE_TR.md)
- [Ham kayıt](results/repair_learning.json), [özet](results/summary.json), [doğrulama](results/verification.json), [son denetim](results/final_audit.json)

Python ve NumPy gerekir. Depo kökünde:

```powershell
python research/living_learning_repair_2026_09_22/repair_learning.py
python research/living_learning_repair_2026_09_22/analyze_results.py
python research/living_learning_repair_2026_09_22/verify_repair.py --rerun
python research/living_learning_repair_2026_09_22/final_audit.py
```

İlk komut kayıtlı sonucu yeniden hesaplayıp karşılaştırır; ilk üretimde `--write` kullanılmıştır. Kaynak ağaçlar ve gizli doğrulama etiketleri deney sürücüsünde tutulur; öğrenici yalnız görev örneklerini alır. Soyut iş maliyeti gerçek CPU/RAM sayısı değildir. Önceki 994 dosya ve HEAD korunur; commit/push yok.
