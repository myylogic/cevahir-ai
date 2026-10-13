# Living learning — Tur 15

Yerel, eşleştirilmiş deneyler: hazır geçmiş vektörü yerine gerçek bit akışından bellek kullanımı. 24 yaşam × 3 görev × 7 yöntem. Yöntemler önceden verilmiş temsil ailelerini kullanır; yeni genel öğrenme ilkesi veya AGI iddiası yoktur.

- [Türkçe araştırma raporu](../../docs/research/living_learning_temporal_2026_09_22/REPORT_TR.md)
- [Deneyden önce yazılmış protokol](../../docs/research/living_learning_temporal_2026_09_22/PROTOCOL.md)
- [Matematik ve birincil literatür](../../docs/research/living_learning_temporal_2026_09_22/MATHEMATICAL_AND_LITERATURE_TR.md)
- [Bugünkü cevap ve gelecekte öğrenme durumu](../../docs/research/living_learning_temporal_2026_09_22/LEARNING_STATE_TR.md)
- [Ham sonuç](results/temporal_learning.json), [özet](results/summary.json), [doğrulama](results/verification.json), [son denetim](results/final_audit.json)

Python ve NumPy gerektirir. Depo kökünde:

```powershell
python research/living_learning_temporal_2026_09_22/temporal_learning.py
python research/living_learning_temporal_2026_09_22/analyze_results.py
python research/living_learning_temporal_2026_09_22/verify_temporal.py --rerun
python research/living_learning_temporal_2026_09_22/verify_learning_state.py
python research/living_learning_temporal_2026_09_22/final_audit.py
```

İlk komut kayıtlı sonucu tam olarak yeniden hesaplayıp karşılaştırır. İlk üretimde `--write` kullanılmıştır; normal tekrar için gerekmez. Sonuçta sıfır görünen çok küçük olasılık hataları makine hassasiyetine tabidir. Öğrenici ham tamamlanmış bölümleri saklamaz; deney sürücüsü yeniden üretilebilirlik için girdileri kaydeder.

Önceki 961 dosya ve HEAD başlangıç özetiyle karşılaştırılır. Commit/push yapılmaz. Ana model ve kitap değiştirilmez.
