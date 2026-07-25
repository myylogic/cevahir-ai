# Modül İncelemesi: Embedding (LanguageEmbedding)

**Yol haritası madde 2.**  
**Dosya:** `src/neural_network_module/dil_katmani_module/language_embedding.py`

---

## 1. Akış (Forward)

```
x [B, T] (token id)
  → embedding(x)           [B, T, embed_dim]
  → [opsiyonel] * sqrt(embed_dim)   (scale_by_sqrt=True ise; ana modelde False)
  → LayerNorm(out)         (varsayılan: her zaman açık)
  → Dropout(out)
  → çıkış [B, T, embed_dim]
```

---

## 2. Ana modelde kullanım (neural_network.py)

- `init_method="xavier_normal"`
- `scale_by_sqrt=False` (gradient patlaması önleme)
- `norm_type` verilmiyor → varsayılan **LayerNorm** uygulanıyor
- Weight tying: `output_layer.weight = self.embedding.embedding.weight`

---

## 3. Ağırlık başlatma (_initialize_weights)

- Xavier normal sonrası **satır bazlı birim norm**: `w = w / ||w||`, ardından `* 0.15`.
- Böylece her token vektörü başlangıçta benzer ölçekte; tek token baskınlaşmıyor.
- `padding_idx` satırı sıfırda tutuluyor.

---

## 4. Tespitler

| Konu | Durum |
|------|--------|
| NaN/Inf | Forward’da LayerNorm + clamp’lı init ile çıkışta NaN/Inf beklenmez. Test: O2 (no_nan_inf, norm_bounded). |
| scale_by_sqrt | Ana modelde **False**; büyük vocab’ta gradient patlaması riski azaltılmış. |
| LayerNorm | Varsayılan açık; çıkış ölçeği stabilize. |
| Çıkış normu | LayerNorm sonrası token vektör normu makul aralıkta. Test: O2 (norm ≤ EMBED_NORM_MAX). |
| Gradient 163k | Epoch 150 log’unda embedding grad çok yüksek (163k). Sebep: weight tying + büyük vocab; grad hem embedding hem output_layer’dan geliyor. Önlem: `embedding_lr_scale=0.001` (train.py) ile embedding LR düşürülmüş. |

---

## 5. Testler (ARCHITECTURE_TEST_STANDARDS)

- **O2:** `TestEmbeddingStandards.test_o2_no_nan_inf`, `test_o2_norm_bounded` (çıkışta NaN/Inf yok, vektör normu ≤ 50).
- **E1 (ek):** Embedding grad norm’un diğer parametrelere göre aşırı baskın olmaması (gradient dominance) — ayrı test ile doğrulanır.

---

## 6. Sonuç

- **Sayısal kararlılık:** Init + LayerNorm uygun; mevcut O2 testleri geçiyor.
- **Ölçek:** scale_by_sqrt=False ve LayerNorm ile çıkış sınırlı.
- **Gradient:** 163k sorunu eğitim tarafında `embedding_lr_scale` ile yumuşatılmış; mimari tarafında ek zorunlu değişiklik yok. İsteğe bağlı: embedding grad norm’un diğer parametrelere oranı için üst sınır testi (E1).

Bu inceleme, yol haritası 2. maddesinin tamamlanmış halidir.
