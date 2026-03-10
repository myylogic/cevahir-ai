# Modül İncelemesi: Positional Encoding (RoPE)

**Yol haritası madde 4.**  
**Dosya:** `src/neural_network_module/dil_katmani_module/positional_encoding.py`

---

## 1. Modlar

- **sinusoidal:** Klasik Transformer PE (buffer), forward’da `x + pe_slice`.
- **learned:** `nn.Embedding(max_len, embed_dim)`, forward’da `x + pe_embed(pos_idx)`.
- **rope:** Ana akışa PE eklenmez; `forward(x)` pass-through (`out = x`) + dropout. RoPE sadece attention içinde `apply_rotary_pos_emb` ile Q/K üzerinde uygulanır.

---

## 2. RoPE Frekansları

- **Formül:** `inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))`, `base=10000`, sonra `freqs = positions * inv_freq` → `[max_len, dim//2]`.
- **Boyut:** `num_heads` verildiğinde `rope_dim = embed_dim // num_heads` (head_dim); verilmezse `rope_dim = embed_dim` (uyumsuzluk uyarısı).
- **Genişletme:** `_grow_to(new_len)` ile `max_len` aşıldığında `rope_freqs` yeniden hesaplanıp büyütülüyor.

---

## 3. apply_rotary_pos_emb

- **Girdi:** `x` [B, T, D] veya [B, H, T, D]; `positions` [T] veya [B,T] (2D ise `positions[0]` kullanılır).
- **Index:** `rope_freqs[positions]` → [T, D//2]. `max(positions)+1 > max_len` ise `_grow_to(max_pos)` çağrılıyor.
- **Döndürme:** Çiftler (x1,x2) üzerinde [cos -sin; sin cos] rotasyonu; norm korunur (ortogonal).
- **Sayısal:** cos/sin sınırlı; girdi sonlu ise çıkışta NaN/Inf beklenmez.

---

## 4. Ana Model Entegrasyonu

- `neural_network.py`: `pe_mode="rope"` ile `PositionalEncoding(..., num_heads=num_heads)`; `use_rope=True` ile attention’a `positional_encoding_ref` veriliyor.
- Attention’da: Q/K projeksiyonundan sonra `apply_rotary_pos_emb(query_proj)` ve `apply_rotary_pos_emb(key_proj)` çağrılıyor; value’ya uygulanmıyor (standart).

---

## 5. Tespitler

| Konu | Durum |
|------|--------|
| RoPE formülü (base=10000, inv_freq) | Standart (LLaMA/GPT stili). |
| head_dim uyumu | `num_heads` verildiğinde rope_dim = head_dim. |
| Uzunluk / ölçek | `_grow_to` ile dinamik uzatma; bağlam kaybı yok. |
| NaN/Inf | Rotation sınırlı; test P1 ile doğrulanır. |
| Forward (rope modu) | Pass-through + dropout; RoPE sadece attention’da. |

---

## 6. Testler

- **P1:** RoPE modunda `apply_rotary_pos_emb(x)` çıkışında **NaN/Inf olmamalı** ve girdi ile aynı şekil korunmalı.

Bu inceleme, yol haritası 4. maddesinin tamamlanmış halidir.
