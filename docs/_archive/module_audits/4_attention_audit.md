# Modül İncelemesi: Attention (MultiHeadAttention)

**Yol haritası madde 3.**  
**Dosya:** `src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py`

---

## 1. Akış (Forward)

- Girdi: query, key, value (B, L, E); key/value None ise self-attention (query kullanılır).
- Projeksiyon: Q, K, V lineer projeksiyonları → (B, L, H, D) head’lere ayrılır.
- RoPE (opsiyonel): positional_encoding.apply_rotary_pos_emb ile Q, K’ya uygulanır.
- **Scaled dot-product:** `scores = Q @ K^T / scale`, `scale = sqrt(d_k) * temperature`.
- Mask: Causal (triu, -inf) ve/veya padding mask additive olarak scores’a eklenir.
- Softmax: `scores - max(scores)` (sayısal kararlılık), sonra `nan_to_num`, sonra softmax.
- Çıkış: `attn_weights @ V`, ardından concat ve out_proj.

---

## 2. Scale (sqrt(d_k))

| Yol | Kod | Durum |
|-----|-----|--------|
| Standard SDPA | `scale = math.sqrt(d_k) * temperature`, `scores = matmul(...) / max(scale, 1e-6)` | Doğru: skorlar 1/sqrt(d_k) ile ölçekleniyor. |
| Flash Attention | `softmax_scale = 1.0 / math.sqrt(D) * temperature` | Doğru. |

Endüstri standardı (Vaswani et al.): skorların `1/sqrt(d_k)` ile ölçeklenmesi; kod buna uygun.

---

## 3. Mask

- **Causal:** `torch.triu(..., diagonal=1)` ile üst üçgen True; float’a çevrilince engellenen yerlere -inf.
- **Padding:** `_prepare_attention_mask` ile (L,S) veya (B,L,S) additive float mask (-inf = engelle).
- Mask, softmax öncesi scores’a ekleniyor; Flash path’te boolean mask’e çevriliyor.

---

## 4. Sayısal kararlılık

- Softmax öncesi: `scores = scores - max(scores, dim=-1, keepdim=True)`.
- `nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)`.
- Boş sequence (seq_len=0) için sıfır çıkış döndürülüyor.

---

## 5. Çıkış projeksiyonu

- `out_proj`: `nn.Linear(embed_dim, embed_dim)`, init `gain=sqrt(1/num_heads)` (çok başlık birleşimini dengelemek için).

---

## 6. Tespitler

| Konu | Durum |
|------|--------|
| Scale 1/sqrt(d_k) | Uygulanıyor (standard + Flash). |
| Causal / padding mask | Uygulanıyor, additive -inf. |
| NaN/Inf | Softmax öncesi max shift + nan_to_num; çıkışta NaN/Inf beklenmez. |
| Attention ağırlıkları | Softmax sonrası; satır bazında toplam 1. Test: A1 (weights sum ≈ 1, no NaN). |

---

## 7. Testler

- **A1:** Tam model forward’dan dönen `attn_weights` (None değilse) NaN içermemeli ve her (b,h,i) için `sum(attn_weights[b,h,i,:]) ≈ 1.0` olmalı.

Bu inceleme, yol haritası 3. maddesinin tamamlanmış halidir.
