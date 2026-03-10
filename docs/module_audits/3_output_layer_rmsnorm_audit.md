# Modül İncelemesi: Çıkış Katmanı + Output RMSNorm

**Tarih:** Mimari inceleme (150 epoch sonrası)  
**Dosyalar:** `src/neural_network.py` (output_norm, output_layer), `src/neural_network_module/ortak_katman_module/rms_norm.py`

---

## 1. Akış (Forward)

```
x [B,T,embed_dim] (layer çıkışı)
  → output_norm(x)   [RMSNorm: x * rsqrt(mean(x^2)+eps) * scale]
  → x_normalized [B,T,embed_dim]
  → output_layer(x_normalized)  [Linear, weight = embedding.weight]
  → logits [B,T,vocab_size]
```

- **Weight tying:** `output_layer.weight = self.embedding.embedding.weight` (aynı referans).
- **Softmax** model içinde yok; loss/inference’da uygulanıyor.

---

## 2. Tespit Edilen Sorunlar

### 2.1 Output RMSNorm çıkışının aşırı büyümesi

- **Teşhis script sonucu (checkpoint yüklü):** `output_norm_out` min=-14.85, max=28.09.
- **RMSNorm formülü:** `out = x * rsqrt(mean(x^2)+eps) * scale`. `x*rsqrt(...)` kısmı son boyutta RMS≈1 yapar; çıkışın büyüklüğü büyük ölçüde **scale** parametresine bağlıdır.
- **Yorum:** Eğitim sırasında `output_norm.scale` çok büyümüş (≈10–30 mertebesi). Bu da logit’leri büyütüyor: `logits = x_normalized @ W^T`; `x_normalized` büyük olunca logit’ler de büyüyor, softmax aşırı sivri (mode collapse) ve entropy nan riski artıyor.

### 2.2 Logit ölçeği üzerinde sınır yok

- Weight tying ile `W` (embedding) genelde küçük (Xavier vb.). Ama **girdi** (output_norm çıkışı) büyükse logit’ler yine büyük olur.
- LLaMA/GPT tarzı kullanımda son norm çıkışı genelde sınırlı kalır; bizde öğrenilen `scale` bunu bozuyor.

### 2.3 RMSNorm implementasyonu

- `rms_norm.py`: Formül doğru (`mean(x^2)`, `rsqrt`, `* scale`). `eps=1e-6` sayısal açıdan makul.
- Sorun implementasyonda değil, **output_norm’un çıkışının** (özellikle scale’in büyümesiyle) logit’lere taşınması.

---

## 3. Önerilen Düzeltmeler

### 3.1 (Uygulandı) Logit ölçek faktörü

- **Ne:** `output_layer(x_normalized)` çıkışını `1/sqrt(embed_dim)` ile çarpmak.
- **Neden:** Logit’leri sınırlı tutar; softmax daha az sivri olur, mode collapse ve entropy nan riski azalır.
- **Yer:** `neural_network.py` forward, `final_output = self.output_layer(x_normalized)` satırı.

### 3.2 (Opsiyonel) output_norm scale kısıtlaması

- Örneğin `scale`’i clamp (max 2–3) veya küçük init (0.1) + büyüme sınırı. İlk aşamada logit scale faktörü yeterli olabilir; gerekirse sonra eklenir.

---

## 4. Sonuç

| Madde | Durum |
|-------|--------|
| RMSNorm formülü | Doğru |
| Weight tying | Doğru kullanılmış |
| output_norm çıkışı (eğitimli model) | Aşırı büyük (scale büyümesi) |
| Logit üzerinde ölçek kontrolü | Yoktu → **1/sqrt(dim) faktörü eklendi** |

Bu inceleme, `MIMARI_INCELEME_YOL_HARITASI.md` içindeki “Çıkış katmanı + Output norm” maddesinin ilk adımıdır.
