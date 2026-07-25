# Causal Mask Kök Neden ve Düzeltme

**Tarih:** Mimari inceleme – causal invariance testi sonrası  
**İlgili test:** `tests/architecture_standards/test_neural_network_forward_comprehensive.py::TestCausalMask::test_causal_logits_at_position_0_unchanged_when_future_tokens_change`

---

## 1. Belirti

Pozisyon 0’daki logitler, aynı pozisyon 0 token’ı korunup yalnızca kuyruk (pozisyon 1, 2, …) değiştirildiğinde **değişiyordu**. Causal (autoregressive) modelde pozisyon 0 çıktısı yalnızca pozisyon 0 girdisine bağlı olmalıydı.

---

## 2. Kök Neden

**Dosya:** `src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py`

`_forward_impl` içinde self-attention şu şekilde çağrılıyordu:

```python
attn_result = self.attn(
    x_norm, x_norm, x_norm,
    mask=mask,
    return_attention_weights=True,
    use_cache=use_cache,
    cache_position=cache_position,
)
```

`MultiHeadAttention.forward` imzasında `causal_mask` parametresi var ve **varsayılan değeri `False`**. Çağrıda `causal_mask` hiç geçirilmediği için attention her zaman **causal olmadan** (tüm pozisyonlar birbirini gördü) çalışıyordu. Ana model `causal_mask=True` ile layer’ı çağırsa bile bu değer attention’a iletilmiyordu.

---

## 3. Düzeltme

Aynı dosyada **iki yerde** (pre-norm ve post-norm dallarında) `self.attn(...)` çağrısına şu argüman eklendi:

```python
causal_mask=causal_mask,
```

Böylece layer’ın aldığı `causal_mask` değeri (ana modelden gelen `use_causal`) doğrudan MultiHeadAttention’a geçirildi.

---

## 4. Doğrulama

- Causal invariance testi artık **geçiyor** (xfail kaldırıldı).
- Tüm `tests/architecture_standards/test_neural_network_forward_comprehensive.py` testleri çalıştırılarak regresyon yokluğu doğrulandı.

---

## 5. Etki

- **Eğitim:** Causal mask olmadan model “geleceği görüyordu”; teacher forcing ile eğitimde bu, hedef token’ların gizlenmemesine ve autoregressive davranışın bozulmasına yol açabilirdi.
- **Inference/üretim:** Decoder tarafında causal mask zaten kritik; mask uygulanmıyorsa pozisyonlar birbirine “sızar”, tutarsız ve tekrarlayan çıktılar (mode collapse) riski artar.

Bu düzeltme, 150 epoch sonrası gözlenen mode collapse ve anlamsız üretimle uyumlu bir **mimari hata**yı giderir; eğitimin yeniden çalıştırılması önerilir.
