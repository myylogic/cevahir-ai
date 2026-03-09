# 🇹🇷 Cevahir AI

**End-to-end open-source Turkish LLM project with custom tokenizer, transformer architecture and cognitive AI pipeline.**

**Türk Gençlerine Armağanımdır.** · Production Ready · Full Stack AI Development


<p align="center">
  <img src="image/cevahir_ai_visual.jpeg" style="max-width:100%;">
</p>



---

## Hakkında

Cevahir AI, **Türkçe** için özel olarak tasarlanmış, uçtan uca açık kaynak bir büyük dil modeli (LLM) projesidir. Endüstri standartlarında, **SOLID** prensipleriyle oluşturulmuş ileri seviye bir mimari sunar; tokenizer eğitiminden model mimarisine, bilişsel yönetim katmanından sohbet altyapısına kadar tüm bileşenler GitHub üzerinden erişime açıktır.

Bu proje, yapay zeka alanında kendini geliştirmek isteyen **Türk gençlerine** bir fırsat ve referans sunmak amacıyla tamamen açık kaynak olarak paylaşılmıştır. Kodları inceleyebilir, deneyebilir ve üzerine yeni çalışmalar inşa edebilirsiniz.

- **Geliştirici:** Muhammed Yasin Yılmaz ([@myylogic](https://github.com/myylogic))
- **Durum:** Production Ready
- **Son güncelleme:** 09.03.2026

---

## Özellikler

- **Türkçe BPE tokenizer** — Uçtan uca eğitim, vocab/merges, encode-decode pipeline (Unicode, İ/ı kuralları, heceleme, morfoloji desteği)
- **Transformer decoder** — RoPE, RMSNorm, SwiGLU, causal mask, weight tying, KV cache, Flash Attention altyapısı
- **Model Management** — Build, eğitim bileşenleri, save/load, forward/predict, TensorBoard
- **Cognitive Management** — Strateji (direct/think/debate/tot), bellek (RAG, vector DB), critic, araç kullanımı, middleware, izleme
- **Sohbet pipeline** — ChattingManager, oturum/geçmiş, Cevahir unified API ile sohbet asistanı akışı

---

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│                    Cevahir (Unified API)                 │
│                     model/cevahir.py                     │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ TokenizerCore│ │ ModelManager  │ │CognitiveMgr  │
│ (Türkçe BPE) │ │ (V-4 NN)     │ │ (Cognitive)  │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        ▼               ▼               ▼
  vocab/merges    Neural Network   Memory/Tools
                  (RoPE, RMSNorm,  (RAG, Critic,
                   SwiGLU, …)       Tools)
```

- **Cevahir** — encode/decode, generate, process (cognitive), generate_batch, process_batch
- **TokenizerCore** — Türkçe BPE, GPU batch, OOV hece fallback
- **ModelManager** — Model yaşam döngüsü, checkpoint, TensorBoard
- **CognitiveManager** — handle(), strateji seçimi, bellek, critic, register_tool(), get_metrics()

---

## Kurulum

Projeyi GitHub’dan indirip kendi ortamınızda çalıştırmanız gerekir. Bağımlılıklar (`requirements.txt` vb.) yaklaşık **200’e yakın kütüphane** içerebilir; kurulum ve ortam yapılandırması için Python, pip/venv ve gerekirse CUDA/PyTorch konusunda bilgi sahibi olmanız önerilir. Detaylı adımlar proje yapısına ve kullandığınız sürümlere göre değişebilir; bu konuda sorumluluk indiren geliştiricidedir.

---

## Hızlı Kullanım

```python
from model.cevahir import Cevahir, CevahirConfig

config = CevahirConfig(
    device="cuda",  # veya "cpu"
    model={
        "vocab_size": 60000,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
    }
)

cevahir = Cevahir(config)

# Sohbet (cognitive katmanı ile)
output = cevahir.process("Merhaba, nasılsın?")
print(output.response)

# Metin üretimi
text = cevahir.generate("Türkiye'nin başkenti", max_new_tokens=50, temperature=0.8)
print(text)
```

---

## Terminal ile test

Eğitilmiş model ile sohbet ve üretim testleri **terminal** üzerinden `chat_pipeline.py` ile yapılabilir:

```bash
python model_management/chat_pipeline.py
```

(Bu script, Cevahir + ChattingManager pipeline'ını kullanır; checkpoint veya kaydedilmiş model gerekir.)

---

## Eğitim

### Eğitim verisi

Model eğitimi için kullanılan veri seti **yaklaşık 680 bin örnek** içerir. Eğitim verisini kendi ortamınızda kullanmak isterseniz aşağıdaki bağlantıdan indirebilirsiniz:

- **[Eğitim verisi (Google Drive)](https://drive.google.com/drive/folders/19G5uGS5YM3rf42OefjM3KsXRyn0ZEshW?usp=sharing)** — ~680k örnek (docx, txt, soru–cevap json vb.); `prepare_cache.py` ile uyumlu formata dönüştürülebilir.

### Sıfırdan eğitim akışı

Sıfırdan eğitim için adımlar **sırayla** şöyledir:

1. **Tokenizer eğitimi** — Vocab ve merges dosyalarını üretir:
   ```bash
   python tokenizer_management/train_bpe.py
   ```
   Çıktı: `vocab.json`, `merges.txt` (veya config’te tanımlı yollar).

2. **Eğitim verisi cache’i** — Ham veriyi autoregressive eğitim formatına çevirir:
   ```bash
   python tokenizer_management/prepare_cache.py
   ```
   Desteklenen veri: **docx**, **txt** (raw metin), **json** (soru–cevap). Çıktı: BOS, EOS, PAD, SEP ve input/target dizileriyle tam hazır, autoregressive formatta cache dosyası. Veriler ortalama **512 token** uzunluğunda chunk’lara bölünür; uzun kalanlar tekrar bölünür, kısa kalanlar **padding** ile doldurulur. Chunk uzunluğu veya padding davranışı değiştirilmek istenirse `tokenizer_management/prepare_cache.py` incelenebilir.

3. **Model eğitimi** — Hazır cache ile eğitim:
   ```bash
   python training_system/train.py
   ```
   Cache’teki veri otomatik yüklenir; eğitim bu format üzerinden ilerler.

Eğitim için GPU önerilir.

### Model parametrelerini değiştirme

Model boyutu ve eğitim hiperparametreleri (embed_dim, num_layers, num_heads, lr, dropout vb.) değiştirilmek istenirse **iki yerde** güncelleme yapılmalı:

- **`model/cevahir.py`** — CevahirConfig / model default değerleri (inference ve pipeline ile uyum için).
- **`training_system/train.py`** — `TRAIN_CONFIG` ve model parametreleri (eğitimde kullanılan değerler).

İkisi birbiriyle uyumlu olmalı; aksi halde eğitilen checkpoint yüklendiğinde shape veya davranış uyuşmazlığı oluşabilir.

---

## Proje Yapısı

```
cevahir_sinir_sistemi/
├── model/                 # Unified API (cevahir.py)
├── cognitive_management/  # Bilişsel katman (strateji, bellek, critic, tools)
├── model_management/      # Model yaşam döngüsü (build, save/load, forward)
├── training_system/       # Eğitim pipeline (train.py, v2)
├── tokenizer_management/  # Türkçe BPE (core, bpe, train_bpe)
├── src/                   # Sinir ağı (CevahirNeuralNetwork, V-4)
├── chatting_management/   # Sohbet (ChattingManager, oturum, context)
└── docs/                  # Dokümantasyon, hata debugging süreçleri
```

**~653 modül dosyası** (.py), **12 çatı modül** (model, cognitive_management, model_management, training_system, tokenizer_management, src, chatting_management, data_loader_management, training_management, openai-data-mining, api, data_processing) — tokenizer, sinir ağı, model yönetimi, eğitim, cognitive, sohbet tamamı modüler ve yapılandırılabilir.

---

## Neden Açık Kaynak?

Yapay zeka geleceğin teknolojisi; ancak kaynaklar çoğu zaman büyük şirketlerin elinde veya yabancı dillere odaklı. Cevahir AI ile:

- **Türkçe odaklı** bir dil modeli pipeline'ını uçtan uca inceleyebilirsiniz.
- Tokenizer eğitimi, Transformer mimarisi, model eğitimi, sohbet ve bilişsel stratejiler **tek bir projede** şeffaf biçimde yer alır.
- Proje, Türk gençlerinin yapay zekayı anlaması ve geliştirmesi için bir **eğitim kaynağı** ve **referans mimari** olarak tasarlandı.

---

## Dokümantasyon

- **Mimari:** `docs/` — sistem mimarisi, katmanlar, veri akışı
- **API referansı:** `docs/` — Cevahir, ModelManager, CognitiveManager, TokenizerCore kullanımı
- **Modül dokümantasyonu:** `model/`, `cognitive_management/`, `model_management/`, `training_system/`, `tokenizer_management/`, `src/` altındaki modüllerde docstring ve README dosyaları
- **Eğitim rehberi:** `training_system/train.py`, `tokenizer_management/train_bpe.py`, `prepare_cache.py` — parametreler ve akış için bu dosyalar incelenebilir
- **Inference / sohbet:** `model_management/chat_pipeline.py`, `model/cevahir.py` — kullanım örnekleri ve config

Proje kökünde yalnızca bu README ve kaynak kod yer alır; ek metinler (tanıtım, süreç özeti vb.) depoda ayrı tutulmaz.

---

## Lisans

Proje **Apache License 2.0** ile lisanslanmıştır; tam açık kaynaktır ve herkesin erişimine açıktır. Detaylar için repo kökündeki `LICENSE` dosyasına bakın.

---

## Katkı

Katkılar memnuniyetle karşılanır. Fork, feature branch, commit ve Pull Request adımlarını izleyebilirsiniz.

---

## İletişim

- **GitHub:** [@myylogic](https://github.com/myylogic)
- **X (Twitter):** [@myylogic](https://x.com/myylogic)
- **Instagram:** [@myylogic](https://instagram.com/myylogic)

- **Proje:** Cevahir AI — Türk Gençlerine Armağanımdır.

---

**Muhammed Yasin Yılmaz** · 09.03.2026 · Production Ready Full Stack AI Development
