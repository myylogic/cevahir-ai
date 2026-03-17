# 🇹🇷 Cevahir AI & Engine

**Full-stack open-source AI engine.**

Tokenizer eğitiminden bilişsel katmana kadar uzanan **uçtan uca (end-to-end)** dil modeli altyapısını tek bir repo içinde sunar. Türkçe LLM projesi olarak başlamış; dil bağımsız mimarisi sayesinde istediğiniz dilde model eğitmenize olanak tanır.

*"Sınırlı kaynaklarla küresel teknoloji devlerine meydan okuyan, Türk gençliğinin vizyonuyla şekillenmiş bir özgürlük manifestosu. Bu sadece bir model değil; kendi yapay zeka dünyanızı inşa etmeniz için tasarlanmış eksiksiz bir fabrikadır."*

**Türk Gençlerine Armağanımdır.** · Open Source · Full-Stack AI Engine · End-to-End

*Cevahir AI & Engine is a full-stack open-source AI engine that provides an end-to-end infrastructure for building and deploying language models—from tokenizer training to cognitive reasoning layers.*

<p align="center">
  <img src="image/6E09AD00-FEB9-4A66-B577-FB64041723BF.png" style="max-width:100%;">
</p>

---

## Vizyon ve Manifesto

Cevahir, devasa GPU çiftliklerinin ve kapalı kutu algoritmaların hüküm sürdüğü bir çağda, bilginin demokratikleşmesini savunur.

- **Sınırlı Kaynak, Sınırsız İnovasyon:** Büyük bütçelerle değil, optimize edilmiş akıllı mimariyle dünya standartlarında iş çıkarılabileceğinin kanıtıdır.
- **Türk Gençliğine Armağan:** Teknoloji tüketen değil, teknolojiye yön veren bir nesil için bir referans mimaridir.
- **Tam yapay zeka altyapısı:** Tokenizer eğitiminden bilişsel katmana kadar uzanan tam yapay zeka altyapısını tek bir repo içinde sunan **nadir** açık kaynak projelerden biridir; her hücresi açık kaynaktır.

Cevahir **sadece Türkçe ile sınırlı değildir.** Motor, Türkçe için önce optimize edilmiş olsa da **dil bağımsız** bir altyapı sunar; istediğiniz dilde ve veri setiyle kendi modellerinizi eğitebilirsiniz.

---

## Özellikler

- **Türkçe BPE tokenizer** — Uçtan uca eğitim, vocab/merges, encode-decode pipeline (Unicode, İ/ı kuralları, heceleme, morfoloji desteği)
- **Transformer decoder** — RoPE, RMSNorm, SwiGLU, causal mask, weight tying, KV cache, Flash Attention altyapısı
- **Model Management** — Build, eğitim bileşenleri, save/load, forward/predict, TensorBoard
- **Cognitive Management** — Strateji (direct/think/debate/tot), bellek (RAG, vector DB), critic, araç kullanımı, middleware, izleme
- **Sohbet pipeline** — ChattingManager, oturum/geçmiş, Cevahir unified API ile sohbet asistanı akışı

---

## Cevahir Engine: Sadece Bir Model Değil, Bir Ekosistem

**Cevahir AI & Engine**, dil modeli inşa etmek için **uçtan uca (end-to-end)** altyapı sunan **full-stack açık kaynak bir AI engine**’dir. Pek çok açık kaynak proje yalnızca *training framework* sunar (tokenizer → model → eğitim → inference). Cevahir’de buna ek olarak **sohbet sistemi**, **cognitive management** (strateji katmanları: think / debate / ToT), **unified engine API**, **araç kullanımı** ve **RAG bellek** tek bir repo’da yer alır; bu yapı projeyi *AI engine / full-stack AI* kategorisine taşır.

Cevahir'in kalbi olan bu altyapı ile **istediğiniz dilde**, istediğiniz veri setiyle kendi özel yapay zeka modellerinizi eğitebilirsiniz.

### 1. Türkçe Odaklı Hibrit Tokenizer (BPE)

Cevahir Engine, Türkçe dil yapısına "yerli" bir bakış açısıyla yaklaşır; aynı altyapı **dil bağımsız** mimariye sahiptir:

- **Byte Pair Encoding (BPE):** Türkçe'nin eklemeli yapısını, Unicode karakterlerini (İ/ı, Ş/ş vb.) ve morfolojik özelliklerini tanıyan özel bir encoding süreci. Diğer diller için de vocab/merges yeniden eğitilerek kullanılabilir.
- **Dil Bağımsız Mimari:** Altyapı Türkçe için optimize edilmiş olsa da, motor **tüm dünya dillerinde** yüksek performanslı modeller üretme kapasitesine sahiptir; proje Türkçe ile sınırlı değildir.
- **GPU Destekli Batch Tokenization:** Milyonlarca satır veriyi saniyeler içinde işleme yeteneği.

### 2. Esnek Model Mimarisi (Transformer V-4)

- **Modüler Yapı:** RoPE, RMSNorm, SwiGLU ve Flash Attention gibi modern bileşenlerle kendi sinir ağı konfigürasyonunuzu (katman sayısı, kafa sayısı, boyut) saniyeler içinde tanımlayın.
- **Sınırsız Model Üretimi:** Kendi dikey uzmanlık modellerinizi (Hukuk, Tıp, Yazılım vb.) sıfırdan eğitebilir veya mevcut ağırlıklar üzerinden devam edebilirsiniz.

### 3. Cognitive Management (Bilişsel Yönetim)

Modelin sadece metin üretmesini değil, **düşünmesini** sağlar:

- **Strateji Katmanları:** Direct, Think, Debate ve Tree of Thoughts (ToT) ile karmaşık problemleri çözme yeteneği.
- **Dinamik Bellek:** RAG ve Vector DB entegrasyonu ile modelin güncel verilerle konuşmasını sağlayan yapı hazır haldedir.

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
- **TokenizerCore** — BPE (Türkçe odaklı, dil bağımsız), GPU batch, OOV hece fallback
- **ModelManager** — Model yaşam döngüsü, checkpoint, TensorBoard
- **CognitiveManager** — handle(), strateji seçimi, bellek, critic, register_tool(), get_metrics()

---

## Kurulum

Projeyi GitHub’dan indirip kendi ortamınızda çalıştırmanız gerekir. Bağımlılıklar (`requirements.txt` vb.) yaklaşık **200’e yakın kütüphane** içerebilir; kurulum ve ortam yapılandırması için Python, pip/venv ve gerekirse CUDA/PyTorch konusunda bilgi sahibi olmanız önerilir. Detaylı adımlar proje yapısına ve kullandığınız sürümlere göre değişebilir; bu konuda sorumluluk indiren geliştiricidedir.

---

## Hızlı Başlangıç (Kendi Modelini Eğit)

```python
from model.cevahir import Cevahir, CevahirConfig

# 1. Kendi mimarinizi tanımlayın
config = CevahirConfig(
    device="cuda",  # veya "cpu"
    model={
        "vocab_size": 60000,  # Türkçe BPE ile optimize; diğer diller için yeniden eğitilebilir
        "embed_dim": 512,    # Kendi kapasitenizi belirleyin
        "num_layers": 8,
        "num_heads": 8,
    }
)

# 2. Motoru başlatın
cevahir = Cevahir(config)

# 3. Sohbet (cognitive katmanı ile)
output = cevahir.process("Merhaba, nasılsın?")
print(output.response)

# 4. Metin üretimi
text = cevahir.generate("Türkiye'nin başkenti", max_new_tokens=50, temperature=0.8)
print(text)

# Kendi verinizle eğitim için: training_system/ rehberine bakın.
```

---

## Terminal ile test

Eğitilmiş model ile sohbet ve üretim testleri **terminal** üzerinden `chat_pipeline.py` ile yapılabilir:

```bash
python model_management/chat_pipeline.py
```

(Bu script, Cevahir + ChattingManager pipeline'ını kullanır; checkpoint veya kaydedilmiş model gerekir.)

---

## Eğitim Esnasında Elde Edilen Örnek Çıktılar - Example Generation During Training

Eğitim sırasında veya epoch sonu testlerinde **TrainingServiceV2** ile alınan inference örnekleridir: modele verilen prompt, üretilen yanıt, token sayısı ve EOS bilgisi logda görülür. Aşağıya bu tür ekran görüntülerini ekleyebilirsiniz.

### Örnek çıktı görselleri

<p align="center">
  <img src="image/1.jpeg" style="max-width:100%;">
</p>
<p align="center">
  <img src="image/2.jpeg" style="max-width:100%;">
</p>
<p align="center">
  <img src="image/3.jpeg" style="max-width:100%;">
</p>
<p align="center">
  <img src="image/4.jpeg" style="max-width:100%;">
</p>
<p align="center">
  <img src="image/5.jpeg" style="max-width:100%;">
</p>
<p align="center">
  <img src="image/6.jpeg" style="max-width:100%;">
</p>
---

## Eğitim

### Eğitim verisi

Model eğitimi için kullanılan veri seti **yaklaşık 680 bin örnek** içerir. Eğitim verisini kendi ortamınızda kullanmak isterseniz aşağıdaki bağlantıdan indirebilirsiniz:

- **[Eğitim verisi (Google Drive)](https://drive.google.com/drive/folders/19G5uGS5YM3rf42OefjM3KsXRyn0ZEshW?usp=sharing)** — ~680k örnek (docx, txt, soru–cevap json vb.); `prepare_cache.py` ile uyumlu formata dönüştürülebilir.

### Eğitilmiş model (indirme)

Sıfırdan eğitim yapmadan doğrudan inference veya sohbet denemek isterseniz, hazır eğitilmiş model ağırlıklarını indirebilirsiniz:



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

## Proje Yapısı ve Modülerlik

Cevahir Engine, **SOLID** prensipleriyle **12 ana çatı modül** ve **653+** modül dosyası (.py) üzerine kuruludur:

- **tokenizer_management/** — Kendi BPE tokenizer'ınızı sıfırdan eğitin (Türkçe veya başka dil).
- **training_system/** — Kendi veri setinizle model eğitimini başlatın.
- **cognitive_management/** — Modele karar verme yetisi kazandırın.
- **src/** — V-4 Neural Network çekirdeğine müdahale edin.
- **model/** — Unified API (cevahir.py).
- **model_management/** — Model yaşam döngüsü (build, save/load, forward).
- **chatting_management/** — Sohbet (ChattingManager, oturum, context).
- **docs/** — Dokümantasyon, hata debugging süreçleri.

```
cevahir_sinir_sistemi/
├── model/                 # Unified API (cevahir.py)
├── cognitive_management/  # Bilişsel katman (strateji, bellek, critic, tools)
├── model_management/      # Model yaşam döngüsü (build, save/load, forward)
├── training_system/       # Eğitim pipeline (train.py, v2)
├── tokenizer_management/  # BPE (Türkçe odaklı, dil bağımsız)
├── src/                   # Sinir ağı (CevahirNeuralNetwork, V-4)
├── chatting_management/   # Sohbet (ChattingManager, oturum, context)
└── docs/                  # Dokümantasyon
```

Çatı modüller: model, cognitive_management, model_management, training_system, tokenizer_management, src, chatting_management, data_loader_management, training_management, openai-data-mining, api, data_processing.

---

## Neden Açık Kaynak?

Yapay zeka geleceğin teknolojisi; ancak kaynaklar çoğu zaman büyük şirketlerin elinde veya yalnızca eğitim framework’üne indirgenmiş durumda. Cevahir AI & Engine, **full-stack** ve **end-to-end** bir AI engine olarak:

- **Uçtan uca** bir dil modeli motorunu (tokenizer → model → cognitive) inceleyebilir, **istediğiniz dilde** kendi modellerinizi eğitebilirsiniz; proje Türkçe ile sınırlı değildir.
- Tokenizer eğitimi, Transformer mimarisi, model eğitimi, sohbet ve bilişsel stratejiler **tek bir projede** şeffaf biçimde yer alır.
- Proje, Türk gençlerinin (ve tüm geliştiricilerin) yapay zekayı anlaması ve geliştirmesi için bir **eğitim kaynağı** ve **referans mimari** olarak tasarlandı.

---

## Dokümantasyon

- **Mimari:** `docs/` — sistem mimarisi, katmanlar, veri akışı
- **API referansı:** `docs/` — Cevahir, ModelManager, CognitiveManager, TokenizerCore kullanımı
- **Modül dokümantasyonu:** `model/`, `cognitive_management/`, `model_management/`, `training_system/`, `tokenizer_management/`, `src/` altındaki modüllerde docstring ve README dosyaları
- **Eğitim rehberi:** `training_system/train.py`, `tokenizer_management/train_bpe.py`, `prepare_cache.py` — parametreler ve akış için bu dosyalar incelenebilir
- **Inference / sohbet:** `model_management/chat_pipeline.py`, `model/cevahir.py` — kullanım örnekleri ve config

Proje kökünde yalnızca bu README ve kaynak kod yer alır; ek metinler (tanıtım, süreç özeti vb.) depoda ayrı tutulmaz.

---

## Lisans ve Katkı

Proje **Apache License 2.0** ile lisanslanmıştır; tam açık kaynaktır ve herkesin erişimine açıktır. Detaylar için repo kökündeki `LICENSE` dosyasına bakın. **Dünyanın her yerinden** geliştiricilerin katkısına açıktır.

Katkılar memnuniyetle karşılanır. Fork, feature branch, commit ve Pull Request adımlarını izleyebilirsiniz.

---

## İletişim

- **GitHub:** [@myylogic](https://github.com/myylogic)
- **X (Twitter):** [@myylogic](https://x.com/myylogic)
- **Instagram:** [@myylogic](https://instagram.com/myylogic)
- **Proje:** Cevahir AI — Türk Gençlerine Armağanımdır.

---

**Geliştirici:** Muhammed Yasin Yılmaz ([@myylogic](https://github.com/myylogic)) · **Durum:** Açık Kaynak / Aktif Geliştirme · **Tarih:** 09.03.2026

<p align="center">
  <img src="image/myy.jpeg" style="max-width:100%;">
</p>

<p align="center">
Cevahir AI & Engine – Creator · Turkish AI Researcher
</p>
