# Greedy Decode Validation Test

## 📋 Genel Bakış

Bu test, modelin gerçekten öğrenip öğrenmediğini kontrol eder. Greedy decode (temperature=0, argmax) stratejisi kullanarak modelin en iyi token'ları seçip seçmediğini test eder.

## 🎯 Test Amacı

- Model'in gerçekten öğrenip öğrenmediğini doğrulamak
- Greedy decode stratejisinin çalışıp çalışmadığını test etmek
- Model çıktılarının anlamlı olup olmadığını kontrol etmek
- Tekrarlama problemini analiz etmek

## 🚀 Test Çalıştırma

### Basit Kullanım

```bash
# Testi çalıştır
python tests/test_greedy_decode_validation.py
```

### Beklenen Çıktılar

Test şu dosyaları oluşturur:

1. **`test_greedy_decode.log`** - Detaylı log dosyası
2. **`test_greedy_decode_results.json`** - Test sonuçları (JSON format)

## 📊 Test Senaryoları

Test şu prompt'ları kullanır:

1. **"selam beni anlayabiliyor musun"** - Basit selamlaşma
2. **"merhaba nasılsın"** - Günlük konuşma
3. **"Türkiye'nin başkenti neresidir"** - Bilgi sorusu

Her prompt için:
- Greedy decode (temperature=0.0) ile generation
- Sampling decode (temperature=1.0) ile karşılaştırma
- Detaylı analiz (anlamlı kelimeler, tekrarlama oranı, prompt uyumu)

## 📈 Test Metrikleri

Test şu metrikleri ölçer:

- **Anlamlı Kelimeler:** Çıktıda anlamlı Türkçe kelime sayısı
- **Tekrarlama Oranı:** Tekrarlanan kelime oranı (0-1 arası)
- **Prompt Uyumu:** Çıktının prompt ile alakalılığı
- **Genel Başarı:** Tüm kriterlere göre başarı durumu

## ✅ Başarı Kriterleri

Test başarılı kabul edilir eğer:

- ✅ Anlamlı kelimeler üretiliyor
- ✅ Makul uzunlukta çıktı (>10 karakter)
- ✅ Tekrarlama oranı < %50
- ✅ Prompt ile alakalı kelimeler var

## 📝 Sonuç Yorumlama

### Başarılı Test (Success Rate >= %50)

Model öğrenmiş ve greedy decode mantıklı sonuçlar üretiyor. Decode stratejisi çalışıyor demektir.

**Sonraki Adımlar:**
- Repetition penalty optimizasyonu
- Decode parametrelerini fine-tune etme

### Başarısız Test (Success Rate < %50)

Model öğrenmiş ama greedy decode anlamlı sonuç üretemiyor. Decode stratejisinde sorun var demektir.

**Sonraki Adımlar:**
- Single step generation debug testi
- Logits distribution analizi
- Model state kontrolü (eval mode, dropout)

## 🔍 Detaylı Analiz

Test sonuçları JSON dosyasında şu bilgileri içerir:

```json
{
  "test_date": "2024-12-21T...",
  "model_path": "saved_models/cevahir_model.pth",
  "test_results": [
    {
      "prompt": "...",
      "greedy_decode": {
        "response": "...",
        "response_length": 123,
        "token_count": 20
      },
      "analysis": {
        "meaningful_words": ["kelime1", "kelime2"],
        "repetition_ratio": 0.15,
        "prompt_relevance": "PARTIAL",
        "overall_success": true
      }
    }
  ],
  "summary": {
    "total_tests": 3,
    "successful_tests": 2,
    "failed_tests": 1,
    "success_rate": 0.67
  }
}
```

## 🐛 Troubleshooting

### Model yüklenemiyor

- `saved_models/cevahir_model.pth` dosyasının var olduğundan emin olun
- Vocab ve merges dosyalarının doğru yolda olduğunu kontrol edin

### Memory hatası

- CUDA kullanıyorsanız, CPU'ya geçmeyi deneyin
- Batch size'ı azaltın

### Import hatası

- Proje kök dizininden çalıştırdığınızdan emin olun
- `sys.path` doğru ayarlanmış olmalı

## 📚 İlgili Dokümanlar

- `MODUL_TEST_YOL_HARITASI.md` - Genel test planı
- `egitim-sonuclari-genel-degerlendirme.json` - Eğitim sonuçları raporu

