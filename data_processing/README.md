# DATA PROCESSING - VERİ İŞLEME SCRİPTLERİ

## 📁 Klasör Yapısı

```
data_processing/
  ├── output/              # Scraping çıktıları (manuel olarak education/ klasörüne taşınacak)
  │   ├── tarih/
  │   ├── islam_tarihi/
  │   ├── dinler_tarihi/
  │   ├── dunya_tarihi/
  │   ├── hastaliklar_tarihi/
  │   └── ...
  ├── pdf_to_docx_converter.py
  ├── subtitle_processor.py
  ├── wikipedia_api_scraper.py
  └── topic_based_scraper.py
```

---

## 🎯 KONU BAZLI SCRAPING

### Desteklenen Konular:

**Tarih ve Kültür:**
- `tarih` - Genel tarih
- `islam_tarihi` - İslam tarihi
- `dinler_tarihi` - Dinler tarihi
- `dunya_tarihi` - Dünya tarihi
- `hastaliklar_tarihi` - Hastalıklar ve tıp tarihi
- `edebiyat_tarihi` - Edebiyat tarihi
- `sanat_tarihi` - Sanat tarihi

**Bilim (Derinlemesine):**
- `bilim_tarihi` - Bilim tarihi
- `kimya` - Kimya bilimi (organik, inorganik, fiziksel, analitik, biyokimya, kuantum kimyası)
- `fizik` - Fizik bilimi (klasik, modern, kuantum, görelilik, termodinamik, mekanik, optik)
- `biyoloji` - Biyoloji bilimi (moleküler, hücre, genetik, evrim, ekosistem, anatomi, fizyoloji)
- `matematik` - Matematik
- `astronomi` - Astronomi, gezegenler, galaksiler, evren, kozmoloji, kara delikler
- `uzay` - Uzay araştırmaları (NASA, ESA, roket, uydu, uzay istasyonu, Mars keşfi)
- `kuantum_fizigi` - Kuantum fiziği (kuantum mekaniği, belirsizlik, dolanıklık, kuantum bilgisayar)
- `fizik_kanunlari` - Fizik kanunları (Newton, termodinamik, enerji korunumu, Maxwell, Einstein)

**Sağlık ve Coğrafya:**
- `saglik` - Sağlık ve tıp
- `kitalar` - Kıtalar ve coğrafya
- `cografya` - Coğrafya bilimi

**Sosyal Bilimler:**
- `ekonomi` - Ekonomi
- `felsefe` - Felsefe
- `psikoloji` - Psikoloji
- `sosyoloji` - Sosyoloji

**Teknoloji ve İcatlar:**
- `teknoloji` - Teknoloji
- `icatlar` - İcatlar ve buluşlar (elektrik, elektronik, uzay, tıp, iletişim, enerji)
- `elektrik` - Elektrik (akım, voltaj, direnç, jeneratör, motor, pil, elektrik üretimi)
- `elektronik` - Elektronik (yarı iletken, transistör, entegre devre, mikroçip, sensör)
- `muzik` - Müzik
- `spor` - Spor

**Bilim İnsanları ve Mucitler:**
- `bilim_insanlari` - Bilim insanları (Einstein, Newton, Darwin, Curie, Tesla, Hawking)
- `mucitler` - Mucitler (Edison, Tesla, Bell, Wright kardeşler, Gutenberg, Musk)

### Derinlik Seviyeleri:
- `shallow` - 50 sayfa/kategori (hızlı test)
- `medium` - 100 sayfa/kategori (önerilen)
- `deep` - 200 sayfa/kategori (kapsamlı)

---

## 📝 KULLANIM ÖRNEKLERİ

### 1. Tarih (Orta Derinlik)
```bash
python data_processing/topic_based_scraper.py \
  --topic tarih \
  --depth medium \
  --output data_processing/output
```

### 2. İslam Tarihi (Derin)
```bash
python data_processing/topic_based_scraper.py \
  --topic islam_tarihi \
  --depth deep \
  --output data_processing/output
```

### 3. Dinler Tarihi (Sınırlı)
```bash
python data_processing/topic_based_scraper.py \
  --topic dinler_tarihi \
  --depth medium \
  --limit 200 \
  --output data_processing/output
```

### 4. Hastalıklar Tarihi
```bash
python data_processing/topic_based_scraper.py \
  --topic hastaliklar_tarihi \
  --depth deep \
  --output data_processing/output
```

---

## 🔄 İŞ AKIŞI

1. **Scraping:** `data_processing/output/` klasörüne kaydet
2. **Kontrol:** Dosyaları kontrol et
3. **Manuel Taşıma:** `education/` klasörüne taşı
4. **Eğitim:** `train.py` veya `train_bpe.py` çalıştır

---

## ⚠️ ÖNEMLİ NOTLAR

### Format Seçimi:
- **TXT formatı önerilir** (RAW_TEXT modu ile uyumlu)
- JSON formatı QA formatına uygun değil (Wikipedia bilgi metinleri, soru-cevap değil)
- Data loader RAW_TEXT modunda `.txt` dosyalarını otomatik okur

### Rate Limiting:
- Script otomatik olarak 0.5 saniye bekleme yapar
- API limitlerini aşmamak için

### Dosya Organizasyonu:
- Her konu için ayrı alt klasör
- Örnek: `data_processing/output/tarih/`, `data_processing/output/islam_tarihi/`

---

## 📊 BEKLENEN SONUÇLAR

### Shallow (50 sayfa/kategori):
- ~50-100 dosya
- ~250,000-500,000 karakter
- ~50,000-100,000 token

### Medium (100 sayfa/kategori):
- ~100-200 dosya
- ~500,000-1,000,000 karakter
- ~100,000-200,000 token

### Deep (200 sayfa/kategori):
- ~200-400 dosya
- ~1,000,000-2,000,000 karakter
- ~200,000-400,000 token

