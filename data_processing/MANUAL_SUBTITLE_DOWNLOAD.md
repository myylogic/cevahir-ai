# Türkçe Altyazı Manuel İndirme Rehberi

## 📥 İndirme Kaynakları

### 1. Opensubtitles.org (Önerilen)
**URL:** https://www.opensubtitles.org
**Durum:** ✅ Aktif, ücretsiz kayıt
**Format:** SRT, VTT
**Türkçe Altyazı:** ✅ Mevcut

**Kullanım:**
1. https://www.opensubtitles.org adresine git
2. Kayıt ol (ücretsiz)
3. Dizi/film adını ara
4. Türkçe altyazıları filtrele
5. İndir

### 2. Subscene.com
**URL:** https://subscene.com
**Durum:** ✅ Aktif
**Format:** SRT
**Türkçe Altyazı:** ✅ Mevcut

**Kullanım:**
1. https://subscene.com adresine git
2. Dizi/film adını ara
3. Türkçe altyazıları seç
4. İndir

### 3. YifySubtitles
**URL:** https://yifysubtitles.com
**Durum:** ✅ Aktif
**Format:** SRT
**Türkçe Altyazı:** ⚠️ Sınırlı

---

## 📋 İndirilecek Diziler

1. **Kurtlar Vadisi**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Kurtlar+Vadisi
   - Subscene: https://subscene.com/subtitles/search?q=Kurtlar+Vadisi

2. **Ezel**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Ezel
   - Subscene: https://subscene.com/subtitles/search?q=Ezel

3. **Muhteşem Yüzyıl**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Muhteşem+Yüzyıl
   - Subscene: https://subscene.com/subtitles/search?q=Muhteşem+Yüzyıl

4. **Diriliş: Ertuğrul**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Diriliş+Ertuğrul
   - Subscene: https://subscene.com/subtitles/search?q=Diriliş+Ertuğrul

5. **Çukur**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Çukur
   - Subscene: https://subscene.com/subtitles/search?q=Çukur

6. **Behzat Ç.**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Behzat+Ç
   - Subscene: https://subscene.com/subtitles/search?q=Behzat+Ç

7. **Şahsiyet**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Şahsiyet
   - Subscene: https://subscene.com/subtitles/search?q=Şahsiyet

8. **Fi**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Fi
   - Subscene: https://subscene.com/subtitles/search?q=Fi

9. **Avrupa Yakası**
   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Avrupa+Yakası
   - Subscene: https://subscene.com/subtitles/search?q=Avrupa+Yakası

10. **Çocuklar Duymasın**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Çocuklar+Duymasın
    - Subscene: https://subscene.com/subtitles/search?q=Çocuklar+Duymasın

11. **Kara Para Aşk**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Kara+Para+Aşk
    - Subscene: https://subscene.com/subtitles/search?q=Kara+Para+Aşk

12. **İçerde**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-İçerde
    - Subscene: https://subscene.com/subtitles/search?q=İçerde

13. **Masum**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Masum
    - Subscene: https://subscene.com/subtitles/search?q=Masum

14. **Börü 2039**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-Börü+2039
    - Subscene: https://subscene.com/subtitles/search?q=Börü+2039

15. **50m2**
    - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-50m2
    - Subscene: https://subscene.com/subtitles/search?q=50m2

---

## 🔄 İşlem Adımları

### 1. Altyazıları İndir
- Yukarıdaki kaynaklardan manuel indir
- `data_processing/subtitles/` klasörüne kaydet
- Her dizi için ayrı klasör oluştur (örn: `Kurtlar_Vadisi/`)

**Klasör Yapısı:**
```
data_processing/
  subtitles/
    Kurtlar_Vadisi/
      S01E01.srt
      S01E02.srt
      ...
    Ezel/
      S01E01.srt
      ...
```

### 2. Altyazıları İşle
```bash
python data_processing/subtitle_processor.py --input data_processing/subtitles --output data_processing/subtitles_processed --format txt
```

### 3. Eğitim Verisine Ekle
- İşlenmiş altyazıları `education/` klasörüne taşı
- BPE training ve model eğitimi sırasında otomatik yüklenecek

---

## ⚠️ Önemli Notlar

- **Telif Hakları:** Altyazılar telif hakkı koruması altında olabilir. Sadece eğitim amaçlı kullanın.
- **Kalite Kontrolü:** İndirilen altyazıları kontrol edin, bozuk dosyaları silin.
- **Format:** SRT formatı tercih edilir (en yaygın).
- **Encoding:** UTF-8 encoding kullanın.

---

## 🤖 Otomatik İndirme (API ile)

Opensubtitles API ile otomatik indirme için:

1. **Kütüphane Yükle:**
   ```bash
   pip install opensubtitlescom
   # veya
   pip install subliminal
   ```

2. **API Key Al:**
   - https://www.opensubtitles.org adresine kayıt ol
   - API key al (ücretsiz)

3. **Script Çalıştır:**
   ```bash
   # Tek dizi
   python data_processing/subtitle_downloader.py --dizi "Kurtlar Vadisi" --sezon 1 --bölüm 1
   
   # Toplu indirme
   python data_processing/subtitle_downloader.py --batch data_processing/turkce_dizi_listesi.json
   
   # Dizi listesi ile
   python data_processing/subtitle_downloader.py --dizi-listesi "Kurtlar Vadisi,Ezel,Behzat Ç."
   ```

**NOT:** API rate limiting'e dikkat edin. Çok fazla istek göndermeyin.

---

## 📊 Tahmini Veri Miktarı

Her dizi için:
- **Ortalama sezon sayısı:** 3-5 sezon
- **Ortalama bölüm sayısı:** 20-30 bölüm/sezon
- **Altyazı başına token:** ~500-2000 token
- **Toplam tahmini:** ~15-30 milyon token (tüm diziler için)

---

## ✅ Kontrol Listesi

- [ ] Opensubtitles.org'a kayıt ol
- [ ] Her dizi için altyazıları indir
- [ ] `data_processing/subtitles/` klasörüne kaydet
- [ ] Altyazıları işle (`subtitle_processor.py`)
- [ ] İşlenmiş altyazıları `education/` klasörüne taşı
- [ ] BPE training çalıştır
- [ ] Model eğitimi başlat

---

**İyi çalışmalar! 🚀**

