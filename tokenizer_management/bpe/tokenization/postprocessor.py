# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: postprocessor.py
Modül: tokenizer_management/bpe/tokenization
Görev: Postprocessor sınıfı - Token çıktılarını post-processing işlemlerinden
       geçirir. Özel tokenları dönüştürür, token listesini birleştirir,
       Unicode normalizasyonu yapar, Türkçe karakter düzeltmeleri yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (post-processing işlemleri)
- Design Patterns: Pipeline Pattern (adım adım işleme)
- Endüstri Standartları: Unicode normalizasyon, Türkçe karakter desteği

KULLANIM:
- BPEManager decode pipeline'ında kullanılır
- Token → metin dönüşümü sonrası düzeltmeler için
- Özel token (BOS/EOS/SEP/PAD) yönetimi için

BAĞIMLILIKLAR:
- unicodedata: Unicode normalizasyonu
- DEFAULT_SPECIAL_TOKENS: Özel token yapılandırması
- DEFAULT_PUNCTUATION_FIXES: Noktalama düzeltmeleri
- TURKISH_CAPITALIZATION: Türkçe büyük/küçük harf kuralları
- PostProcessingError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
import re
import unicodedata
from typing import List, Dict, Optional, Pattern, Tuple, Union

from tokenizer_management.config import DEFAULT_SPECIAL_TOKENS, DEFAULT_PUNCTUATION_FIXES, TURKISH_CAPITALIZATION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class PostProcessingError(Exception):
    """Postprocessing adımlarından herhangi birinde oluşan hata için özel istisna."""
    pass

class Postprocessor:
    """
    BPE veya diğer token çıktılarını alıp aşağıdaki adımları uygulayarak
    okunabilir, düzgün formatlanmış düz metne dönüştürür:

      1) Özel tokenları dönüştürme veya çıkarma
      2) Token listesini birleştirme + Unicode NFC normalizasyonu
      3) Noktalama etrafındaki yanlış boşlukları düzeltme
      4) Genel fazla boşluk temizliği
      5) Arka arkaya gelen nokta (.) tekrarlarını tekilleştirme
      6) Cümle başı büyük harfe çevirme (Türkçe 'i' ve 'ı' için özel)
    """

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        punctuation_fixes: Optional[Dict[str, str]] = None,
        capitalize_sentence: Optional[bool] = None,
        config: Optional[Dict] = None
    ):
        """
        :param special_tokens: Özel token → metin dönüşüm haritası
        :param punctuation_fixes: Regex paterni → ikame metin haritası
        :param capitalize_sentence: Cümle başlarını büyük harfe çevirilsin mi?
        :param config: Config dict (BPE_DETAILED_CONFIG)
        """
        # Config merge
        from tokenizer_management.config import BPE_DETAILED_CONFIG
        self.config = {**BPE_DETAILED_CONFIG}
        if config:
            self.config.update(config)
        
        # capitalize_sentence config'ten
        if capitalize_sentence is None:
            capitalize_sentence = self.config.get("capitalize_sentence", True)
        self.capitalize_sentence = capitalize_sentence
        
        # Özel token tablosunu derin kopya ile al
        self.special_tokens: Dict[str, str] = (
            special_tokens.copy()
            if special_tokens is not None
            else dict(DEFAULT_SPECIAL_TOKENS)
        )

        # Noktalama düzeltme regex haritası
        fixes = punctuation_fixes or DEFAULT_PUNCTUATION_FIXES
        self.punctuation_fixes: List[Tuple[Pattern[str], str]] = []
        for pattern_str, replacement in fixes.items():
            try:
                pattern = re.compile(pattern_str)
            except re.error as e:
                logger.error(f"[X] Geçersiz regex paterni '{pattern_str}': {e}")
                raise PostProcessingError(f"Invalid punctuation pattern: {pattern_str}") from e
            self.punctuation_fixes.append((pattern, replacement))

        self.capitalize_sentence = capitalize_sentence

        # Cümle ayırıcı: . ! ? ardından gelen boşlukları ve string sonunu yakala
        self._sentence_split_re = re.compile(r'([.!?])\s*|\s+|$')
        # Genel fazla boşluk temizleme
        self._whitespace_re = re.compile(r'\s+')

        logger.info("[+] Postprocessor başlatıldı.")

    def process(self, tokens: List[str], preserve_punctuation: bool = False) -> Union[str, List[str]]:
        """
        Token listesini alır, tüm postprocessing adımlarını sırayla uygular
        ve birleştirilmiş bir string veya token listesi döner.
        
        Args:
            tokens: İşlenecek token listesi.
            preserve_punctuation: True ise, noktalama işaretleri ayrı tokenlar olarak korunur ve liste döner.
        
        Returns:
            str: Eğer preserve_punctuation=False ise, birleştirilmiş ve normalize edilmiş string.
            List[str]: Eğer preserve_punctuation=True ise, işlenmiş token listesi.
        
        Raises:
            PostProcessingError: İşlem sırasında hata oluşursa.
        """
        try:
            # 1) Özel token dönüşümü (liste bazlı)
            raw_tokens: List[str] = []
            for tok in tokens:
                mapped = self.special_tokens.get(tok, tok)
                if mapped:
                    raw_tokens.append(mapped)

            if not raw_tokens:
                logger.warning("[!] Boş token listesi alındı.")
                return [] if preserve_punctuation else ""

            # 2) Varsayılan special_tokens kullanılıyorsa, placeholder ve noktalama
            #    öğelerini sona alacak şekilde yeniden sırala
            if self.special_tokens == DEFAULT_SPECIAL_TOKENS:
                known = [t for t in raw_tokens if t not in ("[UNK]", ".", ",", "!", "?", ";", ":")]
                placeholders = [t for t in raw_tokens if t == "[UNK]"]
                punctuation = [t for t in raw_tokens if t in (".", ",", "!", "?", ";", ":")]
                # Ardışık [UNK]'ları tekilleştir
                if placeholders:
                    placeholders = ["[UNK]"]
                raw_tokens = known + placeholders + punctuation

            # 3) Eğer token listesi isteniyorsa, direkt döndür
            if preserve_punctuation:
                logger.debug(f"[+] Postprocess tamamlandı, noktalama korundu: {raw_tokens}")
                return raw_tokens

            # 4) Listeyi birleştir
            text = " ".join(raw_tokens)
            logger.debug(f"[+] Özel token sonrası: '{text}'")

            # 5) Unicode NFC normalizasyonu
            text = self._normalize_unicode(text)
            logger.debug(f"[+] Unicode normalize sonrası: '{text}'")

            # 6) Noktalama etrafındaki yanlış boşlukları düzelt
            text = self._fix_punctuation_spacing(text)
            logger.debug(f"[+] Noktalama boşluk düzeltme sonrası: '{text}'")

            # 7) Birden fazla boşluğu tek boşluğa indir
            text = self._collapse_whitespace(text)
            logger.debug(f"[+] Boşluk temizleme sonrası: '{text}'")

            # 8) Arka arkaya gelen nokta tekrarlarını tekilleştir
            text = re.sub(r"\.{2,}", ".", text)
            logger.debug(f"[+] Ardışık nokta tekilleştirme sonrası: '{text}'")

            # 9) Cümle başı büyük harfe çevir (Türkçe i ve ı düzeltmeli)
            if self.capitalize_sentence:
                text = self._capitalize_sentences(text)
                logger.debug(f"[+] Cümle başı büyük sonrası: '{text}'")

            # Son boşlukları temizle
            text = text.rstrip()
            logger.debug(f"[+] Son temizlik sonrası: '{text}'")
            return text

        except PostProcessingError:
            raise
        except Exception as e:
            logger.error(f"[X] Postprocessor genel hatası: {e}", exc_info=True)
            raise PostProcessingError(str(e)) from e

    def _apply_special_tokens(self, tokens: List[str]) -> str:
        if not tokens:
            logger.warning("[!] Boş token listesi alındı.")
            return ""
        out: List[str] = []
        for tok in tokens:
            mapped = self.special_tokens.get(tok, tok)
            if mapped:
                out.append(mapped)
        result = " ".join(out)
        logger.debug(f"[+] Özel token sonrası: '{result}'")
        return result

    def _normalize_unicode(self, text: str) -> str:
        try:
            # Geçersiz Unicode karakterleri temizle (surrogate'ler vb.)
            text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
            normalized = unicodedata.normalize("NFC", text)
            logger.debug(f"[+] Unicode normalize sonrası: '{normalized}'")
            return normalized
        except UnicodeError as e:
            logger.warning(f"[!] Unicode normalizasyon hatası: {e}")
            return text

    def _fix_punctuation_spacing(self, text: str) -> str:
        for pattern, repl in self.punctuation_fixes:
            new_text = pattern.sub(repl, text)
            if new_text != text:
                logger.debug(f"[+] '{pattern.pattern}' → '{repl}' uygulandı: '{new_text}'")
            text = new_text
        return text

    def _collapse_whitespace(self, text: str) -> str:
        collapsed = self._whitespace_re.sub(" ", text).strip()
        logger.debug(f"[+] Boşluk temizleme sonrası: '{collapsed}'")
        return collapsed
    
    def _capitalize_sentences(self, text: str) -> str:
        # Cümleleri ayırırken noktalama işaretlerini ve boşlukları ayrı gruplar olarak yakala
        parts = [part for part in self._sentence_split_re.split(text) if part]  # None ve boş stringleri filtrele
        result = []
        capitalize_next = True  # İlk cümle için büyük harf başlat

        for i, part in enumerate(parts):
            if part in ".!?":
                result.append(part)  # Noktalama işaretini ekle
                # Sonraki kelimeyi büyük harf yap, ama son parça değilse ve None/boş değilse
                if i < len(parts) - 1 and parts[i + 1] and parts[i + 1].strip():
                    capitalize_next = True
                # Noktalama sonrası boşluk ekle, ama string sonunda değil
                if i < len(parts) - 1 and parts[i + 1] and parts[i + 1].strip() and parts[i + 1] not in ".!?":
                    result.append(" ")
                continue
            if part.strip() and capitalize_next:
                # Cümle başlangıcı: İlk harfi Türkçe kurallarına göre büyüt
                stripped = part.strip()
                if stripped:
                    first = stripped[0]
                    # Türkçe'ye özgü büyük harf dönüşümü
                    first = TURKISH_CAPITALIZATION.get(first, first.upper())
                    result.append(first + stripped[1:])
                    capitalize_next = False
                else:
                    result.append(part)
            else:
                # Cümle içi: Olduğu gibi ekle
                result.append(part)
            # Kelimeler arasında boşluk ekle, ama noktalama öncesi veya string sonunda değil
            if (
                not capitalize_next
                and part.strip()
                and part not in ".!?"
                and i < len(parts) - 1
                and parts[i + 1]
                and parts[i + 1].strip()
                and parts[i + 1] not in ".!?"
            ):
                result.append(" ")

        # Sonucu birleştir
        joined = "".join(result)
        # Cümle sonlarında fazladan boşlukları temizle
        joined = re.sub(r'\s+([.!?])', r'\1', joined)
        # String sonundaki noktadan sonra boşluk olmasın
        joined = re.sub(r'\.\s*$', r'.', joined)
        logger.debug(f"[+] Cümle başı büyük sonrası: '{joined}'")
        return joined.rstrip()

    def reset(self) -> None:
        logger.info("[!] Postprocessor resetleniyor...")
        self.__init__(None, None, True)
        logger.info("[+] Postprocessor varsayılanlara döndü.")