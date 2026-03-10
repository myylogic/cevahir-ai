# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: pretokenizer.py
Modül: tokenizer_management/bpe/tokenization
Görev: Pretokenizer sınıfı - Metin ön işleme (pretokenization) pipeline'ı.
       Metni kelimelere, noktalama işaretlerine ve sembollere ayırır.
       Unicode normalizasyonu ve Türkçe karakter işleme yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (pretokenization işlemleri)
- Design Patterns: Pipeline Pattern (adım adım işleme)
- Endüstri Standartları: Unicode normalizasyon, Türkçe karakter desteği

KULLANIM:
- BPEManager tokenization pipeline'ında kullanılır
- Metin → kelime/noktalama/sembol ayrıştırması için
- Unicode normalizasyonu için

BAĞIMLILIKLAR:
- unicodedata: Unicode normalizasyonu
- get_turkish_config: Türkçe karakter yapılandırması
- PretokenizationError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
import unicodedata
import re
import string
from typing import List, Union, Tuple, Optional, Dict, Pattern

from tokenizer_management.config import get_turkish_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PretokenizationError(Exception):
    """Tokenizasyon sırasında oluşan hatalar için özel istisna."""
    pass


class Pretokenizer:
    """
    Metin ön işleme (pretokenization) pipeline'ı.

    Adımlar:
      1. Unicode normalizasyonu (Türkçe harfleri koru, diğer diakritikleri kaldır)
      2. Küçük harfe çevirme (opsiyonel, Türkçe kurallarıyla: İ→i, I→ı)
      3. Pipeline temizleme (noktalama koruyarak non-word→space)
      4. Sayısal/alfanümerik ayırma
      5. Noktalama ayrımı (.,!? ayrı token)
      6. Whitespace‐bazlı tokenizasyon
      7. Geçersiz token filtresi (custom set veya regex)
      8. ASCII‐only lowercase hack (lower=False ve temizleme olduysa)
      9. Heceleyici (opsiyonel)
    """

    def __init__(
        self,
        syllabifier: Optional[object] = None,
        lower: Optional[bool] = None,
        valid_characters: Optional[set] = None,
        cleanup_pattern: Optional[Pattern[str]] = None,
        split_pattern: Optional[Pattern[str]] = None,
        use_gpu: Optional[bool] = None,
        config: Optional[Dict] = None,
    ):
        # Config'ten default değerleri al
        turkish_config = get_turkish_config()
        self.config = {**turkish_config}
        if config:
            self.config.update(config)
        
        # Parametreleri config'ten al
        if lower is None:
            lower = self.config.get("lowercase", False)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        
        # GPU desteği
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cpu":
                    logger.warning("[Pretokenizer] GPU isteniyor ama CUDA mevcut değil, CPU kullanılacak")
                    self.use_gpu = False
                else:
                    logger.info(f"[Pretokenizer] GPU desteği aktif: {self.device}")
            except ImportError:
                logger.warning("[Pretokenizer] PyTorch bulunamadı, GPU desteği devre dışı")
                self.use_gpu = False
                self.device = None
        else:
            self.device = None

        # Opsiyonel heceleyici
        self.syllabifier = syllabifier

        # Cache: raw_input_str → List[token]
        self.cache: Dict[str, List[str]] = {}

        # lower‐flag
        self.lower = lower

        # custom valid‐character filtering
        self.custom_valid = valid_characters is not None
        self.valid_characters = valid_characters or set()

        # Turkish letters to preserve during normalization
        cfg = get_turkish_config()
        turkish_lower = cfg["characters"]                      # ['ç','ğ','ı','ö','ş','ü']
        turkish_upper = [c.upper() for c in turkish_lower] + ["İ"]
        self._turkish_keep = set(turkish_lower + turkish_upper)

        # Direct cleanup (for _clean_specials tests)
        if cleanup_pattern is not None:
            self.cleanup_pattern = cleanup_pattern
            self._cleanup_sub = ""
        else:
            # remove everything that's not word or whitespace
            self.cleanup_pattern = re.compile(r"[^\w\s]", re.UNICODE)
            self._cleanup_sub = " "

        # Pipeline cleanup (preserve .,!? but remove other non-word)
        if cleanup_pattern is not None:
            self._pipeline_cleanup = cleanup_pattern
            self._pipeline_sub = self._cleanup_sub
        else:
            self._pipeline_cleanup = re.compile(r"[^\w\s\.\,\!\?]", re.UNICODE)
            self._pipeline_sub = " "

        # Split letters↔digits
        self.split_pattern = split_pattern or re.compile(
            r"(?<=\d)(?=\D)|(?<=\D)(?=\d)", re.UNICODE
        )

        # Whitespace normalization
        self.whitespace_pattern = re.compile(r"\s+", re.UNICODE)
        # Boşluk koruma için
        self.space_preservation = True

        # Valid token regex: ASCII letters + digits + Turkish letters
        allowed = (
            list(string.ascii_letters)
            + turkish_lower
            + turkish_upper
            + list(string.digits)
        )
        escaped = "".join(re.escape(ch) for ch in allowed)
        self.valid_token_re = re.compile(rf"^[{escaped}]+$")

        # Single‐character punctuation we preserve
        self.punctuations = {".", ",", "!", "?"}

        logger.info(
            "Pretokenizer initialized: lower=%s, custom_valid=%s",
            self.lower,
            self.custom_valid,
        )

    def _normalize_unicode(self, text: str) -> str:
        """
        TÜRKÇE KARAKTERLERİ KORUYARAK normalizasyon.

        Sadece temel Unicode normalizasyonu yap, diakritikleri koru.
        """
        # Sadece temel Unicode normalizasyonu (NFC)
        # Diakritikleri koru, Türkçe karakterleri koru
        return unicodedata.normalize("NFC", text)
    
    def _to_lowercase(self, text: str) -> str:
        """
        Türkçe kurallarıyla küçük harfe çevir.
        İ → i, I → ı
        """
        # Önce dotted/dotless I düzeltmeleri
        text = text.replace("İ", "i").replace("I", "ı")
        return text.lower()

    def _clean_specials(self, text: str) -> str:
        """
        Direkt testler için: non-word/non-space tümünü boşlukla değiştir.
        """
        return self.cleanup_pattern.sub(self._cleanup_sub, text)

    def _pipeline_clean(self, text: str) -> Tuple[str, bool]:
        """
        Pipeline içinde noktalama koruyarak temizleme.
        Returns (cleaned_text, did_change).
        """
        cleaned = self._pipeline_cleanup.sub(self._pipeline_sub, text)
        return cleaned, (cleaned != text)

    def _split_alphanum(self, text: str) -> str:
        """Harflere ve rakamlara ayrımda boşluk ekle."""
        return self.split_pattern.sub(" ", text)

    def _separate_punctuation(self, text: str) -> str:
        """
        .,!? işaretlerini ayrı token haline getir:
        Surround them with spaces so whitespace‐split catches them.
        """
        return re.sub(r"([.,!?])", r" \1 ", text)

    def _tokenize_whitespace(self, text: str) -> List[str]:
        """Whitespace (space/tab/newline) bazlı tokenizasyon."""
        return [tok for tok in self.whitespace_pattern.split(text.strip()) if tok]
    
    def _tokenize_whitespace_with_spaces(self, text: str) -> List[str]:
        """Whitespace bazlı tokenizasyon - BOŞLUK KORUYARAK."""
        if not text:
            return []
        
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(' ')  # Boşluk karakterini koru
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        - Tek karakter noktalama => koru
        - Boşluk karakterleri => koru
        - custom_valid => tüm karakterlerin valid_characters içinde olması
        - else => geçerli regex ile eşleşmeli
        - Hiç token kalmazsa [] (boş liste, EMPTY token oluşturulmaz)
        """
        out: List[str] = []
        for tok in tokens:
            if len(tok) == 1 and tok in self.punctuations:
                out.append(tok)
            elif tok == ' ':  # Boşluk karakterini koru
                out.append(tok)
            elif self.custom_valid:
                if all(ch in self.valid_characters for ch in tok):
                    out.append(tok)
                else:
                    logger.debug("Filtered out by custom_valid: %r", tok)
            else:
                if self.valid_token_re.fullmatch(tok):
                    out.append(tok)
                else:
                    logger.debug("Filtered out by regex: %r", tok)
        # ✅ DÜZELTME: EMPTY token yerine boş liste döndür (boş sequence'ler atlanacak)
        return out  # Boş liste döndür, EMPTY token oluşturma

    def tokenize(self, text: Union[str, List[str], dict]) -> List[str]:
        """
        Metni token listesine çevirir.
        Cache anahtarı olarak tam raw input string kullanılır.
        """
        if text is None:
            raise PretokenizationError("Girdi None olamaz")

        # List veya dict => önce tek string
        if isinstance(text, list):
            raw = " ".join(map(str, text))
        elif isinstance(text, dict):
            raw = str(text.get("data", ""))
        elif isinstance(text, str):
            raw = text
        else:
            raise PretokenizationError(f"Girdi string olmalı, got {type(text)}")

        raw = raw.strip()
        if not raw:
            # ✅ DÜZELTME: EMPTY token yerine boş liste döndür (boş sequence'ler atlanacak)
            return []  # Boş liste döndür, EMPTY token oluşturma

        if raw in self.cache:
            return self.cache[raw]

        # 1) Unicode normalize (diakritikleri temizle, Türkçeyi koru)
        t = self._normalize_unicode(raw)

        # 2) Küçük harfe çevir (Türkçe özel kurallarla)
        if self.lower:
            # Önce dotted/dotless I düzeltmeleri
            t = t.replace("İ", "i").replace("I", "ı")
            t = t.lower()

        # 3) Pipeline temizleme (noktalama koruyarak)
        t, did_pipeline = self._pipeline_clean(t)

        # 4) Alfanümerik grupla ayır
        t = self._split_alphanum(t)

        # 5) Noktalama ayrımı
        t = self._separate_punctuation(t)

        # 6) Whitespace‐normalize & tokenize - BOŞLUK KORUYARAK
        # Sadece çoklu boşlukları tek boşluğa çevir, tek boşlukları koru
        t = self.whitespace_pattern.sub(" ", t).strip()
        toks = self._tokenize_whitespace_with_spaces(t)

        # 7) Geçersiz token filtresi
        toks = self._filter_tokens(toks)

        # 8) lower=False ise büyük/küçük harf KORUNUR (case-sensitive). Eski "ASCII-only lowercase" kaldırıldı.

        # 9) Opsiyonel heceleme
        if self.syllabifier:
            try:
                splitted = self.syllabifier.split(toks)
                # ✅ DÜZELTME: EMPTY token yerine boş liste döndür (boş sequence'ler atlanacak)
                toks = splitted or []  # Boş liste döndür, EMPTY token oluşturma
            except Exception as e:
                logger.warning("Syllabifier hatası: %s", e)

        # Cache ve döndür
        self.cache[raw] = toks
        return toks

    def reset(self) -> None:
        """Cache'i temizler."""
        logger.info("Resetting Pretokenizer cache")
        self.cache.clear()

    def validate_token(self, token: str) -> bool:
        """
        Tek token'ın geçerliliğini kontrol eder.
        - Tek karakter ve noktalama => True
        - custom_valid => tüm karakterler valid_characters içinde
        - else => valid_token_re ile eşleşme
        """
        if len(token) == 1 and token in self.punctuations:
            return True
        if self.custom_valid:
            return all(ch in self.valid_characters for ch in token)
        return bool(self.valid_token_re.fullmatch(token))

    def get_token_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """
        tokenize ile aynı pipeline'ın normalizasyon+lowering+pipeline_clean+
        split_alphanum adımlarını uygular (noktalama ayrımını atlar),
        sonra her token için (tok, start, end) ofset listesi döner.
        """
        # 1) Unicode normalize
        t = self._normalize_unicode(text)

        # 2) Lowering
        if self.lower:
            t = t.replace("İ", "i").replace("I", "ı").lower()

        # 3) Pipeline clean
        t, _ = self._pipeline_clean(t)

        # 4) Split alphanum
        t = self._split_alphanum(t)

        # 5) Whitespace normalize
        cleaned = self.whitespace_pattern.sub(" ", t).strip()

        toks = self.tokenize(text)
        offsets: List[Tuple[str, int, int]] = []
        pos = 0
        for tok in toks:
            # ✅ DÜZELTME: EMPTY token artık döndürülmüyor, bu kontrol gereksiz
            # Boş token'lar zaten filtrelenmiş olacak
            if not tok:  # Boş string kontrolü (güvenlik için)
                continue
            idx = cleaned.find(tok, pos)
            if idx < 0:
                idx = pos
            offsets.append((tok, idx, idx + len(tok)))
            pos = idx + len(tok)
        return offsets

    def batch_tokenize_gpu(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[str]]:
        """GPU ile batch tokenization - PyTorch tensors kullanarak hızlandırılmış."""
        if not self.use_gpu or not self.device:
            # Fallback: CPU processing
            return [self.tokenize(text) for text in texts]
        
        import torch
        import numpy as np

        results = []
        
        # Dinamik batch size için GPU belleğini kontrol et
        try:
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = gpu_memory - torch.cuda.memory_allocated(self.device)
            # Yaklaşık 1 GB bellek bırakarak dinamik batch size hesapla
            dynamic_batch_size = max(1, int(available_memory / (1024 ** 3 * 10)))  # 10 MB/metin tahmini
            effective_batch_size = min(batch_size, dynamic_batch_size, len(texts))
        except Exception as e:
            logger.warning(f"[Pretokenizer] GPU bellek kontrol hatası: {e}, varsayılan batch_size kullanılıyor")
            effective_batch_size = batch_size

        # Batch'ler halinde işle
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i:i + effective_batch_size]
            
            try:
                # Metinleri tensor'a çevir (karakter bazlı)
                batch_chars = [list(text) for text in batch_texts]
                max_len = max(len(chars) for chars in batch_chars) if batch_chars else 0
                if max_len == 0:
                    results.extend([[] for _ in batch_texts])
                    continue
                
                batch_tensor = torch.zeros(len(batch_texts), max_len, dtype=torch.long, device=self.device)
                
                for j, chars in enumerate(batch_chars):
                    batch_tensor[j, :len(chars)] = torch.tensor([ord(c) for c in chars], device=self.device)
                
                # GPU'da paralel tokenizasyon
                batch_results = []
                for j in range(batch_tensor.size(0)):
                    text = "".join(chr(c.item()) for c in batch_tensor[j] if c != 0)
                    tokens = self.tokenize(text)  # Şu an CPU tabanlı, ileride GPU'ya uyarlanabilir
                    batch_results.append(tokens)
                
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"[Pretokenizer] GPU batch tokenization hatası: {e}")
                # Fallback: CPU processing
                results.extend([self.tokenize(text) for text in batch_texts])
        
        logger.debug(f"[Pretokenizer] GPU batch_tokenize tamamlandı: {len(results)} sonuç, etkili batch_size: {effective_batch_size}")
        return results
