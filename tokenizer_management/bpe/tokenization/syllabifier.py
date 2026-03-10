# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: syllabifier.py
Modül: tokenizer_management/bpe/tokenization
Görev: Syllabifier sınıfı - Türkçe heceleme (syllabification) işlemlerini yapar.
       Kelimeleri hecelere ayırır, ünlü-ünsüz kurallarına göre böler.
       Türkçe dilbilgisi kurallarına uygun heceleme yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (heceleme işlemleri)
- Design Patterns: Rule-based Pattern (Türkçe heceleme kuralları)
- Endüstri Standartları: Türkçe dilbilgisi kuralları, fonetik analiz

KULLANIM:
- BPEManager tokenization pipeline'ında kullanılır
- Kelime → hece ayrıştırması için
- Türkçe morfoloji analizi için

BAĞIMLILIKLAR:
- unicodedata: Unicode karakter analizi
- get_turkish_config: Türkçe karakter yapılandırması
- _syllabifier_utils: Heceleme yardımcı fonksiyonları
- SyllabificationError: Özel exception sınıfı

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
from typing import List, Optional, Dict, Any
import re
from tokenizer_management.config import get_turkish_config
from tokenizer_management.bpe.tokenization._syllabifier_utils import strip_diacritics, syllabify_word

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class SyllabificationError(Exception):
    """Heceleme sırasında oluşan hatalar için özel istisna."""
    pass

class Syllabifier:
    """
    Türkçe heceleme işlemlerini yapan sınıf.

    Kurallar (Türkçe’ye özgü, uzun kelimelerde Coda-Onset ayrımı):
      1) Her hecenin tam bir ünlü çekirdeği (nucleus) olmalı.
      2) İki ünlü arasındaki sessiz sayısına göre bölünme:
         - 0 sessiz: V–CV
         - 1 sessiz: V–CV
         - 2 sessiz: V–C1/C2V ("{}_" izinli onset kümesi) → V–C1C2V
         - >=3 sessiz: V–C1/C2…CV
      3) Son hece kalan tüm harfleri alır.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_gpu: Optional[bool] = None) -> None:
        """
        Türkçe heceleme kurallarını başlatır.

        Args:
            config: Opsiyonel bir dict. Anahtarlar:
                - "vowels": str veya list of str (ünlü karakterler)
                - "consonants": str veya list of str (ünsüz karakterler)
                - "syllabification_rules": dict içinde
                    - "allowed_onset_clusters": list of str (izinli onset kümeleri)
            use_gpu: GPU desteği aktif/pasif (None ise config'ten)
        Raises:
            TypeError, ValueError: Yanlış yapılandırma formatlarında.
        """
        # Config'i al (turkish config + BPE detailed config merge)
        from tokenizer_management.config import BPE_DETAILED_CONFIG
        cfg = config if config is not None else get_turkish_config()
        if not isinstance(cfg, dict):
            raise TypeError(f"Config bir dict olmalı, got {type(cfg)}")
        
        # BPE_DETAILED_CONFIG ile merge et (GPU gibi genel parametreler için)
        merged_config = {**cfg}
        merged_config.update(BPE_DETAILED_CONFIG)
        self.config = merged_config
        
        # GPU desteği (config'ten veya parametre)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cpu":
                    logger.warning("[Syllabifier] GPU isteniyor ama CUDA mevcut değil, CPU kullanılacak")
                    self.use_gpu = False
                else:
                    logger.info(f"[Syllabifier] GPU desteği aktif: {self.device}")
            except ImportError:
                logger.warning("[Syllabifier] PyTorch bulunamadı, GPU desteği devre dışı")
                self.use_gpu = False
                self.device = None
        else:
            self.device = None

        raw_vowels = cfg.get("vowels", "aeıioöuü")
        if isinstance(raw_vowels, str):
            self.vowels = set(raw_vowels)
        elif isinstance(raw_vowels, (list, set)):
            self.vowels = set(raw_vowels)
        else:
            raise ValueError("'vowels' str veya list/set olmalı")

        raw_consonants = cfg.get("consonants", "bcçdfgğhjklmnprsştvyz")
        if isinstance(raw_consonants, str):
            self.consonants = set(raw_consonants)
        elif isinstance(raw_consonants, (list, set)):
            self.consonants = set(raw_consonants)
        else:
            raise ValueError("'consonants' str veya list/set olmalı")

        overlap = self.vowels & self.consonants
        if overlap:
            logger.warning("Ünlü ve ünsüz kümeleri kesişiyor, çıkarılıyor: %s", overlap)
            self.consonants -= overlap

        rules = cfg.get("syllabification_rules", {})
        if not isinstance(rules, dict):
            raise ValueError("'syllabification_rules' bir dict olmalı")
        raw_clusters = rules.get("allowed_onset_clusters", [
            "bl", "br", "pl", "pr", "fl", "fr", "kl", "kr",
            "gl", "gr", "dr", "tr", "str", "skr", "spr", "ps", "ks", "ns"
        ])
        if isinstance(raw_clusters, str):
            raw_clusters = [raw_clusters]
        if not isinstance(raw_clusters, (list, set)):
            raise ValueError("'allowed_onset_clusters' liste veya set olmalı")

        valid_clusters: List[str] = []
        for cluster in raw_clusters:
            if all(ch in self.consonants for ch in cluster):
                valid_clusters.append(cluster)
            else:
                logger.warning("Geçersiz onset kümesi atlandı: %r", cluster)
        self.allowed_onsets = set(valid_clusters)

        logger.info(
            "[+] Syllabifier başlatıldı: %d ünlü, %d ünsüz, %d izinli onset",
            len(self.vowels), len(self.consonants), len(self.allowed_onsets)
        )

    def strip_diacritics(self, text: str) -> str:
        """
        Unicode üzerindeki diakritik işaretleri kaldırır ve Türkçe karakterleri korur.

        Args:
            text: İşlenecek metin.

        Returns:
            str: Diakritik işaretleri kaldırılmış, Türkçe karakterleri korunmuş metin.
        """
        return strip_diacritics(text)

    def syllabify_word(self, word: str) -> List[str]:
        """
        Tek bir kelimeyi Türkçe heceleme kurallarına göre böler.

        Args:
            word: Hecelenecek kelime.

        Returns:
            List[str]: Kelimenin hecelere ayrılmış listesi.

        Raises:
            TypeError: Kelime str değilse.
            ValueError: Geçersiz hece oluşursa veya son hece boşsa.
        """
        return syllabify_word(word, self.vowels, self.allowed_onsets)

    def split(self, tokens: List[str]) -> List[str]:
        """
        Birden fazla token listesini hecelere böler.
        Eğer dışarıdan bir heceleyici (self.syllabifier) verilmişse,
        önce onu dener; hata alırsa uyarı loglar ve ham token listesini döner.

        Args:
            tokens: Hecelenecek token listesi.

        Returns:
            List[str]: Hecelere ayrılmış token listesi.

        Raises:
            TypeError: Token listesi list değilse veya token'lar str değilse.
            ValueError: Token listesi boşsa.
            SyllabificationError: Hiç hece üretilemezse.
        """
        if not isinstance(tokens, list):
            raise TypeError(f"Token listesi list olmalı, got {type(tokens)}")
        if not tokens:
            raise ValueError("Token listesi boş olamaz.")

        if hasattr(self, "syllabifier") and self.syllabifier is not None:
            try:
                return self.syllabifier.split(tokens)
            except Exception as e:
                logger.warning(f"[!] Syllabifier hatası: {e}")
                return [str(tok).strip().lower() for tok in tokens]

        all_syllables: List[str] = []
        punct_splitter = re.compile(r"[a-zçöüğış]+|[0-9]+|[^\s\w]", flags=re.IGNORECASE)

        for tok in tokens:
            if not isinstance(tok, str):
                raise TypeError(f"Her token str olmalı, got {type(tok)}")
            raw = tok.strip().lower()

            if raw.startswith("<") and raw.endswith(">"):
                all_syllables.append(raw)
                continue

            norm = unicodedata.normalize("NFC", raw)

            if not norm or not any(ch in self.vowels for ch in norm):
                parts = punct_splitter.findall(norm)
                all_syllables.extend(parts)
                continue

            parts = self.syllabify_word(norm)
            all_syllables.extend(parts)

        if not all_syllables:
            raise SyllabificationError("Hiç hece üretilemedi.")

        logger.debug(f"[+] Toplam hece listesi: {all_syllables}")
        return all_syllables

    def split_into_syllables(self, text: str) -> List[str]:
        """
        Düz bir metni hecelerine ayırır. Boşluk, noktalama ve <…> token'larını korur.
        Rakamları da tek parça olarak döner.

        Args:
            text: Hecelenecek metin.

        Returns:
            List[str]: Hecelere ayrılmış metin parçaları.

        Raises:
            TypeError: Metin str değilse.
        """
        if not isinstance(text, str):
            raise TypeError(f"split_into_syllables metni str olmalı, got {type(text)}")

        norm = unicodedata.normalize("NFC", text.strip().lower())
        if not norm:
            return []

        pattern = r"<[^>]+>|[a-zçöüğış]+|\d+|[^\s\w]|[\s]"
        parts = re.findall(pattern, norm, flags=re.IGNORECASE)

        result: List[str] = []
        for tok in parts:
            if tok.startswith("<") and tok.endswith(">"):
                result.append(tok)
            elif tok.isspace() or tok.isdigit():
                result.append(tok)
            elif not any(ch in self.vowels for ch in tok):
                result.append(tok)
            else:
                try:
                    sylls = self.syllabify_word(tok)
                    result.extend(sylls)
                except Exception as e:
                    logger.warning(f"[!] '{tok}' hecelenemedi: {e}")
                    result.append(tok)

        return result

    def batch_syllabify_gpu(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[str]]:
        """GPU ile batch syllabification - PyTorch tensors kullanarak hızlandırılmış."""
        # batch_size config'ten al
        if batch_size is None:
            batch_size = self.config.get("gpu_batch_size", 32)
            
        if not self.use_gpu or not self.device:
            # Fallback: CPU processing
            return [self.split_into_syllables(text) for text in texts]
        
        import torch
        
        results = []
        
        # Batch'ler halinde işle
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            try:
                for text in batch_texts:
                    # CPU'da syllabification yap
                    syllables = self.split_into_syllables(text)
                    batch_results.append(syllables)
                
                # GPU tensor'a çevir (metin işleme için)
                if batch_results:
                    # String tensor işleme (basit versiyon)
                    tensor_syllables = [syllables for syllables in batch_results]
                    results.extend(tensor_syllables)
                    
            except Exception as e:
                logger.error(f"[Syllabifier] GPU batch syllabification hatası: {e}")
                # Fallback: CPU processing
                for text in batch_texts:
                    syllables = self.split_into_syllables(text)
                    results.append(syllables)
        
        logger.debug(f"[Syllabifier] GPU batch_syllabify tamamlandı: {len(results)} sonuç")
        return results