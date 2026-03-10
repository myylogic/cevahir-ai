# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_manager.py
Modül: tokenizer_management/bpe
Görev: BPEManager sınıfı - BPE (Byte Pair Encoding) tokenization orkestrasyonu.
       Encoder/Decoder/Trainer yaşam döngüsü ve senkronizasyonu, vocab & merges
       atomik okuma/yazma, Türkçe/Unicode dostu normalizasyon işlemlerini yönetir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (BPE tokenization orkestrasyonu),
                     Dependency Inversion (BaseTokenizerManager interface'i),
                     Open/Closed (genişletilebilir tokenization pipeline)
- Design Patterns: Strategy Pattern (farklı tokenization stratejileri),
                  Factory Pattern (Encoder/Decoder/Trainer oluşturma),
                  Singleton Pattern (vocab/merges cache yönetimi)
- Endüstri Standartları: GPT-2/3/4 BPE tokenization, SentencePiece benzeri
                         yaklaşım, Türkçe morfoloji desteği

KULLANIM:
- Tokenization: encode() / decode() - Metin ↔ token dönüşümleri
- BPE Training: train() - Yeni vocab/merges oluşturma
- Vocab Yönetimi: get_vocab() / set_vocab() - Vocab okuma/yazma
- TokenizerCore tarafından kullanılır

BAĞIMLILIKLAR:
- BPEEncoder: Encoding işlemleri
- BPEDecoder: Decoding işlemleri
- BPETrainer: BPE training işlemleri
- Pretokenizer, Syllabifier, Morphology, Postprocessor: Tokenization pipeline

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import shutil
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from .bpe_encoder import BPEEncoder
from .bpe_decoder import BPEDecoder, BPEDecodingError
from .bpe_trainer import BPETrainer, BPETrainingError
from .tokenization.pretokenizer import Pretokenizer
from .tokenization.syllabifier import Syllabifier
from .tokenization.morphology import Morphology
from .tokenization.postprocessor import Postprocessor
from tokenizer_management.config import (
    BPE_CONFIG,
    BPE_DETAILED_CONFIG,
    get_bpe_detailed_config,
)
from tokenizer_management.base_tokenizer_manager import BaseTokenizerManager

# Utils
from tokenizer_management.bpe.bpe_manager_utils import (
    normalize_vocab,
    default_vocab,
    clean_tokens,
    read_json,
    write_json,
    get_valid_ids,
    next_id,
    DEFAULT_SPECIALS,
)

logger = logging.getLogger(__name__)


class BPETokenError(Exception):
    """Genel BPEManager hataları için exception."""
    pass


class BPEManager(BaseTokenizerManager):
    """
    Orkestrasyon katmanı:
    - Encoder/Decoder/Trainer yaşam döngüsü ve senkronizasyonu
    - Vocab & merges’in atomik okunup yazılması
    - Train/Inference moduna uygun, parametrelenebilir tokenizasyon
    - Türkçe/Unicode dostu normalizasyon ve noktalama ayrıştırma
    """
    _instances: Dict[Tuple[str, str], "BPEManager"] = {}

    # ------------------------------ Path çözümleme ------------------------------

    @classmethod
    def _resolve_paths(cls, vocab_file: Optional[str], merges_file: Optional[str]) -> Tuple[str, str]:
        vf = os.path.abspath(vocab_file or os.getenv("VOCAB_FILE") or BPE_CONFIG["vocab_file"])
        mf = os.path.abspath(merges_file or os.getenv("MERGES_FILE") or BPE_CONFIG["merges_file"])
        return vf, mf

    def __new__(cls, vocab=None, vocab_file=None, merges_file=None, use_gpu=None, config=None):
        # config parametresi __init__'e geçirilecek, __new__'de kullanılmaz
        vf, mf = cls._resolve_paths(vocab_file, merges_file)
        # use_gpu None ise key'e None ekle (config'ten alınacak)
        key = (vf, mf, use_gpu)  # GPU flag'i key'e ekle
        if key in cls._instances:
            return cls._instances[key]
        inst = super().__new__(cls)
        cls._instances[key] = inst
        return inst

    def __init__(self, vocab: Optional[Dict[str, Any]] = None, vocab_file: Optional[str] = None, merges_file: Optional[str] = None, use_gpu: Optional[bool] = None, config: Optional[Dict[str, Any]] = None):
        self.vocab_file, self.merges_file = self._resolve_paths(vocab_file, merges_file)
        self._ensure_merges_file()

        # Singleton yeniden init edilmesin
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        
        # ============================================================================
        # CONFIG MERGE: Detaylı config + override mekanizması
        # ============================================================================
        self.config = {**BPE_DETAILED_CONFIG}  # Default: Detaylı config'ten
        if config:
            self.config.update(config)  # Override: Kullanıcı config'i ile
        
        # GPU support (config'ten, parametre sadece override için)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        self.use_gpu = use_gpu

        # Vocab yükle/oluştur
        if vocab is not None:
            self._vocab = normalize_vocab(vocab)
            os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
            write_json(self.vocab_file, self._vocab)
        else:
            self._ensure_vocab_file()

        # Inference vs Training modu kontrolü
        # ÖNEMLİ: Vocab ve merges eğitimde bir kez oluşturulur (train_bpe.py) ve SABİT kalır
        # Inference modunda (cevahir.py) vocab'a ekleme YAPILMAMALI
        vocab_file_exists = os.path.exists(self.vocab_file)
        self.read_only = self.config.get("read_only", False)
        
        # Inference modu: Vocab dosyası mevcutsa VE (read_only=True VEYA vocab parametresi None ise)
        # Bu durumda vocab zaten eğitimde oluşturulmuş, inference modundasak ekleme yapma
        is_inference_mode = vocab_file_exists and (self.read_only or vocab is None)
        
        # Özel tokenları garantile (vocab yüklendikten sonra, bileşenler başlatılmadan önce)
        if is_inference_mode:
            # Inference modunda: Sadece kontrol et, ekleme yapma (vocab sabit)
            self._check_special_tokens_in_vocab()
            logger.debug("[BPEManager] Inference modu: Vocab sabit, ekleme yapılmıyor")
        else:
            # Eğitim modunda: Eksik tokenları ekle (sadece eğitim sırasında)
            self._ensure_special_tokens_in_vocab()
        
        # Bileşenleri başlat ve mevcut merges'i senkronize et
        self._initialize_components()
        # GPU flag'i trainer'a geçir
        if hasattr(self.trainer, 'use_gpu'):
            self.trainer.use_gpu = self.use_gpu
        self.trainer.set_merges(self._read_merges_file())
        self._sync_components()

        # Türkçe taban alfabe + noktalama temelini sağla (UNK fallback'ını azaltır)
        # SADECE eğitim modunda ekleme yap (inference modunda vocab sabit)
        if not is_inference_mode:
            added = self._ensure_base_alphabet_in_vocab()
            if added:
                logger.info("[BPEManager] Eğitim modu: Base alphabet/punct eklendi: +%d token", added)
        else:
            logger.debug("[BPEManager] Inference modu: Base alphabet kontrolü atlandı (vocab sabit, ekleme yapılmıyor)")

        logger.info(
            "BPEManager hazır | vocab_file=%s (%d token) | merges_file=%s",
            self.vocab_file, len(self._vocab), self.merges_file
        )

    # ------------------------------- I/O yardımcıları -------------------------------

    def _read_json(self, path: str) -> Any:
        return read_json(path)

    def _write_json(self, path: str, obj: Any) -> None:
        write_json(path, obj)

    def _ensure_vocab_file(self) -> None:
        try:
            raw = self._read_json(self.vocab_file)
            self._vocab = normalize_vocab(raw)
            logger.info("Vocab diskten yüklendi: %s (%d tokens)", self.vocab_file, len(self._vocab))
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Vocab bulunamadı/bozuk → oluşturuluyor: %s", self.vocab_file)
            self._vocab = default_vocab()
            self._write_json(self.vocab_file, self._vocab)

    def _ensure_merges_file(self) -> None:
        try:
            if not os.path.exists(self.merges_file):
                os.makedirs(os.path.dirname(self.merges_file), exist_ok=True)
                with open(self.merges_file, "w", encoding="utf-8") as f:
                    f.write("")
                if not os.path.exists(self.merges_file):
                    raise BPETokenError(f"Merges dosyası oluşturulamadı: {self.merges_file}")
                logger.info("Yeni merges dosyası oluşturuldu: %s", self.merges_file)
        except Exception as e:
            logger.error("Merges dosyası oluşturulamadı: %s", e)
            raise BPETokenError(f"Merges dosyası oluşturulamadı: {e}")

    def _read_merges_file(self) -> List[Tuple[str, str]]:
        merges: List[Tuple[str, str]] = []
        try:
            with open(self.merges_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        except FileNotFoundError:
            logger.warning("Merges dosyası bulunamadı, boş liste dönülüyor.")
        except Exception as e:
            logger.warning("Merges dosyası okunamadı: %r", e)
        return merges

    def _write_merges_atomic(self, merges: List[Tuple[str, str]]) -> None:
        """
        Merges’i atomik olarak yazar: .tmp → replace.
        Windows’ta arada dosya kilitleri yaşanabildiği için küçük bir retry/backoff uygula,
        son çare olarak shutil.move dene. Başarısız olursa orijinal hatayı yükselt.
        """
        os.makedirs(os.path.dirname(self.merges_file), exist_ok=True)
        tmp = f"{self.merges_file}.tmp"
        # tmp'yi yaz
        with open(tmp, "w", encoding="utf-8") as f:
            for a, b in merges:
                f.write(f"{a} {b}\n")
            f.flush()
            os.fsync(f.fileno())

        last_err: Optional[Exception] = None
        # 6 deneme: 0ms, 25ms, 50ms, 75ms, 100ms, 150ms
        for attempt in range(6):
            try:
                os.replace(tmp, self.merges_file)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.025 * attempt if attempt > 0 else 0.0)
        # son çare
        try:
            shutil.move(tmp, self.merges_file)
            return
        except Exception as e:
            last_err = last_err or e
            # tmp kalmışsa temizle
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise last_err

    # ----------------------------- Bileşen init/sync -----------------------------

    def _initialize_components(self) -> None:
        self.encoder = BPEEncoder(self._vocab, use_gpu=self.use_gpu)
        self.decoder = BPEDecoder(self._vocab, use_gpu=self.use_gpu)
        self.trainer = BPETrainer(self._vocab, use_gpu=self.use_gpu)

        # Pretokenizer: lowercase config'e göre (case-sensitive için False kullan)
        normalize_lowercase = self.config.get("normalize_lowercase", False)
        lowercase_setting = self.config.get("lowercase", False)  # Default False: büyük/küçük ayrımı korunsun
        self.pretokenizer = Pretokenizer(use_gpu=self.use_gpu, lower=lowercase_setting, config=self.config)
        self.syllabifier = Syllabifier(use_gpu=self.use_gpu)
        self.morphology = Morphology()
        self.postprocessor = Postprocessor()

        logger.info("Bileşenler başlatıldı.")

    def _sync_components(self) -> None:
        merges = self.trainer.get_merges()
        self.encoder.set_vocab(self._vocab)
        self.encoder.set_merges(merges)
        self.decoder.set_vocab(self._vocab)
        self.decoder.set_merges(merges)
        logger.debug("Encoder/Decoder/Trainer senkronize edildi. merges=%d", len(merges))

    # ------------------ Normalizasyon / Tag temizliği / Tokenizasyon ------------------

    _TAG_RE = re.compile(r"\[(SYSTEM|USER|ASSISTANT|INTERNAL(?:\s+\w+)?)\]", re.IGNORECASE)

    @staticmethod
    def _normalize_text(text: str, *, lowercase: bool = False) -> str:
        """
        Hafif-normalizasyon: NFC, çoklu boşluk → tek, opsiyonel lowercase (Türkçe duyarlı).
        Noktalama çevresi boşluklarını kaba şekilde toparlar (ince ayar decode’da).
        """
        s = unicodedata.normalize("NFC", text or "")
        if lowercase:
            # Türkçe I/İ kaba-uyumlu küçük harf
            s = s.replace("I", "ı").replace("İ", "i").lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _strip_or_map_tags(cls, text: str, *, map_to_special: bool = False) -> str:
        if map_to_special:
            return cls._TAG_RE.sub(lambda m: f"<{m.group(1).upper()[:3]}>", text)
        return cls._TAG_RE.sub("", text)

    @staticmethod
    def _is_punct_char(ch: str) -> bool:
        """Unicode kategoriye göre noktalama/simge kontrolü."""
        if not ch:
            return False
        cat = unicodedata.category(ch)
        # P* = punctuation, S* = symbol
        return cat.startswith("P") or cat.startswith("S")

    @classmethod
    def _split_punct(cls, token: str) -> tuple[list[str], str, list[str]]:
        """
        'evet,' -> ([], 'evet', [','])
        "'sonuç'," -> (["'"], "sonuç", [","])
        "(örnek)" -> (["("], "örnek", [")"])
        Tamamen noktalama ise core='' döner: "..." -> ([".",".","."], "", [])
        """
        s = token or ""
        if not s:
            return [], "", []

        # leading punctuation
        start = 0
        lead: list[str] = []
        while start < len(s) and cls._is_punct_char(s[start]):
            lead.append(s[start])
            start += 1

        # trailing punctuation
        end = len(s) - 1
        trail_chars: list[str] = []
        while end >= start and cls._is_punct_char(s[end]):
            trail_chars.append(s[end])
            end -= 1
        trail = list(reversed(trail_chars))

        core = s[start:end + 1] if end >= start else ""
        return lead, core, trail

    def _tokenize_with_punct(self, text: str, *, include_whole_words: bool, include_syllables: bool, include_sep: bool) -> list[str]:
        """
        split() yerine kelime gövdesi + noktalama ayrıştırması.
        <SEP> yalnız kelimeler arası konur (noktalama, bir önceki kelimeye yapışık kalır).
        Tag’leri (ör. [SYSTEM]) silent-drop ederiz; istersen map_to_special=True yapabilirsin.
        """
        # 0) Tag temizliği + hafif normalizasyon (eğitimde nasılsa ona uy)
        cleaned = self._strip_or_map_tags(text, map_to_special=False)
        lowercase = self.config.get("normalize_lowercase", False)
        cleaned = self._normalize_text(cleaned, lowercase=lowercase)

        # Boşlukları koruyarak tokenize et
        pretokenized = self.pretokenizer.tokenize(cleaned or "")
        out: list[str] = []
        
        for i, raw in enumerate(pretokenized):
            # Boşluk tokenlarını koru
            if raw == ' ':
                out.append(' ')
                continue
                
            lead, core, trail = self._split_punct(raw)

            # baştaki noktalama
            out.extend(lead)

            # kelime gövdesi
            if core:
                if include_whole_words:
                    out.append(core + "</w>")
                if include_syllables:
                    heceler = self.syllabifier.syllabify_word(core) or [core]
                    out.extend([h for h in heceler if h and h.strip()])
                    
                    # Morfoloji analizi (config'ten kontrol edilebilir - default: False)
                    # NOT: Morfoloji analizi çok fazla token ekliyor, over-segmentation'a neden oluyor
                    include_morphology = self.config.get("include_morphology", False)
                    if include_morphology:
                        for hece in heceler:
                            if hece and hece.strip() and hece.isalpha():
                                try:
                                    morphemes = self.morphology.analyze([hece])
                                    out.extend([m for m in morphemes if m and m.strip()])
                                except Exception as e:
                                    # Morfoloji analizi başarısız olursa heceyi olduğu gibi ekle
                                    logger.debug(f"Morfoloji analizi hatası '{hece}': {e}")
                                    out.append(hece)

            # sondaki noktalama
            out.extend(trail)

            # ENDÜSTRİ STANDARDI: <SEP> token minimal kullanım (GPT/Claude/Gemini gibi modeller)
            # <SEP> token sadece özel durumlar için kullanılmalı (örn: çok uzun metinler, özel formatlar)
            # Normal tokenization'da <SEP> kullanımı over-segmentation'a neden olur
            # NOT: include_sep=False ise <SEP> eklenmez (default: False - endüstri standardı)
            if include_sep and i < len(pretokenized) - 1 and core:
                # Özel durum kontrolü: Sadece gerçekten gerekli durumlarda <SEP> ekle
                # Örnek: Çok uzun metinler, özel formatlar, vb.
                # Normal tokenization'da <SEP> eklenmez
                out.append("<SEP>")

        return [t for t in out if t is not None and (t.strip() or t == ' ')]

    # ------------------------------------ Public API ------------------------------------

    def encode(
        self,
        text: str,
        mode: str = "inference",
        *,
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        add_special_tokens: Optional[bool] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Metni tokenize edip token ID'lerine çevirir (yan etkisiz).
        
        Args:
            text: Tokenize edilecek metin
            mode: "train" veya "inference"
            include_whole_words: 'kelime</w>' ekle
            include_syllables: hecelemeden alt tokenlar
            include_sep: kelimeler arası <SEP>
            add_special_tokens: BOS/EOS ekleme kontrolü
                - None (default): Mode'a göre karar verir
                  * inference: True (BOS ekle, model EOS üretir)
                  * train: False (DataLoader'da eklenecek)
                - True: Her zaman BOS/EOS ekle
                - False: Hiç ekleme
        
        ENDÜSTRI STANDARDI (GPT-2/3/4):
            - Training: BOS/EOS DataLoader'da eklenir (autoregressive format için)
            - Inference: BOS eklenir, model EOS üretir
        """
        if not isinstance(text, str):
            raise TypeError("Encode: Girdi metni str olmalı")

        # ✅ Boş text kontrolü - encode_sequence hatası önlemek için
        text_stripped = text.strip()
        if not text_stripped:
            logger.debug("[BPEManager] encode: Boş string, boş liste döndürülüyor")
            return [], []

        # ✅ KRİTİK DÜZELTME: Special token kontrolü
        # Eğer text tam olarak bir special token ise, direkt special token ID'sini döndür
        # Bu, special token'ların string olarak encode edilmesini önler
        if text_stripped in DEFAULT_SPECIALS:
            # Vocab'tan special token ID'sini al
            special_token_info = self._vocab.get(text_stripped)
            if special_token_info:
                special_id = special_token_info.get("id") if isinstance(special_token_info, dict) else special_token_info
                if special_id is not None:
                    logger.debug(f"[BPEManager] Special token tespit edildi: '{text_stripped}' → ID {special_id}")
                    return [text_stripped], [int(special_id)]
            # Fallback: DEFAULT_SPECIALS kullan (vocab'ta yoksa)
            special_id = DEFAULT_SPECIALS.get(text_stripped)
            if special_id is not None:
                logger.debug(f"[BPEManager] Special token (fallback): '{text_stripped}' → ID {special_id}")
                return [text_stripped], [special_id]

        # Parametreleri config'ten al (None ise)
        if include_whole_words is None:
            include_whole_words = self.config.get("include_whole_words", True)
        if include_syllables is None:
            include_syllables = self.config.get("include_syllables", True)
        if include_sep is None:
            include_sep = self.config.get("include_sep", True)
        
        # ✅ ENDÜSTRI STANDARDI: add_special_tokens kontrolü
        if add_special_tokens is None:
            # Default davranış: Mode'a göre
            # Inference: BOS ekle (model EOS üretir)
            # Train: Ekleme (DataLoader'da eklenecek - autoregressive format için)
            add_special_tokens = (mode.lower() == "inference")
        
        # Config'ten override (eğer varsa)
        if "add_special_tokens" in self.config:
            add_special_tokens = self.config.get("add_special_tokens", add_special_tokens)

        # ✅ Token listesi oluştur
        tokens: List[str] = []
        
        # BOS ekle (eğer gerekirse)
        if add_special_tokens:
            tokens.append("<BOS>")
        
        # Tokenize et
        tokenized = self._tokenize_with_punct(
            text,
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
        )
        
        # ✅ Boş tokenize sonucu kontrolü
        if not tokenized:
            # Eğer add_special_tokens=False ise ve tokenize sonucu boşsa, boş liste döndür
            if not add_special_tokens:
                logger.debug("[BPEManager] encode: Tokenize sonucu boş ve add_special_tokens=False, boş liste döndürülüyor")
                return [], []
            # Eğer add_special_tokens=True ise, sadece BOS var, onu da kontrol et
            if not tokens:
                logger.debug("[BPEManager] encode: Tokenize sonucu boş ve tokens listesi boş, boş liste döndürülüyor")
                return [], []
        
        tokens.extend(tokenized)
        
        # EOS ekle (eğer gerekirse)
        # NOT: Inference'ta genellikle EOS eklenmez (model üretir)
        # Ama config'ten kontrol edilebilir (geriye dönük uyumluluk için)
        if add_special_tokens and self.config.get("add_eos_on_encode", False):
            tokens.append("<EOS>")
        # Boşluk tokenlarını koru, sadece gerçekten boş olanları sil
        tokens = [t for t in tokens if t is not None and (t.strip() or t == ' ')]
        
        # ✅ Boş token listesi kontrolü - encode_sequence hatası önlemek için
        if not tokens:
            logger.debug("[BPEManager] encode: Token listesi boş, boş liste döndürülüyor")
            return [], []

        # ============================================================================
        # VOCAB GÜNCELLEME STRATEJİSİ (Config'ten kontrol edilir)
        # ============================================================================
        # Sabit vocab stratejisi: auto_vocab_update=False (config'te)
        # Dinamik vocab stratejisi: auto_vocab_update=True (nadiren kullanılır)
        # ============================================================================
        if mode == "train" and self.config.get("auto_vocab_update", False) and (include_whole_words or include_syllables):
            # Vocab limit kontrolü
            current_vocab_size = len(self._vocab)
            max_vocab_size = self.config.get("max_vocab_size", 60000)
            
            if current_vocab_size >= max_vocab_size:
                logger.warning(f"[BPEManager] Vocab limit aşıldı: {current_vocab_size}/{max_vocab_size}")
            else:
                cands = [
                    t for t in tokens
                    if t not in ("<BOS>", "<EOS>", "<SEP>")
                    and not (t.startswith("<") and t.endswith(">"))  # <...> tag'leri hariç
                ]
                try:
                    added = self.auto_update_vocab(cands)
                    if added:
                        new_size = len(self._vocab)
                        logger.info(f"[BPEManager] Vocab genişledi: +{added} token (toplam: {new_size})")
                        
                        # Alert mekanizması (config'ten interval)
                        vocab_alert_interval = self.config.get("vocab_growth_alert", 1000)
                        if new_size % vocab_alert_interval == 0:
                            logger.warning(f"[BPEManager] UYARI: Vocab büyüyor! Şu an: {new_size}")
                except Exception as e:
                    logger.error(f"[BPEManager] auto_update_vocab hatası: {e}")

        # Yalnız heceleme kullanılıyorsa (whole-word yoksa) heceleri vocab'a ekle ki decode reconstruct edebilsin
        if include_syllables and not include_whole_words:
            syll_cands = [
                t for t in tokens
                if t not in ("<BOS>", "<EOS>", "<SEP>") and not (t.startswith("<") and t.endswith(">"))
            ]
            try:
                self.auto_update_vocab(syll_cands)
            except Exception as e:
                logger.debug("[BPEManager] encode sırasında auto_update_vocab başarısız: %r", e)

        # ID’lere çevir
        try:
            token_ids = self.encoder.encode_sequence(tokens)
        except Exception as e:
            raise BPETokenError(f"Encode sırasında hata: {e}") from e

        # ============================================================================
        # OOV (Out-of-Vocabulary) KELİMELER İÇİN SYLLABLE FALLBACK MEKANİZMASI
        # ============================================================================
        # Eğer include_syllables=False ise ve use_syllables_for_oov=True ise,
        # UNK token görürsek o kelimeler için syllable fallback yap
        # Bu sayede hem over-segmentation azalır hem de OOV kelimeler işlenebilir
        # ============================================================================
        use_syllables_for_oov = self.config.get("use_syllables_for_oov", True)
        unk_id = self.encoder._unk_id
        
        if not include_syllables and use_syllables_for_oov and unk_id is not None:
            # UNK token var mı kontrol et
            # NOT: encode_sequence bir token'ı birden fazla ID'ye çevirebilir (BPE merge'ler)
            # Bu yüzden token-ID eşleşmesini takip etmeliyiz
            unk_count = sum(1 for tid in token_ids if tid == unk_id)
            
            if unk_count > 0:
                # Hangi token'ların UNK ürettiğini bul
                # Her token'ı tek tek encode edip UNK içerip içermediğini kontrol et
                unk_token_indices = []
                current_id_index = 0
                
                for i, token in enumerate(tokens):
                    # Bu token'ı encode et
                    token_id_list = self.encoder._encode_token_to_ids(token)
                    # Bu token'ın ID'leri arasında UNK var mı?
                    if unk_id in token_id_list:
                        unk_token_indices.append(i)
                    current_id_index += len(token_id_list)
                
                if unk_token_indices:
                    # UNK olan token'ları tespit et (whole word token'ları, </w> ile bitenler)
                    # Bu token'lar için syllable fallback yap
                    fallback_tokens = []
                    for i, token in enumerate(tokens):
                        if i in unk_token_indices:
                            # UNK token bulundu, bu bir whole word token olabilir (örn: "kelime</w>")
                            # Veya başka bir token olabilir
                            # Eğer </w> ile bitiyorsa, kelimeyi çıkar ve hecelere ayır
                            if token.endswith("</w>"):
                                word = token[:-4]  # "</w>" kısmını çıkar
                                # Kelimeyi hecelere ayır
                                heceler = self.syllabifier.syllabify_word(word) or [word]
                                # Whole word'ü koru, heceleri de ekle
                                fallback_tokens.append(token)  # Orijinal whole word token'ı koru
                                fallback_tokens.extend([h for h in heceler if h and h.strip()])
                            else:
                                # </w> ile bitmiyorsa, token'ı olduğu gibi ekle
                                fallback_tokens.append(token)
                        else:
                            # UNK değilse, olduğu gibi ekle
                            fallback_tokens.append(token)
                    
                    # Fallback token'ları ile tekrar encode et
                    try:
                        fallback_token_ids = self.encoder.encode_sequence(fallback_tokens)
                        
                        # Fallback sonrası UNK sayısını kontrol et
                        fallback_unk_count = sum(1 for tid in fallback_token_ids if tid == unk_id)
                        original_unk_count = unk_count
                        
                        if fallback_unk_count < original_unk_count:
                            # Fallback başarılı, yeni token'ları kullan
                            logger.info(
                                f"[BPEManager] OOV syllable fallback: {original_unk_count} → {fallback_unk_count} UNK token "
                                f"({original_unk_count - fallback_unk_count} kelime işlendi)"
                            )
                            tokens = fallback_tokens
                            token_ids = fallback_token_ids
                        else:
                            # Fallback başarısız, orijinal token'ları kullan
                            logger.debug(
                                f"[BPEManager] OOV syllable fallback başarısız: "
                                f"{original_unk_count} → {fallback_unk_count} UNK token"
                            )
                    except Exception as e:
                        # Fallback sırasında hata olursa, orijinal token'ları kullan
                        logger.warning(f"[BPEManager] OOV syllable fallback hatası: {e}, orijinal token'lar kullanılıyor")
                
        # Üyelik kontrollü doğrulama
        valid_ids = get_valid_ids(self._vocab)
        for idx, tid in enumerate(token_ids):
            if tid not in valid_ids:
                raise BPETokenError(f"Geçersiz token id: token='{tokens[idx]}', id={tid}")

        # HATA #2 DÜZELTME: Special token'lar train ve inference modlarında aynı şekilde korunmalı
        # Autoregressive eğitim için special token'lar (<BOS>, <EOS>, <SEP>) gerekli
        # Önceki kod train modunda special token'ları kaldırıyordu, bu autoregressive eğitimi bozuyordu
        # Config'ten kontrol edilebilir (default: True = special token'ları koru)
        keep_specials_in_train = self.config.get("keep_specials_in_train", True)
        
        if mode == "train" and not keep_specials_in_train:
            # Eski davranış (önerilmez, autoregressive eğitimi bozar)
            logger.warning("[BPEManager] UYARI: Train modunda special token'lar kaldırılıyor! "
                          "Bu autoregressive eğitimi bozabilir. keep_specials_in_train=True önerilir.")
            mask_idx = [i for i, t in enumerate(tokens) if t not in ("<BOS>", "<EOS>", "<SEP>")]
            out_tokens = [tokens[i] for i in mask_idx]
            out_ids = [token_ids[i] for i in mask_idx]
            return out_tokens, out_ids

        # Default: Special token'ları koru (train ve inference modlarında aynı)
        return tokens, token_ids

    def preview_tokens(self, text: str, *, include_whole_words: Optional[bool] = None, include_syllables: Optional[bool] = None, include_sep: Optional[bool] = None) -> list[str]:
        # Config'ten default değerleri al
        if include_whole_words is None:
            include_whole_words = self.config.get("include_whole_words", True)
        if include_syllables is None:
            include_syllables = self.config.get("include_syllables", True)
        if include_sep is None:
            include_sep = self.config.get("include_sep", True)
            
        return self._tokenize_with_punct(
            text,
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
        )

    def ensure_tokens_in_vocab(self, texts: list[str], *, include_whole_words: Optional[bool] = None, include_syllables: Optional[bool] = None, include_sep: Optional[bool] = None) -> int:
        # Config'ten default değerleri al
        if include_whole_words is None:
            include_whole_words = self.config.get("include_whole_words", True)
        if include_syllables is None:
            include_syllables = self.config.get("include_syllables", True)
        if include_sep is None:
            include_sep = self.config.get("include_sep", True)
            
        candidates = set()
        total_texts = len(texts)
        logger.info(f"[BPEManager] Vocab genişletme: {total_texts:,} metin işlenecek...")
        
        # GPU batch processing kullan
        if self.use_gpu and torch.cuda.is_available():
            logger.info(f"[BPEManager] GPU batch processing ile vocab genişletme başlıyor...")
            candidates = self._ensure_tokens_in_vocab_gpu(
                texts, 
                include_whole_words=include_whole_words,
                include_syllables=include_syllables,
                include_sep=include_sep
            )
        else:
            # CPU sequential processing (eski yöntem)
            progress_log_interval = self.config.get("progress_log_interval", 5000)
            for i, t in enumerate(texts):
                if i % progress_log_interval == 0:  # Config'ten interval
                    logger.info(f"[BPEManager] İşlenen: {i:,}/{total_texts:,} ({(i/total_texts)*100:.1f}%)")
                
                candidates.update(self.preview_tokens(
                    t,
                    include_whole_words=include_whole_words,
                    include_syllables=include_syllables,
                    include_sep=include_sep
                ))

        specials = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}

        def is_taglike(s: str) -> bool:
            # <BOS>, <EOS>, <PAD>, <UNK> gibi *tamamen* köşeli parantezli olanlar
            return s.startswith("<") and s.endswith(">")

        # <SEP>’i özel olarak tut; onun dışındaki *tam tag*’leri ele.
        filtered = [
            t for t in candidates
            if (t == "<SEP>") or (t not in specials and not is_taglike(t))
        ]
        added = self.auto_update_vocab(filtered)   # kaydedip sync ediyor
        return added

    def decode(
        self,
        token_ids: List[int],
        method: str = "bpe",
        *,
        remove_specials: Optional[bool] = None,
        remove_tags: Optional[bool] = None,
        sep_token: str = "<SEP>",
        collapse_spaces: Optional[bool] = None,
        lowercase: Optional[bool] = None,
        prefer: Optional[str] = None,   # "word" | "syllable" | "auto" | None
    ) -> str:
        """
        ID listesini metne çevirir.

        method:
            - 'bpe'  : self.decoder.decode(...)  (önerilen; parça-bazlı prefer burada uygulanır)
            - 'raw'  : reverse_vocab ile kaba dönüşüm (diagnostic)
        prefer:
            - "word" / "syllable" / "auto" : doğrudan BPEDecoder.decode'a iletilir
            - None: tercih filtresi uygulanmaz (decoder varsayılanı kullanır)
        """
        # Parametreleri config'ten al (None ise)
        if remove_specials is None:
            remove_specials = self.config.get("remove_specials", True)
        if remove_tags is None:
            remove_tags = self.config.get("remove_tags", True)
        if collapse_spaces is None:
            collapse_spaces = self.config.get("collapse_spaces", True)
        if lowercase is None:
            lowercase = self.config.get("lowercase", False)
        if prefer is None:
            prefer = self.config.get("prefer_mode", None)
        
        # Girdi doğrulama (bool check yumuşatıldı - KRİTİK HATA #3 düzeltmesi!)
        if not isinstance(token_ids, list):
            raise TypeError("Decode: token_ids bir liste olmalı.")
        # Boş liste için boş string döndür (graceful handling)
        if not token_ids:
            logger.debug("[BPEManager] decode: boş liste, boş string döndürülüyor")
            return ""
        # Bool check kaldırıldı çünkü Python'da bool zaten int'tir
        if method not in ("raw", "bpe"):
            raise BPEDecodingError("Decode: method 'raw' veya 'bpe' olmalı.")
        
        # Filter out invalid token IDs (out of vocab range) gracefully
        vocab_size = len(self.decoder.reverse_vocab) if hasattr(self.decoder, 'reverse_vocab') else 0
        if vocab_size > 0:
            valid_ids = [tid for tid in token_ids if 0 <= tid < vocab_size]
            if not valid_ids:
                logger.debug("[BPEManager] decode: tüm token ID'ler geçersiz, boş string döndürülüyor")
                return ""
            if len(valid_ids) < len(token_ids):
                invalid_count = len(token_ids) - len(valid_ids)
                logger.warning(f"[BPEManager] decode: {invalid_count} geçersiz token ID filtrelendi")
            token_ids = valid_ids

        # ---------- 'raw' yolu ----------
        if method == "raw":
            rev = self.decoder.reverse_vocab  # id -> token
            specials = {"<BOS>", "<EOS>", "<PAD>", "<UNK>"}
            # Special'ları (SEP hariç) düş, SEP'i boşluk yap
            toks = [rev.get(i, "<UNK>") for i in token_ids if rev.get(i, "<UNK>") not in specials]
            out_parts: List[str] = []
            for t in toks:
                if t == sep_token:
                    out_parts.append(" ")
                elif t.endswith("</w>"):
                    # </w> token'ını kaldır ve kelimeyi ekle
                    word = t[:-4]  # </w> kısmını çıkar
                    out_parts.append(word)
                else:
                    out_parts.append(t)
            text = "".join(out_parts)
            if collapse_spaces:
                text = re.sub(r"\s+", " ", text).strip()
            if lowercase:
                text = text.lower()
            text = unicodedata.normalize("NFC", text)
            return text

        # ---------- 'bpe' yolu ----------
        text = self.decoder.decode(
            token_ids,
            remove_specials=remove_specials,
            remove_tags=remove_tags,
            sep_token=sep_token,
            collapse_spaces=collapse_spaces,
            lowercase=lowercase,
            prefer=prefer,
        )

        if not isinstance(text, str):
            logger.debug("[BPEManager] Decoder non-string üretti (%r). Boş stringe düşülüyor.", type(text))
            text = ""

        text = unicodedata.normalize("NFC", text)
        return text

    def tokenize(self, text: Union[str, List[str], dict]) -> List[str]:
        """
        Pretokenizer → Syllabifier → (opsiyonel) clean_tokens
        """
        toks = self.pretokenizer.tokenize(text)
        sylls: List[str] = []
        for tok in toks:
            sylls.extend(self.syllabifier.split([tok]))
        # sylls = clean_tokens(sylls)  # gerekirse
        return sylls

    # ---------------------------------------- Train ----------------------------------------

    
    def _process_corpus_in_chunks(self, corpus, chunk_size=None):
        """Corpus'u chunk'lara böl ve işle"""
        if chunk_size is None:
            chunk_size = self.config.get("chunk_size", 2000)
        for i in range(0, len(corpus), chunk_size):
            chunk = corpus[i:i + chunk_size]
            yield chunk

    def _stream_corpus_processing(self, corpus, chunk_size=None, include_whole_words=None, include_syllables=None, include_sep=None):
        """Streaming corpus processing - büyük veri setleri için"""
        # Config'ten default değerleri al
        if chunk_size is None:
            chunk_size = self.config.get("chunk_size", 2000)
        if include_whole_words is None:
            include_whole_words = self.config.get("include_whole_words", True)
        if include_syllables is None:
            include_syllables = self.config.get("include_syllables", True)
        if include_sep is None:
            include_sep = self.config.get("include_sep", True)
            
        total_items = len(corpus)
        processed_items = 0
        
        for i in range(0, total_items, chunk_size):
            chunk = corpus[i:i + chunk_size]
            chunk_results = []
            
            for item in chunk:
                if isinstance(item, str):
                    toks = self._tokenize_with_punct(
                        item,
                        include_whole_words=include_whole_words,
                        include_syllables=include_syllables,
                        include_sep=include_sep,
                    )
                elif isinstance(item, list) and all(isinstance(t, str) for t in item):
                    toks = [t for t in item]
                else:
                    continue
                
                cleaned = [t for t in toks if t not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>") and t and t.strip()]
                if cleaned:
                    chunk_results.append(cleaned)
            
            processed_items += len(chunk)
            progress_pct = (processed_items / total_items) * 100
            logger.info(f"[BPEManager] Streaming progress: {progress_pct:.1f}% ({processed_items}/{total_items})")
            
            yield chunk_results
    
    def _tokenize_chunk_optimized(self, chunk):
        """Chunk'ı optimize edilmiş şekilde tokenize et"""
        # Memory-efficient tokenization
        return self._tokenize_with_punct(chunk)

    def train(
        self,
        corpus: List[Union[str, List[str]]],
        *,
        target_merges: Optional[int] = None,
        max_iter: Optional[int] = None,
        min_frequency: Optional[int] = None,
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        append_eos: Optional[bool] = None,
        protect_specials: Optional[bool] = None,
    ) -> None:
        """
        BPE modelini eğitir ve sonuçları diske yazar.
        Varsayılan eğitim tokenizasyonu yalın tutulur (yalnız 'kelime</w>'); heceleme isteğe bağlıdır.
        """
        import time
        import psutil
        
        if not corpus:
            raise ValueError("Train: corpus boş olamaz.")

        # Parametreleri config'ten al (None ise)
        if max_iter is None:
            max_iter = self.config.get("max_iter", 60000)
        if min_frequency is None:
            min_frequency = self.config.get("min_frequency", 2)
        if include_whole_words is None:
            include_whole_words = self.config.get("include_whole_words", True)
        if include_syllables is None:
            include_syllables = self.config.get("include_syllables", True)
        if include_sep is None:
            include_sep = self.config.get("include_sep", True)
        if append_eos is None:
            append_eos = self.config.get("append_eos", True)
        if protect_specials is None:
            protect_specials = self.config.get("protect_specials", True)

        # Başlangıç zamanı ve bellek izleme
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        merges_to_do = (
            target_merges
            if (isinstance(target_merges, int) and target_merges > 0)
            else self.config.get("merge_operations", 60000)
        )

        logger.info(f"[BPEManager] BPE Eğitimi başlıyor...")
        logger.info(f"[BPEManager] Corpus boyutu: {len(corpus):,} cümle")
        logger.info(f"[BPEManager] Hedef merge sayısı: {merges_to_do:,}")
        logger.info(f"[BPEManager] Maksimum iterasyon: {max_iter:,}")
        logger.info(f"[BPEManager] Minimum frekans: {min_frequency}")
        logger.info(f"[BPEManager] Başlangıç bellek kullanımı: {initial_memory:.2f} MB")

        # 0) Disktekileri yükle ve bileşenleri tazele
        logger.info("[BPEManager] Vocab ve merges dosyaları yükleniyor...")
        self._ensure_vocab_file()
        self._ensure_merges_file()
        self._initialize_components()
        self.trainer.set_merges(self._read_merges_file())
        
        after_init_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"[BPEManager] Bileşenler yüklendi - Bellek: {after_init_memory:.2f} MB (+{after_init_memory-initial_memory:.2f} MB)")

        # 1) Korpusu trainer girişi için token listelerine çevir
        logger.info("[BPEManager] Corpus tokenizasyonu başlıyor...")
        trainer_input: List[List[str]] = []
        all_tokens: Set[str] = set()
        
        # ADAPTIVE: Büyük corpus'lar için streaming processing kullan
        if len(corpus) > 10000:  # 10K+ item için streaming
            logger.info("[BPEManager] Streaming corpus processing kullanılıyor...")
            chunk_size = self.config.get("chunk_size", 2000)
            for chunk_results in self._stream_corpus_processing(corpus, chunk_size=chunk_size, include_whole_words=include_whole_words, include_syllables=include_syllables, include_sep=include_sep):
                trainer_input.extend(chunk_results)
                for result in chunk_results:
                    all_tokens.update(result)
        else:
            # Normal processing
            # Progress tracking
            total_items = len(corpus)
            processed_items = 0
            last_log_time = time.time()
            log_interval = self.config.get("log_interval", 30.0)  # Config'ten (saniye)

            for i, item in enumerate(corpus):
                if isinstance(item, str):
                    toks = self._tokenize_with_punct(
                        item,
                        include_whole_words=include_whole_words,
                        include_syllables=include_syllables,
                        include_sep=include_sep,
                    )
                elif isinstance(item, list) and all(isinstance(t, str) for t in item):
                    toks = [t for t in item]  # dışarıdan hazırlanmış tokenlar
                else:
                    raise TypeError(f"Corpus elemanı str veya List[str] olmalı, bulundu: {item!r}")

                cleaned = [t for t in toks if t not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>") and t and t.strip()]
                if cleaned:
                    trainer_input.append(cleaned)
                    all_tokens.update(cleaned)
                
                processed_items += 1
            
            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                progress_pct = (processed_items / total_items) * 100
                current_memory = process.memory_info().rss / 1024 / 1024
                elapsed = current_time - start_time
                
                logger.info(f"[BPEManager] Tokenizasyon: {processed_items:,}/{total_items:,} "
                           f"({progress_pct:.1f}%) | "
                           f"Unique tokens: {len(all_tokens):,} | "
                           f"Bellek: {current_memory:.1f}MB | "
                           f"Süre: {elapsed:.1f}s")
                last_log_time = current_time

        tokenization_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"[BPEManager] Tokenizasyon tamamlandı!")
        logger.info(f"[BPEManager] İşlenen cümle: {len(trainer_input):,}")
        logger.info(f"[BPEManager] Unique token: {len(all_tokens):,}")
        logger.info(f"[BPEManager] Tokenizasyon sonrası bellek: {tokenization_memory:.2f} MB (+{tokenization_memory-initial_memory:.2f} MB)")

        # 2) Token frekanslarını hesapla
        logger.info("[BPEManager] Token frekansları hesaplanıyor...")
        token_frequencies = self._calculate_token_frequencies(trainer_input)
        logger.info(f"[BPEManager] Token frekansları hesaplandı: {len(token_frequencies):,} unique token")

        # 3) Vocab size kontrolü ve token seçimi
        # NOT: GPT ve diğer BPE sistemleri genellikle karakterlerle başlar (~256 token),
        #      bizim sistemimiz kelime bazlı başlıyor. Bu yüzden merge için yer bırakmak kritik!
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        min_vocab_size = self.config.get("min_vocab_size", 30000)
        vocab_size_buffer = self.config.get("vocab_size_buffer", 5000)
        target_merges = self.config.get("target_merges", 60000)
        
        # GPT BENZERİ YAKLAŞIM: Karakter bazlı başlangıç + merge'lerle vocab inşa
        # GPT karakter bazlı başlar (~256 token, %0.5), merge'lerle vocab'ı inşa eder
        # Bizim sistem: initial_vocab_ratio ile başlangıç vocab belirlenir, merge'lerle genişletilir
        initial_vocab_ratio = self.config.get("initial_vocab_ratio", 0.15)
        
        # Oran bazlı yaklaşım (GPT benzeri - düşük başlangıç, agresif merge)
        initial_vocab_size = int(max_vocab_size * initial_vocab_ratio)
        expected_merges = max_vocab_size - initial_vocab_size - vocab_size_buffer
        
        # Minimum limit kontrolü (GPT benzeri yaklaşım için minimal başlangıç)
        # Base alphabet karakterleri + special tokenlar minimum olmalı
        base_alphabet_size = 256  # Karakterler + noktalama + rakamlar (GPT benzeri)
        min_required_vocab = base_alphabet_size + len(DEFAULT_SPECIALS)  # ~260 token minimum
        
        # Initial vocab size minimum kontrolü (minimal başlangıç - merge için maksimum yer bırak)
        # min_vocab_size kontrolünü kaldırdık (çok yüksek oluyordu)
        # Sadece base alphabet + special tokens minimum olmalı
        initial_vocab_size = max(initial_vocab_size, min_required_vocab)  # Sadece base alphabet minimum
        expected_merges = max(0, max_vocab_size - initial_vocab_size - vocab_size_buffer)  # Negatif olmamalı
        
        logger.info(f"[BPEManager] GPT BENZERİ YAKLAŞIM: initial_vocab_ratio={initial_vocab_ratio} (%{int(initial_vocab_ratio*100)})")
        logger.info(f"[BPEManager] Karakter bazlı başlangıç + merge'lerle vocab inşa stratejisi")
        
        logger.info(f"[BPEManager] Vocab hesaplama: max={max_vocab_size:,}, expected_merges={expected_merges:,}, buffer={vocab_size_buffer:,}")
        # Vocab'ta yer varken UNK üretmeyelim: max_vocab_size'a sığan tüm token'ları ekle
        max_initial_slots = max(initial_vocab_size, max_vocab_size - vocab_size_buffer)
        if len(all_tokens) <= max_vocab_size:
            selected_tokens = sorted(all_tokens)
            logger.info(f"[BPEManager] Tüm token'lar vocab'e eklenecek ({len(selected_tokens):,} token) — vocab'ta yer var, UNK yok.")
        else:
            logger.warning(f"[BPEManager] Unique token sayısı ({len(all_tokens):,}) max_vocab_size'tan ({max_vocab_size:,}) büyük!")
            logger.info(f"[BPEManager] En sık kullanılan {max_initial_slots:,} token seçiliyor...")
            selected_tokens = self._select_top_tokens(all_tokens, token_frequencies, max_initial_slots)
            excluded_tokens = all_tokens - set(selected_tokens)
            logger.info(f"[BPEManager] Vocab'e eklenen: {len(selected_tokens):,} token")
            logger.info(f"[BPEManager] Vocab'e eklenmeyen: {len(excluded_tokens):,} token (UNK olarak işlenecek)")
            if excluded_tokens:
                sample_excluded = list(excluded_tokens)[:10]
                logger.info(f"[BPEManager] Eklenmeyen token örnekleri: {sample_excluded}")

        # 4) Base alphabet/punct karakterlerini garanti altına al (char fallback için kritik!)
        # Bu, karakterlerin vocab'te olmasını garanti eder (UNK oranını düşürür)
        # ÖNEMLİ: Bu, trainer.update_vocab()'dan ÖNCE yapılmalı ki karakterler vocab'te olsun
        base_alphabet_added = self._ensure_base_alphabet_in_vocab()
        if base_alphabet_added > 0:
            logger.info(f"[BPEManager] Base alphabet/punct eklendi: +{base_alphabet_added} token (char fallback için)")
            # Trainer vocab'ini de güncelle (base alphabet karakterleri eklenmiş olabilir)
            # self._vocab'deki yeni karakterleri trainer vocab'ine ekle
            trainer_vocab = self.trainer.get_vocab()
            base_chars_added_to_trainer = 0
            for char in self._vocab:
                if char not in trainer_vocab:
                    # Trainer vocab'ine ekle
                    nid = next_id(trainer_vocab)
                    trainer_vocab[char] = {"id": nid, "total_freq": 0, "positions": []}
                    base_chars_added_to_trainer += 1
            if base_chars_added_to_trainer > 0:
                logger.info(f"[BPEManager] Base alphabet karakterleri trainer vocab'ine eklendi: +{base_chars_added_to_trainer} token")
        
        # 5) Eğitim öncesi vocab'ı seçilen tokenlarla genişlet (monotonik ID)
        self.trainer.update_vocab(selected_tokens)
        
        # Eğitim sırasında yeni tokenları takip etmek için
        initial_vocab_size = len(self.trainer.get_vocab())
        initial_all_tokens = set(all_tokens)
        
        # Memory monitoring
        self._monitor_memory_usage("Eğitim öncesi")
        
        # Eğitim sırasında yeni tokenları takip etmek için
        logger.info(f"[BPEManager] Eğitim öncesi vocab boyutu: {initial_vocab_size}")
        logger.info(f"[BPEManager] Eğitim öncesi unique token: {len(initial_all_tokens)}")

        # 3) BPE eğitim döngüsü
        try:
            self.trainer.train(
                trainer_input,
                target_merges=merges_to_do,
                max_iter=max_iter,
                min_frequency=min_frequency,
                append_eos=append_eos,
                protect_specials=protect_specials,
            )
            
            # Eğitim sonrası yeni tokenları vocab'a ekle
            final_vocab = self.trainer.get_vocab()
            new_tokens = set(final_vocab.keys()) - initial_all_tokens
            if new_tokens:
                logger.info(f"[BPEManager] Eğitim sonrası {len(new_tokens)} yeni token bulundu")
                all_tokens.update(new_tokens)
                # Yeni tokenları vocab'a ekle (zaten trainer'da var ama emin olmak için)
                self.trainer.update_vocab(sorted(new_tokens))
                
                # Vocab'ı güncelle
                self._vocab = self.trainer.get_vocab()
                
                # Final rapor
                logger.info(f"[BPEManager] Final vocab boyutu: {len(self._vocab)}")
                logger.info(f"[BPEManager] Final unique token: {len(all_tokens)}")
            else:
                logger.info(f"[BPEManager] Eğitim sonrası yeni token bulunamadı")
        except Exception as e:
            logger.error("BPE eğitimi sırasında hata: %s", e)
            raise BPETrainingError(f"BPE eğitimi başarısız: {e}") from e

        # 4) Trainer'dan çıkan vocab & merges ile tek kaynaklı doğruluk
        we_vocab = self.trainer.get_vocab()
        self._vocab = we_vocab        # trainer'ın güncel sözlüğü
        merges_list = self.trainer.get_merges()
        
        # Eğitim sırasında eklenen yeni tokenları kontrol et ve all_tokens'a ekle
        final_vocab_size = len(we_vocab)
        new_tokens_added = final_vocab_size - initial_vocab_size
        if new_tokens_added > 0:
            logger.info(f"[BPEManager] Eğitim sırasında {new_tokens_added} yeni token vocab'a eklendi")
            # Tüm yeni tokenları all_tokens setine ekle
            all_vocab_tokens = set(we_vocab.keys())
            all_tokens.update(all_vocab_tokens)
            logger.info(f"[BPEManager] all_tokens seti güncellendi: {len(all_tokens)} unique token")
        else:
            logger.info(f"[BPEManager] Eğitim sırasında yeni token eklenmedi")

        # 4.1) Trainer hiç merge üretmediyse: istatistik tabanlı sentetik merges
        if merges_to_do and merges_to_do > 0 and not merges_list:
            logger.debug("[BPEManager] Trainer merges üretmedi, sentetik merges devreye alınıyor.")
            merges_list = self._synthesize_merges(trainer_input, merges_to_do)
            self.trainer.set_merges(merges_list)

        # 5) Diske yaz ve senkronize et (atomik)
        logger.info("[BPEManager] Sonuçlar diske kaydediliyor...")
        self.save_vocab()
        self.save_merges(merges_list)
        self.finalize_vocab()

        # Final rapor
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Frekans bilgilerini güncelle
        logger.info("[BPEManager] Frekans bilgileri güncelleniyor...")
        self._update_token_frequencies(trainer_input)
        
        logger.info("=" * 60)
        logger.info("BPE EĞİTİMİ TAMAMLANDI!")
        logger.info("=" * 60)
        logger.info(f"Toplam süre: {total_time:.2f} saniye ({total_time/60:.1f} dakika)")
        logger.info(f"Final merges: {len(merges_list):,}")
        logger.info(f"Final vocab boyutu: {len(self._vocab):,}")
        logger.info(f"İşlenen cümle: {len(trainer_input):,}")
        logger.info(f"Unique token: {len(all_tokens):,}")
        logger.info(f"Final bellek kullanımı: {final_memory:.2f} MB (+{memory_increase:.2f} MB)")
        logger.info(f"Ortalama hız: {len(trainer_input)/total_time:.1f} cümle/saniye")
        logger.info("=" * 60)

    # -------------------------------------- Persist --------------------------------------

    
    def _monitor_memory_usage(self, stage: str):
        """Memory usage monitoring"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 8000:  # 8GB threshold
            logger.warning(f"[BPEManager] High memory usage at {stage}: {memory_mb:.1f} MB")
            import gc
            gc.collect()
            memory_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"[BPEManager] Memory after cleanup: {memory_after:.1f} MB")
        
        return memory_mb

    def _update_token_frequencies(self, tokenized_corpus: List[List[str]]) -> None:
            """
            Eğitim sonrası token frekanslarını güncelle
            """
            from collections import Counter
            
            # Tüm tokenları topla
            all_tokens = []
            for sequence in tokenized_corpus:
                all_tokens.extend(sequence)
            
            # Frekansları hesapla
            token_freqs = Counter(all_tokens)
            
            # Vocab'daki frekansları güncelle
            updated_count = 0
            for token, freq in token_freqs.items():
                if token in self._vocab:
                    self._vocab[token]["total_freq"] = freq
                    updated_count += 1
            
            logger.info(f"[BPEManager] {updated_count} token frekansı güncellendi")
            
            # Vocab'ı kaydet
            self.save_vocab()

    def finalize_vocab(self, sample_texts: Optional[List[str]] = None) -> None:
        """
        Diskteki vocab/merges'i tekrar RAM'e al ve encoder/decoder/trainer'ı senkronize et.
        İlk eğitim sırasında yaygın tokenları da ekle.
        """
        self._vocab = self.load_vocab()
        merges = self._read_merges_file()
        self.trainer.set_merges(merges)
        self._sync_components()
        
        # Base alphabet/punct garanti
        added = self._ensure_base_alphabet_in_vocab()
        if added:
            logger.info("[BPEManager] finalize_vocab: base alphabet/punct +%d token eklendi.", added)
        
        # İlk eğitim sırasında yaygın tokenları ekle
        if sample_texts and len(self._vocab) < 1000:  # Vocab küçükse (ilk eğitim)
            common_added = self._ensure_common_tokens_in_vocab(sample_texts)
            if common_added:
                logger.info("[BPEManager] finalize_vocab: yaygın tokenlar +%d token eklendi.", common_added)
        
        logger.info("[BPEManager] finalize_vocab | vocab=%d merges=%d", len(self._vocab), len(merges))

    def load_vocab(self) -> Dict[str, dict]:
        raw = self._read_json(self.vocab_file)
        return normalize_vocab(raw)

    def save_vocab(self) -> None:
        self._write_json(self.vocab_file, self._vocab)
        logger.debug("Vocab kaydedildi: %s", self.vocab_file)

    def save_merges(self, merges: Optional[List[Tuple[str, str]]] = None) -> None:
        merges_to_write = merges if merges is not None else self.trainer.get_merges()
        self._write_merges_atomic(merges_to_write or [])
        logger.debug("Merges kaydedildi (atomik): %s", self.merges_file)

    # ---------------------------------- Yardımcılar ----------------------------------

    def get_vocab(self) -> Dict[str, dict]:
        return self._vocab

    def get_merges(self) -> List[Tuple[str, str]]:
        return self.trainer.get_merges()

    def set_vocab(self, new_vocab: Dict[str, Union[int, dict]]) -> None:
        self._vocab = normalize_vocab(new_vocab)
        self._write_json(self.vocab_file, self._vocab)
        self._sync_components()
        # Base alphabet/punct garanti
        added = self._ensure_base_alphabet_in_vocab()
        if added:
            logger.info("[BPEManager] set_vocab: base alphabet/punct +%d token eklendi.", added)
        logger.info("Yeni vocab yüklendi ve senkronize edildi. size=%d", len(self._vocab))

    def reset(self) -> None:
        self._vocab = default_vocab()
        self._write_json(self.vocab_file, self._vocab)
        self._write_merges_atomic([])  # merges temizle
        self._initialize_components()
        self._sync_components()
        logger.info("BPEManager sıfırlandı: default vocab & temiz merges")

    def finalize(self) -> None:
        self.save_vocab()
        self.save_merges()
        self._sync_components()
        logger.info("BPEManager finalize edildi: vocab & merges kesinleşti")

    def load_vocab_and_merges(self) -> None:
        """
        TokenizerManager ile API uyumu için:
        Vocab ve merges dosyalarını yükler, trainer ve encoder/decoder'ı senkronize eder.
        """
        self._vocab = self.load_vocab()
        merges = self._read_merges_file()
        self.trainer.set_merges(merges)
        self._sync_components()
        # Base alphabet/punct garanti
        added = self._ensure_base_alphabet_in_vocab()
        if added:
            logger.info("[BPEManager] load_vocab_and_merges: base alphabet/punct +%d token eklendi.", added)
        logger.info("BPEManager: Vocab & merges diskten yüklendi ve senkronize edildi.")

    # -------------------------- Internal: sentetik merges ------------------------

    def _synthesize_merges(self, sequences: List[List[str]], limit: int) -> List[Tuple[str, str]]:
        """
        Eğer trainer herhangi bir merge üretemediyse (örneğin yalnız kelime/SEP tokenizasyonu),
        en sık görülen komşu token çiftlerinden basit bir merges listesi üret.
        Bu, API sözleşmesini (dosyada & bellekte merges varlığı) garanti eder.
        """
        if limit <= 0:
            return []
        from collections import Counter

        forbidden = {"<BOS>", "<EOS>", "<PAD>", "<UNK>"}
        pair_counter: Counter = Counter()

        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                if a in forbidden or b in forbidden:
                    continue
                pair_counter[(a, b)] += 1

        if not pair_counter:
            return []

        candidates = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        merges: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for (a, b), _freq in candidates:
            if (a, b) in seen:
                continue
            merges.append((a, b))
            seen.add((a, b))
            if len(merges) >= limit:
                break
        return merges

    # ---------------------- Internal: base alphabet inşa ----------------------

    def _ensure_special_tokens_in_vocab(self) -> int:
        """
        DEFAULT_SPECIALS'taki özel tokenların vocab'da mevcut olduğunu garanti eder.
        Eksik özel tokenlar eklenir, yanlış ID'li olanlar düzeltilir.
        Read-only modda dosyaya kaydetmez.
        """
        added = 0
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            meta = self._vocab.get(sp_tok)
            if meta is None:
                # Eksik özel token ekle
                self._vocab[sp_tok] = {"id": sp_id, "total_freq": 0, "positions": []}
                added += 1
                logger.info("[BPEManager] Eksik özel token eklendi: %s -> id=%d", sp_tok, sp_id)
            else:
                # ID kontrolü - yanlışsa düzelt
                current_id = meta.get("id")
                if current_id != sp_id:
                    logger.warning(
                        "[BPEManager] Özel token ID düzeltiliyor: %s id=%d -> %d",
                        sp_tok, current_id, sp_id
                    )
                    self._vocab[sp_tok] = {"id": sp_id, "total_freq": meta.get("total_freq", 0), "positions": meta.get("positions", [])}
        
        if added > 0 and not getattr(self, 'read_only', False):
            # Vocab değişti, dosyaya kaydet (read-only modda değilse)
            self.save_vocab()
        
        return added
    
    def _check_special_tokens_in_vocab(self) -> None:
        """
        Read-only modda: Özel tokenları sadece kontrol et, ekleme yapma.
        """
        missing = []
        mismatched = []
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            meta = self._vocab.get(sp_tok)
            if meta is None:
                missing.append((sp_tok, sp_id))
            else:
                current_id = meta.get("id")
                if current_id != sp_id:
                    mismatched.append((sp_tok, current_id, sp_id))
        
        if missing:
            logger.warning(
                "[BPEManager] Read-only mod: Eksik özel tokenlar tespit edildi (eklenmedi): %s",
                missing
            )
        if mismatched:
            logger.warning(
                "[BPEManager] Read-only mod: Özel token ID uyuşmazlıkları tespit edildi (düzeltilmedi): %s",
                mismatched
            )

    def _ensure_base_alphabet_in_vocab(self) -> int:
        """
        OPTIMIZED: Sadece base karakterleri ekle (vocab gereksiz büyümesin).
        GPT gibi modellerde karakterler her zaman vocab'te olmalı (char fallback için).
        NOT: char + '</w>' formları eklenmez, sadece tek karakterler eklenir.
        
        ✅ DÜZELTME: get_base_alphabet() kullan (tutarlılık için)
        ⚠️ NOT: Vocab küçük harflerle oluşturulmuş, bu yüzden get_base_alphabet() sadece küçük harfleri döndürüyor
        """
        # ✅ get_base_alphabet() kullan (tutarlılık için)
        from tokenizer_management.bpe.bpe_manager_utils import get_base_alphabet
        base = get_base_alphabet()  # Küçük harfler, rakamlar, noktalama, boşluk, </w>

        specials = set(DEFAULT_SPECIALS.keys())
        to_add: List[str] = []
        
        # Sadece tek karakterleri ekle (char + '</w>' formları eklenmez)
        # Karakter bazlı parçalarken </w> sembolüne ihtiyaç yok
        # ÖNEMLİ: Hem 'ch' hem de 'ch</w>' formunu kontrol et - tekrar ekleme!
        for ch in base:
            ch_with_suffix = ch + "</w>"
            # Eğer hem 'ch' hem de 'ch</w>' yoksa, sadece 'ch' ekle
            # Eğer 'ch</w>' varsa, 'ch' eklemeye gerek yok (zaten var)
            if ch not in self._vocab and ch_with_suffix not in self._vocab:
                to_add.append(ch)

        # <SEP> özel tokenını garanti altına al
        if "<SEP>" not in self._vocab:
            to_add.append("<SEP>")
        
        # </w> token'ını da ekle (yalnız başına kullanılabilir, BPE merge'ler için)
        if "</w>" not in self._vocab:
            to_add.append("</w>")

        # Eklemeyi uygula
        added = 0
        if to_add:
            for tok in to_add:
                if tok in specials:
                    # DEFAULT_SPECIALS setiyle çakışma olmasın; zaten default_vocab bunu ekliyor
                    continue
                if tok in self._vocab:
                    continue
                nid = next_id(self._vocab)
                self._vocab[tok] = {"id": nid, "total_freq": 0, "positions": []}
                added += 1
            # Persist + sync (read-only modda dosyaya kaydetme)
            if not getattr(self, 'read_only', False):
                self.save_vocab()
            self._sync_components()
        return added

    def _ensure_common_tokens_in_vocab(self, sample_texts: List[str]) -> int:
        """
        YENİ: Yaygın heceler, morfemler ve kelimeleri vocab'a ekle.
        Bu, ilk eğitim sırasında bilinmeyen token sorununu çözer.
        """
        from tokenizer_management.bpe.tokenization.syllabifier import Syllabifier
        from tokenizer_management.bpe.tokenization.morphology import Morphology
        from tokenizer_management.bpe.tokenization.pretokenizer import Pretokenizer
        
        logger.info("[BPEManager] Yaygın tokenları vocab'a ekleniyor...")
        
        # Tokenizer bileşenlerini başlat
        pretokenizer = Pretokenizer()
        syllabifier = Syllabifier()
        morphology = Morphology()
        
        # Tüm tokenları topla
        all_tokens = set()
        
        for text in sample_texts[:100]:  # İlk 100 metni analiz et
            try:
                # 1. Pretokenizer
                pretokenized = pretokenizer.tokenize(text)
                
                for token in pretokenized:
                    if token.isalpha():
                        # 2. Syllabifier
                        syllables = syllabifier.syllabify_word(token)
                        all_tokens.update(syllables)
                        
                        # 3. Morphology
                        morphemes = morphology.analyze([token])
                        all_tokens.update(morphemes)
                    else:
                        all_tokens.add(token)
                        
            except Exception as e:
                logger.debug(f"[BPEManager] Token analizi hatası: {e}")
                continue
        
        # Vocab'a ekle
        added = 0
        for token in all_tokens:
            if token not in self._vocab and len(token) > 0:
                nid = next_id(self._vocab)
                self._vocab[token] = {"id": nid, "total_freq": 0, "positions": []}
                added += 1
        
        if added > 0:
            logger.info(f"[BPEManager] {added} yaygın token vocab'a eklendi")
            self.save_vocab()
            self._sync_components()
        
        return added

    # ----------------------- Vocab size kontrolü ve token seçimi -----------------------

    def _calculate_token_frequencies(self, trainer_input: List[List[str]]) -> Dict[str, int]:
        """
        Token frekanslarını hesapla
        """
        frequencies = {}
        for tokens in trainer_input:
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _select_top_tokens(self, all_tokens: Set[str], token_frequencies: Dict[str, int], max_vocab_size: int) -> List[str]:
        """
        En sık kullanılan token'ları seç
        Special token'ları garantile, kalan slotları en sık kullanılan token'larla doldur
        """
        # Mevcut vocab'deki token'ları kontrol et
        existing_vocab = set(self.trainer.get_vocab().keys())
        
        # Special token'ları garantile (zaten vocab'de varsa ekleme)
        special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"}
        selected = []
        for st in special_tokens:
            if st not in existing_vocab:
                selected.append(st)
        
        # Kalan slot sayısı (mevcut vocab size'ı dikkate al)
        current_vocab_size = len(existing_vocab)
        remaining_slots = max_vocab_size - current_vocab_size - len(selected)
        
        if remaining_slots <= 0:
            logger.warning(f"[BPEManager] max_vocab_size ({max_vocab_size}) çok küçük, mevcut vocab size: {current_vocab_size}")
            return selected
        
        # En sık kullanılan token'ları seç (special token'lar ve mevcut vocab'deki token'lar hariç)
        sorted_tokens = sorted(
            [(token, freq) for token, freq in token_frequencies.items() 
             if token not in special_tokens and token not in existing_vocab],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top N token'ı seç
        top_tokens = [token for token, freq in sorted_tokens[:remaining_slots]]
        selected.extend(top_tokens)
        
        return selected

    # ----------------------- Public: vocab genişletme (NEW) -----------------------

    def auto_update_vocab(self, tokens: List[str]) -> int:
        """
        Ham tokenları vocab’a ekler (merge türevlerini burada eklemeyiz).
        Monotonik ID ataması için utils.next_id kullanılır.
        - Special'lar (<PAD>, <UNK>, <BOS>, <EOS>) asla eklenmez/yenilenmez
        - Tam tag görünümlü (<...>) olanlar hariç tutulur; yalnız <SEP> tutulur
        """
        if not tokens:
            return 0

        specials = set(DEFAULT_SPECIALS.keys())

        def is_taglike(s: str) -> bool:
            return isinstance(s, str) and s.startswith("<") and s.endswith(">")

        cleaned_set = set(clean_tokens(tokens))
        # <SEP>’i koru, diğer full-tag’leri at
        filtered = {
            t for t in cleaned_set
            if (t == "<SEP>") or (t not in specials and not is_taglike(t))
        }

        unseen = sorted(t for t in filtered if t not in self._vocab)
        if not unseen:
            return 0

        added = 0
        for tok in unseen:
            nid = next_id(self._vocab)
            self._vocab[tok] = {"id": nid, "total_freq": 0, "positions": []}
            added += 1

        if added:
            self.save_vocab()
            self._sync_components()
        return added

    def _ensure_tokens_in_vocab_gpu(
        self, 
        texts: List[str], 
        *, 
        include_whole_words: bool, 
        include_syllables: bool, 
        include_sep: bool
    ) -> Set[str]:
        """
        GPU ile batch processing ile vocab genişletme
        """
        import torch
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp
        
        candidates = set()
        total_texts = len(texts)
        batch_size = self.config.get("gpu_batch_size", 32)  # GPU batch size (config'ten)
        
        logger.info(f"[BPEManager] GPU batch processing: {total_texts:,} metin, batch_size={batch_size}")
        
        # Metinleri batch'lere böl
        batches = [texts[i:i + batch_size] for i in range(0, total_texts, batch_size)]
        
        batch_log_interval = self.config.get("batch_log_interval", 10)
        for batch_idx, batch in enumerate(batches):
            if batch_idx % batch_log_interval == 0:  # Config'ten interval
                processed = batch_idx * batch_size
                logger.info(f"[BPEManager] GPU Batch: {processed:,}/{total_texts:,} ({(processed/total_texts)*100:.1f}%)")
            
            # Batch'i paralel işle (ThreadPoolExecutor ile)
            with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
                futures = []
                for text in batch:
                    future = executor.submit(
                        self.preview_tokens,
                        text,
                        include_whole_words=include_whole_words,
                        include_syllables=include_syllables,
                        include_sep=include_sep
                    )
                    futures.append(future)
                
                # Sonuçları topla
                future_timeout = self.config.get("future_timeout", 30)
                for future in futures:
                    try:
                        tokens = future.result(timeout=future_timeout)  # Config'ten timeout
                        candidates.update(tokens)
                    except Exception as e:
                        logger.warning(f"[BPEManager] GPU batch processing hatası: {e}")
                        continue
        
        logger.info(f"[BPEManager] GPU batch processing tamamlandı. {len(candidates):,} unique token bulundu.")
        return candidates
