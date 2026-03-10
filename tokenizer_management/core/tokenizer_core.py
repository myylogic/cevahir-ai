# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tokenizer_core.py
Modül: tokenizer_management/core
Görev: TokenizerCore sınıfı - Veri tokenizasyonu, encoding/decoding işlemleri
       ve eğitim verisi hazırlama işlemlerini yönetir. BPEManager ile entegre
       çalışarak veri → tokenizasyon → eğitim örneği üretimi → inference
       akışını birleştirir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (tokenizasyon ve veri hazırlama),
                     Dependency Inversion (BPEManager interface'i),
                     Open/Closed (genişletilebilir mod yapısı)
- Design Patterns: Strategy Pattern (farklı encoding modları),
                  Facade Pattern (BPEManager ve DataLoaderManager entegrasyonu)
- Endüstri Standartları: GPT-2/3/4, BERT, T5 tokenization standartları,
                         Autoregressive training format (BOS/EOS/SEP/PAD)

KULLANIM:
- Eğitim verisi hazırlama: load_training_data() - Cache hazırlama için
- Inference: encode() / decode() - Metin tokenizasyonu için
- Batch işlemler: batch_encode() - Toplu encoding için
- Cache hazırlama: prepare_cache.py ile birlikte kullanılır

BAĞIMLILIKLAR:
- BPEManager: BPE tokenization işlemleri
- DataLoaderManager: Veri yükleme işlemleri (QA çiftleri, RAW text)
- TokenizerCoreError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable

# Data loader import
from data_loader_management.data_loader_manager import (
    DataLoaderManager,
    DataLoaderConfig,
    LoadMode,
    DataLoaderError as DataLoadError,
    UnsupportedFormatError,
    DataDirectoryNotFoundError as DLFileNotFound,
)

from tokenizer_management.bpe.bpe_manager import (
    BPEManager,
    BPETokenError,
    BPEDecodingError,
    BPETrainingError,
)

from tokenizer_management.bpe.bpe_manager_utils import get_valid_ids

logger = logging.getLogger(__name__)

# --- Geriye dönük uyumluluk alias'ları (eski isimleri kullanan kodlar/except blokları için) ---
DataDirectoryNotFoundError = DLFileNotFound
DataLoaderError = DataLoadError


class TokenizerCoreError(Exception):
    """Genel TokenizerCore hataları için exception."""
    pass


class TokenizerCore:
    """
    TokenizerCore, veri → tokenizasyon → eğitim örneği üretimi → inference akışını birleştirir.
    BPEManager ile hizalıdır; tüm normalizasyon/tokenizasyon işleri BPE tarafındadır.
    """

    # ------------------------------- Yaşam Döngüsü -------------------------------

    def __init__(self, config: Dict[str, Any]):
        # Config'i sakla (UNK ratio monitoring için)
        self.config = config
        
        data_dir = config.get("data_dir")
        
        # GPU desteği
        self.use_gpu = config.get("use_gpu", False)
        self.batch_size = config.get("batch_size", 32)
        
        # GPU kontrolü
        if self.use_gpu:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cpu":
                    logger.warning("[TokenizerCore] GPU isteniyor ama CUDA mevcut değil, CPU kullanılacak")
                    self.use_gpu = False
                else:
                    logger.info(f"[TokenizerCore] GPU desteği aktif: {self.device}")
            except ImportError:
                logger.warning("[TokenizerCore] PyTorch bulunamadı, GPU desteği devre dışı")
                self.use_gpu = False
                self.device = None
        else:
            self.device = None

        if data_dir:
            p = Path(data_dir)
            if not p.is_dir():
                raise TokenizerCoreError(f"Geçerli bir data_dir girilmeli: {str(p)!r}")

            # Yeni API: DataLoaderConfig ile kur
            dl_cfg = DataLoaderConfig(
                data_dir=p,
                mode=LoadMode.QA_TRAIN,  # testler eğitim (QA) akışını bekliyor
                strict=False,
            )
            self.data_loader = DataLoaderManager(dl_cfg)
            try:
                logger.info(f"[TokenizerCore] DataLoaderManager hazır → dir={p}, mode={dl_cfg.mode}")
            except Exception:
                logger.info(f"[TokenizerCore] DataLoaderManager hazır → dir={p}")
        else:
            self.data_loader = None
            logger.info("[TokenizerCore] DataLoaderManager atlandı (data_dir verilmedi)")

        # Vocab & merges path'leri
        self.vocab_path = config.get("vocab_path") or config.get("vocab_file")
        self.merges_path = config.get("merges_path") or config.get("merges_file")
        if not self.vocab_path or not self.merges_path:
            raise TokenizerCoreError(
                "config içinde 'vocab_path'/'vocab_file' ve 'merges_path'/'merges_file' tanımlı olmalı"
            )
        
        # BPEManager'ın _resolve_paths metodunu simüle et (os.path.abspath kullanıyor)
        # Bu, Windows'ta /nonexistent gibi path'lerin nasıl çözümleneceğini görmek için
        resolved_vocab_path = os.path.abspath(self.vocab_path)
        resolved_merges_path = os.path.abspath(self.merges_path)
        
        # Check if original path looks like an absolute path (Unix-style or Windows-style)
        # Windows'ta / ile başlayan path'ler relative olarak algılanır ama test için mutlak gibi davranmalıyız
        # Unix-style absolute path: starts with / but not followed by : (which would be Windows drive)
        original_vocab_is_abs = os.path.isabs(self.vocab_path) or (
            os.name == 'nt' and self.vocab_path.startswith('/') and 
            len(self.vocab_path) > 1 and self.vocab_path[1] != ':'
        )
        original_merges_is_abs = os.path.isabs(self.merges_path) or (
            os.name == 'nt' and self.merges_path.startswith('/') and 
            len(self.merges_path) > 1 and self.merges_path[1] != ':'
        )
        
        # Validate resolved vocab_path parent directory
        vocab_parent = os.path.dirname(resolved_vocab_path)
        
        if original_vocab_is_abs:
            # For absolute paths (including Unix-style on Windows), parent directory must exist and be writable
            # On Windows, /nonexistent/path.json -> C:\...\nonexistent\path.json (resolved to current dir)
            # But we want to check if the resolved parent exists (which it won't if /nonexistent doesn't exist)
            if not os.path.exists(vocab_parent):
                raise TokenizerCoreError(
                    f"Vocab path parent directory does not exist: {vocab_parent} (resolved from: {self.vocab_path})"
                )
            if not os.access(vocab_parent, os.W_OK):
                raise TokenizerCoreError(
                    f"Vocab path parent directory is not writable: {vocab_parent}"
                )
        else:
            # For relative paths, try to create parent directory
            try:
                os.makedirs(vocab_parent, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise TokenizerCoreError(
                    f"Cannot create vocab path parent directory: {vocab_parent}, error: {e}"
                ) from e
        
        # Validate resolved merges_path parent directory
        merges_parent = os.path.dirname(resolved_merges_path)
        
        if original_merges_is_abs:
            # Same logic as vocab_path
            if not os.path.exists(merges_parent):
                raise TokenizerCoreError(
                    f"Merges path parent directory does not exist: {merges_parent} (resolved from: {self.merges_path})"
                )
            if not os.access(merges_parent, os.W_OK):
                raise TokenizerCoreError(
                    f"Merges path parent directory is not writable: {merges_parent}"
                )
        else:
            # For relative paths, try to create parent directory
            try:
                os.makedirs(merges_parent, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise TokenizerCoreError(
                    f"Cannot create merges path parent directory: {merges_parent}, error: {e}"
                ) from e

        # BPEManager - GPU desteği ile
        # Read-only mode: Eğer vocab dosyası mevcutsa ve test modunda ise, vocab'a ekleme yapma
        bpe_config = {}
        if config.get("read_only", False):
            bpe_config["read_only"] = True
        
        self.tokenizer = BPEManager(
            vocab_file=self.vocab_path,
            merges_file=self.merges_path,
            use_gpu=self.use_gpu,
            config=bpe_config,
        )
        logger.info(
            "[TokenizerCore] BPEManager hazır "
            f"(vocab={self.vocab_path}, merges={self.merges_path}, gpu={self.use_gpu})"
        )

        # --- MOD-BAĞIMLI VARSAYILANLAR (config ile override edilebilir) ---
        self._train_defaults = {
            "include_whole_words": bool(config.get("train_include_whole_words", True)),
            "include_syllables":  bool(config.get("train_include_syllables",  True)),
            "include_sep":        bool(config.get("train_include_sep",        True)),
        }
        self._infer_defaults = {
            "include_whole_words": bool(config.get("inference_include_whole_words", True)),
            "include_syllables":   bool(config.get("inference_include_syllables",   False)),
            "include_sep":         bool(config.get("inference_include_sep",         False)),
        }

        # Özel ID önbelleği
        self._cached_special_ids: Optional[Dict[str, int]] = None

    # ------------------------------- Eğitim API’si --------------------------------

    def train_model(
        self,
        corpus: List[str],
        *,
        method: str = "bpe",
        vocab_size: Optional[int] = None,
        max_iter: Optional[int] = None,
        min_frequency: Optional[int] = None,
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        append_eos: Optional[bool] = None,
        protect_specials: Optional[bool] = None,
        sample_ratio: Optional[float] = None,  # Yeni parametre: 0.1 için %10 sampling
    ) -> None:
        """Dışarıdan verilen corpus ile BPE eğitimi yapar ve persist eder."""
        if not corpus:
            raise TokenizerCoreError("train_model: Empty corpus provided.")
        if method.lower() != "bpe":
            raise TokenizerCoreError(f"Unsupported method: {method!r}")

        # Yeni: Sampling uygula (opsiyonel, akışı bozmadan)
        original_size = len(corpus)
        if sample_ratio is not None and 0 < sample_ratio < 1:
            if original_size < 100:  # Küçük corpus'ta sampling yapma (threshold ayarlanabilir)
                logger.warning(f"Corpus küçük ({original_size}), sampling atlandı.")
            else:
                import random
                sample_size = max(1, int(original_size * sample_ratio))
                corpus = random.sample(corpus, sample_size)
                logger.info(f"[TokenizerCore] Sampling uygulandı: {original_size} → {len(corpus)} (%{int(sample_ratio*100)})")

        logger.info("[TokenizerCore] train_model başlıyor: %d satır (sampled)", len(corpus))
        try:
            kwargs = {
                "include_whole_words": include_whole_words,
                "include_syllables": include_syllables,
                "include_sep": include_sep,
                "append_eos": append_eos,
                "protect_specials": protect_specials,
            }
            # vocab_size parametresi max_vocab_size limiti olarak kullanılır
            # target_merges ayrı bir parametre (merge sayısı)
            if vocab_size is not None:
                # vocab_size vocab boyutu limiti olarak kullanılır
                # target_merges merge sayısı limiti (vocab_size'dan farklı olabilir)
                # Şu an target_merges yok, sadece max_vocab_size kullanılıyor
                pass  # vocab_size zaten config'te max_vocab_size olarak kullanılıyor
            if isinstance(max_iter, int):
                kwargs["max_iter"] = max_iter
            if isinstance(min_frequency, int):
                kwargs["min_frequency"] = min_frequency

            self.tokenizer.train(corpus, **kwargs)  # Artık sampled corpus ile

            # Persist + sync (değişmedi)
            self.tokenizer.save_vocab()
            self.tokenizer.save_merges()
            self.tokenizer.finalize_vocab()

            logger.info(
                "[TokenizerCore] train_model tamamlandı: "
                f"vocab={self.vocab_path}, merges={self.merges_path}"
            )
        except (BPETrainingError, BPETokenError) as e:
            logger.error(f"[TokenizerCore] train_model hatası: {e}")
            raise

    def train_from_loader(
        self,
        *,
        method: str = "bpe",
        **train_kwargs: Any,
    ) -> None:
        """
        DataLoaderManager ile ham veriyi okuyup corpus oluşturur ve train eder.
        QA_TRAIN modunda: Soru+Cevap metinleri birleştirilerek corpus oluşturulur.
        TEXT_INFER modunda: Text listesi doğrudan corpus'tur.
        """
        if self.data_loader is None:
            raise TokenizerCoreError("train_from_loader: DataLoader yok (data_dir verilmemiş).")

        try:
            data = self.data_loader.load()
        except (DataDirectoryNotFoundError, UnsupportedFormatError, DataLoaderError) as e:
            logger.error(f"[TokenizerCore] train_from_loader yükleme hatası: {e}")
            raise TokenizerCoreError(f"train_from_loader: {e}")

        corpus: List[str]
        if self.data_loader.cfg.mode == LoadMode.QA_TRAIN:
            assert isinstance(data, list)
            # Raw text detection: Boş soru = raw text chunk
            qa_texts = []
            raw_texts = []
            
            for q, a in data:
                if q == "" and a != "":
                    # Bu raw text chunk (DOCX/TXT'ten gelen)
                    raw_texts.append(a)
                else:
                    # Bu gerçek QA çifti (JSON'dan gelen)
                    qa_texts.extend([q, a])
            
            # Corpus oluştur: QA metinleri + raw text chunks (duplicate yok)
            corpus = qa_texts + raw_texts
            
            logger.info(f"[TokenizerCore] Corpus oluşturuldu: {len(qa_texts)} QA metni + {len(raw_texts)} raw text chunk = {len(corpus)} toplam")
        else:  # TEXT_INFER
            corpus = list(data)  # List[str]

        if not corpus:
            raise TokenizerCoreError("train_from_loader: Boş corpus (geçerli metin yok).")

        self.train_model(corpus, method=method, **train_kwargs)

    # ------------------------------ Persist / Sync -------------------------------

    def finalize_vocab(self, sample_texts: Optional[List[str]] = None) -> None:
        """Var olan vocab/merges dosyalarını yükler ve senkronize eder."""
        logger.info("[TokenizerCore] finalize_vocab: vocab+merges yükleniyor")
        try:
            self.tokenizer.load_vocab_and_merges()
            # BPE Manager'ın finalize_vocab metodunu çağır (yaygın tokenları eklemek için)
            self.tokenizer.finalize_vocab(sample_texts)
            self._cached_special_ids = None
            logger.info("[TokenizerCore] finalize_vocab tamamlandı")
        except BPETokenError as e:
            logger.error(f"[TokenizerCore] finalize_vocab hatası: {e}")
            raise

    def save(self) -> None:
        """Vocab & merges'ı diske yazar (atomik)."""
        try:
            self.tokenizer.finalize()
            self._cached_special_ids = None
        except Exception as e:
            logger.error(f"[TokenizerCore] save hatası: {e}")
            raise

    def reset(self) -> None:
        """Vocab'ı varsayılanlara döndürür, merges'ı temizler ve bileşenleri yeniler."""
        try:
            self.tokenizer.reset()
            self._cached_special_ids = None
        except Exception as e:
            logger.error(f"[TokenizerCore] reset hatası: {e}")
            raise

    # ------------------------------ Yardımcılar ----------------------------------

    def _resolve_include_flags(
        self,
        mode: str,
        include_whole_words: Optional[bool],
        include_syllables: Optional[bool],
        include_sep: Optional[bool],
    ) -> Tuple[bool, bool, bool]:
        """None bırakılan bayrakları moda göre akıllı doldur."""
        if mode.lower() == "train":
            d = self._train_defaults
        else:
            d = self._infer_defaults
        iw = d["include_whole_words"] if include_whole_words is None else include_whole_words
        isy = d["include_syllables"]  if include_syllables  is None else include_syllables
        isp = d["include_sep"]        if include_sep        is None else include_sep
        return bool(iw), bool(isy), bool(isp)

    def _special_ids(self) -> Dict[str, int]:
        """Özel token ID’lerini (örn. UNK) getirir, cache’ler."""
        if self._cached_special_ids is not None:
            return self._cached_special_ids
        v = self.get_vocab()
        out: Dict[str, int] = {}
        for key in ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"):
            meta = v.get(key)
            if isinstance(meta, dict) and isinstance(meta.get("id"), int):
                out[key] = int(meta["id"])
        self._cached_special_ids = out
        return out

    # ------------------------------ Encode / Decode ------------------------------

    def encode(
        self,
        text: str,
        *,
        mode: str = "inference",
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        add_special_tokens: Optional[bool] = None,
        **kwargs  # Accept additional kwargs for compatibility
    ) -> Tuple[List[str], List[int]]:
        """Tekil metni BPE ile encode eder.
        
        ENDÜSTRI STANDARDI: add_special_tokens parametresi BPEManager'a iletilir.
        """
        # [OK] Boş string kontrolü (ENDÜSTRI STANDARDI)
        if not text or not text.strip():
            logger.debug("[TokenizerCore] encode: Boş string, boş liste döndürülüyor")
            return [], []
        
        # [OK] NOT: Special token kontrolü BPEManager.encode()'da yapılıyor
        # TokenizerCore sadece wrapper, encoding mantığı BPEManager'da
        
        iw, isy, isp = self._resolve_include_flags(
            mode, include_whole_words, include_syllables, include_sep
        )
        # --- TRAIN modunda iw=False & isy=False ise boş dizi oluşmaması için akıllı fallback ---
        if mode.lower() == "train" and not iw and not isy:
            logger.debug("[TokenizerCore] encode(train): iw=False & isy=False geldi, iw=True fallback uygulanıyor.")
            iw = True

        try:
            # [OK] ENDÜSTRI STANDARDI: add_special_tokens parametresini BPEManager'a ilet
            tokens, token_ids = self.tokenizer.encode(
                text,
                mode=mode,
                include_whole_words=iw,
                include_syllables=isy,
                include_sep=isp,
                add_special_tokens=add_special_tokens,
            )
            sp = self._special_ids()
            unk_id = sp.get("<UNK>", None)
            unk_cnt = int(sum(1 for i in token_ids if (unk_id is not None and i == unk_id)))
            unk_ratio = (unk_cnt / max(1, len(token_ids))) if token_ids else 0.0

            # UNK ratio monitoring (config'ten threshold)
            max_unk_ratio = self.config.get("max_unk_ratio", 0.01)  # Default: %1
            if unk_ratio > max_unk_ratio:
                logger.warning(
                    f"[TokenizerCore] YUKSEK UNK ORANI! unk_ratio={unk_ratio:.4f} > {max_unk_ratio} "
                    f"(unk_cnt={unk_cnt}/{len(token_ids)} token). "
                    f"Vocab veya BPE training iyilestirilmeli!"
                )

            logger.debug(
                "[TokenizerCore] encode: mode=%s, iw=%s, isy=%s, isp=%s, text='%s', token_ids_len=%d, unk_cnt=%d, unk_ratio=%.4f",
                mode, iw, isy, isp, (str(text)[:50] if isinstance(text, str) else str(text)) , len(token_ids), unk_cnt, unk_ratio
            )
            return tokens, token_ids
        except BPETokenError as e:
            logger.error(f"[TokenizerCore] encode hatası: {e}")
            raise TokenizerCoreError(f"encode hatası: {e}") from e
        except Exception as e:
            logger.error(f"[TokenizerCore] Beklenmeyen encode hatası: {e}")
            raise TokenizerCoreError(f"Beklenmeyen encode hatası: {e}") from e

    def encode_with_stats(
        self,
        text: str,
        *args,
        **kwargs,
    ) -> Tuple[List[str], List[int], Dict[str, float]]:
        """encode() + istatistik (UNK sayısı/oranı, uzunluk)."""
        toks, ids = self.encode(text, *args, **kwargs)
        sp = self._special_ids()
        unk_id = sp.get("<UNK>", None)
        unk_cnt = int(sum(1 for i in ids if (unk_id is not None and i == unk_id)))
        stats = {
            "length": float(len(ids)),
            "unk_count": float(unk_cnt),
            "unk_ratio": float(unk_cnt / max(1, len(ids))),
        }
        return toks, ids, stats

    def batch_encode(
        self,
        texts: Iterable[str],
        *,
        mode: str = "inference",
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        skip_invalid: bool = True,
        add_special_tokens: Optional[bool] = None,
    ) -> List[Tuple[List[str], List[int]]]:
        """Çoklu metni encode eder; hatalı olanları istersek atlar. GPU desteği ile optimize edilmiş."""
        out: List[Tuple[List[str], List[int]]] = []
        iw, isy, isp = self._resolve_include_flags(mode, include_whole_words, include_syllables, include_sep)
        
        # GPU batch processing
        if self.use_gpu and self.device:
            return self._batch_encode_gpu(texts, mode, iw, isy, isp, skip_invalid, add_special_tokens)
        
        # CPU processing (original)
        for t in texts:
            try:
                out.append(
                    self.encode(
                        t,
                        mode=mode,
                        include_whole_words=iw,
                        include_syllables=isy,
                        include_sep=isp,
                        add_special_tokens=add_special_tokens,
                    )
                )
            except TokenizerCoreError:
                if skip_invalid:
                    logger.debug(f"[TokenizerCore] batch_encode skip: {str(t)[:50]}")
                    continue
                raise
        return out

    def _batch_encode_gpu(
        self,
        texts: Iterable[str],
        mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        skip_invalid: bool,
        add_special_tokens: Optional[bool] = None,
    ) -> List[Tuple[List[str], List[int]]]:
        import torch
        
        text_list = list(texts)
        if not text_list:
            logger.warning("[TokenizerCore] GPU batch_encode: Boş text listesi")
            return []
        
        results = []
        
        for i in range(0, len(text_list), self.batch_size):
            batch_texts = text_list[i:i + self.batch_size]
            batch_results = []
            
            logger.debug(f"[TokenizerCore] İşlenen batch: {i}-{i+len(batch_texts)}/{len(text_list)}")
            for text in batch_texts:
                if not text.strip():
                    logger.warning(f"[TokenizerCore] GPU batch_encode: Boş metin atlandı: '{str(text)[:50]}'")
                    continue
                try:
                    tokens, token_ids = self.encode(
                        text,
                        mode=mode,
                        include_whole_words=include_whole_words,
                        include_syllables=include_syllables,
                        include_sep=include_sep,
                        add_special_tokens=add_special_tokens,
                    )
                    if token_ids:
                        tensor_ids = torch.tensor(token_ids, device=self.device)
                        batch_results.append((tokens, tensor_ids))
                        logger.debug(f"[TokenizerCore] GPU batch_encode başarılı: text='{str(text)[:50]}', token_ids_len={len(token_ids)}")
                    else:
                        logger.warning(f"[TokenizerCore] GPU batch_encode boş çıktı: text='{str(text)[:50]}'")
                        if skip_invalid:
                            continue
                        raise TokenizerCoreError(f"Boş token_ids: text='{str(text)[:50]}'")
                except TokenizerCoreError as e:
                    logger.warning(f"[TokenizerCore] GPU batch_encode hatası: text='{str(text)[:50]}', hata={e}")
                    if skip_invalid:
                        continue
                    raise
            
            # Batch sonuçlarını CPU'ya geri taşı
            for tokens, tensor_ids in batch_results:
                cpu_ids = tensor_ids.cpu().tolist()
                results.append((tokens, cpu_ids))
            
            logger.debug(f"[TokenizerCore] Batch işlendi: {len(batch_results)}/{len(batch_texts)} sonuç")
        
        logger.info(f"[TokenizerCore] GPU batch_encode tamamlandı: {len(results)}/{len(text_list)} sonuç")
        return results

    def decode(
        self,
        ids: List[int],
        *,
        method: str = "bpe",
        remove_specials: bool = True,
        remove_tags: bool = True,
        sep_token: str = "<SEP>",
        collapse_spaces: bool = True,
        lowercase: bool = False,
        prefer: Optional[str] = None,
        skip_special_tokens: Optional[bool] = None,  # Map to remove_specials
        **kwargs  # Accept additional kwargs for compatibility
    ) -> str:
        """ID → metin dönüşümü."""
        # Handle None input
        if ids is None:
            raise TokenizerCoreError("Decode: ids cannot be None")
        
        # Handle empty list gracefully
        if not ids:
            logger.debug("[TokenizerCore] decode: boş liste, boş string döndürülüyor")
            return ""
        
        # Map skip_special_tokens to remove_specials if provided
        if skip_special_tokens is not None:
            remove_specials = skip_special_tokens
        
        try:
            # Filter out invalid token IDs (out of vocab range)
            vocab_size = self.get_vocab_size()
            valid_ids = [tid for tid in ids if 0 <= tid < vocab_size]
            
            if not valid_ids:
                logger.debug("[TokenizerCore] decode: geçersiz token ID'ler, boş string döndürülüyor")
                return ""
            
            # If some IDs were invalid, log warning
            if len(valid_ids) < len(ids):
                invalid_count = len(ids) - len(valid_ids)
                logger.warning(f"[TokenizerCore] decode: {invalid_count} geçersiz token ID filtrelendi")
            
            text = self.tokenizer.decode(
                valid_ids,
                method=method,
                remove_specials=remove_specials,
                remove_tags=remove_tags,
                sep_token=sep_token,
                collapse_spaces=collapse_spaces,
                lowercase=lowercase,
                prefer=prefer,
            )
            logger.debug(
                "[TokenizerCore] decode: method=%s, len(ids)=%d, remove_specials=%s, prefer=%s",
                method, len(valid_ids), remove_specials, prefer
            )
            return text
        except (BPEDecodingError, ValueError) as e:
            # Handle ValueError from empty list in BPEManager
            if "boş liste" in str(e).lower() or "empty" in str(e).lower():
                logger.debug("[TokenizerCore] decode: boş liste hatası, boş string döndürülüyor")
                return ""
            logger.error(f"[TokenizerCore] decode hatası: {e}")
            # ÖNEMLİ: Harici testler TokenizerCoreError bekliyor
            raise TokenizerCoreError(f"decode hatası: {e}") from e
        except Exception as e:
            logger.error(f"[TokenizerCore] Beklenmeyen decode hatası: {e}")
            raise TokenizerCoreError(f"Beklenmeyen decode hatası: {e}") from e

    def batch_decode(
        self,
        sequences: Iterable[List[int]],
        *,
        method: str = "bpe",
        remove_specials: bool = True,
        remove_tags: bool = True,
        sep_token: str = "<SEP>",
        collapse_spaces: bool = True,
        lowercase: bool = False,
        skip_invalid: bool = True,
    ) -> List[str]:
        """Çoklu id dizisini decode eder; hatalı olanları istersek atlar."""
        outs: List[str] = []
        for seq in sequences:
            try:
                outs.append(
                    self.decode(
                        seq,
                        method=method,
                        remove_specials=remove_specials,
                        remove_tags=remove_tags,
                        sep_token=sep_token,
                        collapse_spaces=collapse_spaces,
                        lowercase=lowercase,
                    )
                )
            except (TokenizerCoreError, BPEDecodingError):
                if skip_invalid:
                    logger.debug(f"[TokenizerCore] batch_decode skip: ids_len={len(seq)}")
                    continue
                raise
        return outs

    # --------------------------- Eğitim Verisi Hazırlama --------------------------

    def load_training_data(
        self,
        *,
        # data loader alan isimleri (opsiyonel; geriye dönük test uyumu için kabul edilir)
        input_field: Optional[str] = None,
        target_field: Optional[str] = None,
        encode_mode: str = "train",
        include_whole_words: Optional[bool] = None,
        include_syllables: Optional[bool] = None,
        include_sep: Optional[bool] = None,
        include_source_id: bool = True,  # ✅ YENİ: source_id ekle (overlap kontrolü için)
    ) -> List[Tuple[List[int], List[int], int]]:
        """
        DataLoader → (Q,A) çiftleri → (input_ids, target_ids) eğitim örnekleri.
        Not: DataLoader mode'u QA_TRAIN olmalıdır.
        input_field / target_field argümanları şimdilik opsiyonel ve
        DataLoaderManager'a iletilmez; yalnızca harici testlerdeki TypeError'ı
        önlemek için kabul edilir.
        """
        import time
        import psutil
        
        if self.data_loader is None:
            raise TokenizerCoreError("load_training_data: DataLoader yok (data_dir/config verilmedi).")
        if self.data_loader.cfg.mode != LoadMode.QA_TRAIN:
            raise TokenizerCoreError("load_training_data: DataLoader mode 'QA_TRAIN' olmalı.")

        # Başlangıç zamanı ve bellek izleme
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"[TokenizerCore] Eğitim verisi yükleniyor...")
        logger.info(f"[TokenizerCore] Başlangıç bellek kullanımı: {initial_memory:.2f} MB")

        # 1) Ham (Q,A) çiftleri (sadece JSON'dan)
        try:
            pairs = self.data_loader.load()  # List[Tuple[str,str]]
        except (DataDirectoryNotFoundError, UnsupportedFormatError, DataLoaderError) as e:
            logger.error(f"[TokenizerCore] load_training_data yükleme hatası: {e}")
            raise TokenizerCoreError(f"load_training_data: {e}")
        
        # 2) RAW dataları ayrı yükle (DOCX/TXT'ten) - ✅ file_index ile
        raw_data_loader = DataLoaderManager(
            DataLoaderConfig(
                data_dir=self.data_loader.cfg.data_dir,
                mode=LoadMode.RAW_TEXT,  # RAW text mode
                max_tokens_per_text=self.data_loader.cfg.max_tokens_per_text,
                overlap_tokens=self.data_loader.cfg.overlap_tokens,
            )
        )
        
        try:
            # ✅ YENİ: file_index ile yükle (overlap önleme için)
            if hasattr(raw_data_loader, 'load_with_file_index'):
                raw_texts_with_file_idx = raw_data_loader.load_with_file_index()  # List[Tuple[str, int]]
                logger.info(f"[TokenizerCore] RAW datalar yüklendi (file_index ile): {len(raw_texts_with_file_idx)} chunk")
            else:
                # Fallback: Eski metod (geriye dönük uyumluluk)
                logger.warning(f"[TokenizerCore] UYARI: DataLoader file_index desteklemiyor (overlap olabilir)!")
                raw_texts = raw_data_loader.load()  # List[str]
                raw_texts_with_file_idx = [(text, idx) for idx, text in enumerate(raw_texts)]
                logger.info(f"[TokenizerCore] RAW datalar yüklendi (fallback): {len(raw_texts_with_file_idx)} chunk")
        except Exception as e:
            logger.warning(f"[TokenizerCore] RAW data yükleme hatası: {e}")
            raw_texts_with_file_idx = []
        
        after_load_memory = process.memory_info().rss / 1024 / 1024
        qa_pairs = len(pairs)
        raw_chunks = len(raw_texts_with_file_idx)  # ✅ DÜZELTME: raw_texts_with_file_idx kullan
        logger.info(f"[TokenizerCore] {qa_pairs:,} örnek yüklendi (QA:{qa_pairs}, RAW:{raw_chunks})")
        logger.info(f"[TokenizerCore] Veri yükleme sonrası bellek: {after_load_memory:.2f} MB (+{after_load_memory-initial_memory:.2f} MB)")

        # NOT: Vocab genişletme kaldırıldı - Sabit vocab stratejisi için vocab eğitim sırasında değişmemeli!
        # Vocab ve merges BPE training sırasında oluşturulmuş olmalı (train_bpe.py ile)
        # Eğitim sırasında vocab genişletme yapılmaz - bu "kelime salatası" sorununa neden oluyor!
        logger.info(f"[TokenizerCore] Vocab genişletme ATLANDI - Sabit vocab stratejisi (vocab boyutu: {len(self.tokenizer.get_vocab()):,})")

        # Vocab/merges senkronizasyonu (vocab genişletmeden SONRA değil, sadece yükleme için)
        # NOT: finalize_vocab() sadece vocab/merges yükler ve base alphabet ekler (vocab genişletmez)
        self.finalize_vocab()  # sample_texts=None geçilmiyor - vocab genişletme yapmaz
        valid_ids = set(get_valid_ids(self.tokenizer.get_vocab()))
        logger.debug(f"[TokenizerCore] Vocab boyutu: {len(valid_ids)}")

        # 4) Encode (vocab büyümesin diye inference modu ile)
        iw_enc, isy_enc, isp_enc = self._resolve_include_flags(encode_mode, include_whole_words, include_syllables, include_sep)

        # Tüm encoding verilerini hazırla: QA çiftleri + RAW text chunks
        all_encoding_data = []
        qa_examples: List[Tuple[List[int], List[int]]] = []  # [OK] QA çiftleri için token ID'leri (ayrı işlenecek)
        
        # [OK] DÜZELTME: QA çiftlerini ayrı encode et, <SEP> token ile birleştir
        logger.info(f"[TokenizerCore] QA çiftleri ayrı encode ediliyor... ({len(pairs)} çift)")
        qa_skipped = 0
        for q, a in pairs:
            # ✅ Boş text kontrolü - encode çağrısından ÖNCE
            if not q or not a or (not q.strip() and not a.strip()):
                logger.warning(f"[TokenizerCore] Boş QA çifti atlandı (q='{q[:20] if q else ''}...', a='{a[:20] if a else ''}...')")
                qa_skipped += 1
                continue
            
            if q == "" and a != "":
                # Bu raw text chunk - sadece cevabı kullan
                all_encoding_data.append(("raw", a))
            else:
                # [OK] DÜZELTME: Soru ve cevabı ayrı encode et
                try:
                    # Soru ve cevabı ayrı encode et
                    q_tokens, q_ids = self.encode(
                        q,
                        mode="train",
                        include_whole_words=iw_enc,
                        include_syllables=isy_enc,
                        include_sep=False,  # [OK] SEP manuel ekleyeceğiz
                        add_special_tokens=False
                    )
                    
                    a_tokens, a_ids = self.encode(
                        a,
                        mode="train",
                        include_whole_words=iw_enc,
                        include_syllables=isy_enc,
                        include_sep=False,  # [OK] SEP manuel ekleyeceğiz
                        add_special_tokens=False
                    )
                    
                    # ✅ Boş sequence kontrolü - encode sonrası da kontrol et
                    if not q_ids or not a_ids:
                        logger.warning(f"[TokenizerCore] Boş QA sequence atlandı (q={len(q_ids)}, a={len(a_ids)})")
                        qa_skipped += 1
                        continue
                    
                    # ✅ ENDÜSTRİ STANDARDI: SEP token ID'sini al (soru ve cevap arasına eklenir)
                    special_ids = self._special_ids()
                    sep_id = special_ids.get("<SEP>")
                    
                    if sep_id is None:
                        # Fallback: DEFAULT_SPECIALS kullan
                        from tokenizer_management.bpe.bpe_manager_utils import DEFAULT_SPECIALS
                        sep_id = DEFAULT_SPECIALS.get("<SEP>", 4)
                        logger.warning(f"[TokenizerCore] SEP token ID vocab'ta bulunamadı, fallback kullanılıyor: {sep_id}")
                    
                    # ✅ ENDÜSTRİ STANDARDI: QA formatı [q_ids, SEP_ID, a_ids]
                    # SEP_ID QA'yı ayırt etmek için yeterli (RAW text'te SEP_ID yok)
                    # PAD token sadece padding için kullanılır (formatlama sırasında eklenir)
                    inp_ids = q_ids + [sep_id] + a_ids
                    tgt_ids = q_ids + [sep_id] + a_ids  # Target = Input (autoregressive)
                    
                    # QA örneği olarak ekle
                    qa_examples.append((inp_ids, tgt_ids))
                    
                except Exception as e:
                    logger.warning(f"[TokenizerCore] QA çifti encode hatası: {e}")
                    qa_skipped += 1
                    continue
        
        # ✅ RAW text chunks'ları ekle (file_index ile)
        for raw_text, file_idx in raw_texts_with_file_idx:
            all_encoding_data.append(("raw", raw_text, file_idx))  # ✅ file_idx ekle
        
        logger.info(f"[TokenizerCore] Encoding başlıyor - QA:{len(qa_examples)} (atlandı:{qa_skipped}), RAW:{len(all_encoding_data)}")
        examples: List[Tuple[List[int], List[int], int]] = []  # ✅ DEĞİŞTİ: source_id eklendi
        skipped = 0
        
        # ✅ YENİ: source_id counter (overlap kontrolü için)
        current_source_id = 0
        
        # Progress tracking
        total_examples = len(qa_examples) + len(all_encoding_data)  # [OK] QA + RAW
        processed_examples = 0
        last_log_time = time.time()
        log_interval = 3.0  # Her 3 saniyede bir log
        
        # [OK] Uzun chunk kontrolü için değişkenler
        max_seq_length = self.config.get("max_seq_length", 768)
        max_tokens_for_chunk = max_seq_length - 2  # BOS ve EOS için alan bırak (512 - 2 = 510)
        long_chunks_count = 0  # Uzun chunk sayacı

        # GPU batch encoding kullan
        logger.info(f"[TokenizerCore] GPU batch encoding başlıyor... (use_gpu={self.use_gpu})")
        
        # [OK] ENDÜSTRI STANDARDI: Training data için BOS/EOS ekleme
        # add_special_tokens=False → BOS/EOS eklenmez (DataLoader'da eklenecek)
        # ✅ DÜZELTME: all_encoding_data artık 3 elemanlı (data_type, text, file_idx)
        all_texts = []
        for item in all_encoding_data:
            if len(item) == 3:
                _, text, _ = item  # (data_type, text, file_idx)
            else:
                _, text = item  # (data_type, text) - fallback
            all_texts.append(text)
        
        # Sadece QA kullanıldığında RAW=0 → all_texts boş; batch_encode atlanır (uyarı önlenir)
        if all_texts:
            batch_results = self.batch_encode(
                all_texts,
                mode="train",  # [OK] train mode kullan
                include_whole_words=iw_enc,
                include_syllables=isy_enc,
                include_sep=isp_enc,
                skip_invalid=True,
                add_special_tokens=False  # [OK] BOS/EOS ekleme (DataLoader'da eklenecek)
            )
        else:
            batch_results = []
        
        # [OK] ENDÜSTRI STANDARDI: Autoregressive format için target oluşturma
        # Sonuçları işle
        max_seq_length = self.config.get("max_seq_length", 768)
        max_tokens_for_chunk = max_seq_length - 2  # BOS ve EOS için alan bırak (512 - 2 = 510)
        long_chunks_count = 0  # Uzun chunk sayacı
        
        for i, item in enumerate(all_encoding_data):
            # ✅ DÜZELTME: file_idx desteği ekle
            if len(item) == 3:
                data_type, text_to_encode, file_idx = item  # ✅ file_idx var
            else:
                data_type, text_to_encode = item  # Eski format (fallback)
                file_idx = i  # Fallback: chunk index
            
            if i >= len(batch_results):
                logger.warning(f"[TokenizerCore] Batch sonucu eksik, {data_type} örneği atlandı")
                skipped += 1
                continue
                
            try:
                tokens, inp_ids = batch_results[i]
                
                # [OK] KRİTİK DÜZELTME: Gerçek token sayısını kontrol et ve yeniden böl
                # Eğer chunk çok uzunsa (max_seq_length'den uzun), token ID'lerine göre böl
                # Bu, veri kaybını önler
                if len(inp_ids) > max_tokens_for_chunk:
                    long_chunks_count += 1
                    if long_chunks_count <= 5:  # İlk 5 uzun chunk için detaylı log
                        logger.warning(
                            f"[TokenizerCore] ⚠️  Chunk çok uzun! {len(inp_ids)} token > {max_tokens_for_chunk} "
                            f"(max_seq_length={max_seq_length}). Yeniden bölünüyor..."
                        )
                    
                    # [OK] Uzun chunk'ı token ID'lerine göre böl (veri kaybı yok!)
                    # Overlap ekle (son N token'ı yeni chunk'a kopyala - bağlam korunur)
                    overlap_tokens = 20  # Son 20 token'ı yeni chunk'a kopyala
                    split_chunks = self._split_token_ids_by_length(
                        inp_ids,
                        max_length=max_tokens_for_chunk,
                        overlap=overlap_tokens
                    )
                    
                    # ✅ DÜZELTME: Uzun chunk split - file_idx kullan (dosya bazlı source_id)
                    # Aynı dosyanın split chunk'ları aynı source_id ile
                    chunk_source_id = file_idx  # ✅ DOCUMENT-level source_id
                    
                    for split_inp_ids in split_chunks:
                        if not split_inp_ids or len(split_inp_ids) == 0:
                            continue
                        split_tgt_ids = list(split_inp_ids)
                        examples.append((split_inp_ids, split_tgt_ids, chunk_source_id))  # ✅ file_idx kullan
                        processed_examples += 1
                    
                    continue  # Bu chunk işlendi, bir sonrakine geç
                
                # [OK] ENDÜSTRI STANDARDI (GPT-2/3/4): Autoregressive Training Format
                # BOS/EOS zaten eklenmedi (add_special_tokens=False kullandık)
                # 
                # TokenizerCore'dan gelen:
                #   inp_ids = [t1, t2, t3, ..., tN]  (BOS/EOS yok)
                #
                # TrainingService'te eklenecek:
                #   Input:  [BOS, t1, t2, t3, ..., tN]
                #   Target: [t1, t2, t3, ..., tN, EOS]
                #
                # Model öğrenir:
                #   BOS → t1, t1 → t2, t2 → t3, ..., tN → EOS
                #
                # Bu yüzden burada:
                #   inp_ids = [t1, t2, t3, ..., tN]
                #   tgt_ids = [t1, t2, t3, ..., tN]  (AYNI! TrainingService'te alignment yapılacak)
                
                # [OK] Boş sequence kontrolü
                if not inp_ids or len(inp_ids) == 0:
                    logger.warning(f"[TokenizerCore] Boş sequence, {data_type} örneği atlandı")
                    skipped += 1
                    continue
                
                # [OK] Target = Input (TrainingService'te BOS/EOS eklenecek ve alignment yapılacak)
                tgt_ids = list(inp_ids)
                
            except Exception as e:
                logger.warning(f"[TokenizerCore] Batch sonucu işleme hatası, {data_type} örneği atlandı: {str(e)}")
                skipped += 1
                continue

            if any(tid not in valid_ids for tid in inp_ids) or any(tid not in valid_ids for tid in tgt_ids):
                logger.warning(f"[TokenizerCore] OOB id tespit edildi, {data_type} örneği atlandı.")
                skipped += 1
                continue

            # ✅ DÜZELTME: source_id = file_idx (DOCUMENT-level, overlap önleme için)
            # RAW text için dosya bazlı source_id kullan
            source_id = file_idx  # ✅ Aynı dosyanın chunk'ları aynı source_id
            examples.append((inp_ids, tgt_ids, source_id))
            processed_examples += 1
            
            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                progress_pct = (processed_examples / total_examples) * 100
                current_memory = process.memory_info().rss / 1024 / 1024
                elapsed = current_time - start_time
                
                logger.info(f"[TokenizerCore] Encoding: {processed_examples:,}/{total_examples:,} "
                           f"({progress_pct:.1f}%) | "
                           f"Başarılı: {len(examples):,} | "
                           f"Atlanan: {skipped:,} | "
                           f"Bellek: {current_memory:.1f}MB | "
                           f"Süre: {elapsed:.1f}s")
                last_log_time = current_time

        # [OK] Uzun chunk uyarısı
        if long_chunks_count > 0:
            logger.info(
                f"[TokenizerCore] ℹ️  {long_chunks_count} RAW chunk max_seq_length ({max_seq_length})'den uzun! "
                f"Otomatik yeniden bölündü (veri kaybı yok). "
                f"DataLoaderManager'da max_tokens_per_text={self.data_loader.cfg.max_tokens_per_text} "
                f"(token tahmini iyileştirilebilir)"
            )
        
        # [OK] QA çiftlerini ekle (validation ile)
        logger.info(f"[TokenizerCore] QA çiftleri ekleniyor... ({len(qa_examples)} örnek)")
        qa_long_chunks_count = 0  # QA uzun chunk sayacı
        
        for inp_ids, tgt_ids in qa_examples:
            # Validation
            if not inp_ids or len(inp_ids) == 0:
                logger.warning(f"[TokenizerCore] Boş QA sequence atlandı")
                skipped += 1
                continue
            
            if any(tid not in valid_ids for tid in inp_ids) or any(tid not in valid_ids for tid in tgt_ids):
                logger.warning(f"[TokenizerCore] OOB id tespit edildi, QA örneği atlandı.")
                skipped += 1
                continue
            
            # [OK] KRİTİK: QA çiftleri için de uzunluk kontrolü (veri kaybını önle!)
            if len(inp_ids) > max_tokens_for_chunk:
                qa_long_chunks_count += 1
                if qa_long_chunks_count <= 5:  # İlk 5 uzun QA chunk için detaylı log
                    logger.warning(
                        f"[TokenizerCore] ⚠️  QA chunk çok uzun! {len(inp_ids)} token > {max_tokens_for_chunk} "
                        f"(max_seq_length={max_seq_length}). Yeniden bölünüyor..."
                    )
                
                # [OK] Uzun QA chunk'ı token ID'lerine göre böl (veri kaybı yok!)
                overlap_tokens = 20  # Son 20 token'ı yeni chunk'a kopyala (bağlam korunur)
                split_chunks = self._split_token_ids_by_length(
                    inp_ids,
                    max_length=max_tokens_for_chunk,
                    overlap=overlap_tokens
                )
                
                # ✅ DÜZELTME: Uzun QA split - unique source_id (offset ile)
                # QA split chunk'ları için aynı source_id (aynı QA'dan geliyorlar)
                qa_source_id = 1000000 + current_source_id  # ✅ Offset: QA source_id'leri 1M'den başlar
                current_source_id += 1
                
                for split_inp_ids in split_chunks:
                    if not split_inp_ids or len(split_inp_ids) == 0:
                        continue
                    split_tgt_ids = list(split_inp_ids)  # Target = Input (autoregressive)
                    examples.append((split_inp_ids, split_tgt_ids, qa_source_id))  # ✅ Aynı QA için aynı source_id
                    processed_examples += 1
                
                continue  # Bu QA chunk işlendi, bir sonrakine geç
            
            # ✅ DÜZELTME: QA chunk için unique source_id
            # QA çiftleri: Her QA ayrı bir "dokuman" gibi davranılır
            # Her QA için benzersiz source_id (offset ile file_idx'lerle çakışmayı önle)
            qa_source_id = 1000000 + current_source_id  # ✅ Offset: QA source_id'leri 1M'den başlar
            examples.append((inp_ids, tgt_ids, qa_source_id))
            current_source_id += 1
            processed_examples += 1
        
        # ✅ DEDUPLICATION: Aynı source_id içinde duplicate'leri filtrele
        logger.info(f"[TokenizerCore] Deduplication kontrolü yapılıyor...")
        import hashlib
        deduplicated_examples = []
        seen_hashes = {}  # {source_id: {hash}}
        duplicate_count = 0
        
        for inp_ids, tgt_ids, source_id in examples:
            # Hash oluştur (inp_ids bazlı, PAD token yok zaten)
            seq_str = str(inp_ids)
            seq_hash = hashlib.sha256(seq_str.encode()).hexdigest()
            
            # source_id bazlı hash kontrolü
            if source_id not in seen_hashes:
                seen_hashes[source_id] = set()
            
            if seq_hash not in seen_hashes[source_id]:
                seen_hashes[source_id].add(seq_hash)
                deduplicated_examples.append((inp_ids, tgt_ids, source_id))
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.warning(
                f"[TokenizerCore] ⚠️  {duplicate_count:,} duplicate örnek filtrelendi "
                f"({duplicate_count/len(examples)*100:.2f}%) - Aynı source_id içinde duplicate'ler temizlendi"
            )
        else:
            logger.info(f"[TokenizerCore] ✅ Duplicate örnek yok!")
        
        examples = deduplicated_examples
        
        # QA uzun chunk uyarısı
        if qa_long_chunks_count > 0:
            logger.warning(
                f"[TokenizerCore] ⚠️  {qa_long_chunks_count} QA chunk max_seq_length ({max_seq_length})'den uzun! "
                f"Yeniden bölündü (veri kaybı yok)."
            )
            processed_examples += 1

        # Final rapor
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # [OK] Total examples güncelle (QA + RAW)
        total_examples = len(qa_examples) + len(all_encoding_data)
        
        logger.info("=" * 50)
        logger.info("EĞİTİM VERİSİ HAZIRLAMA TAMAMLANDI!")
        logger.info("=" * 50)
        logger.info(f"Toplam süre: {total_time:.2f} saniye ({total_time/60:.1f} dakika)")
        logger.info(f"İşlenen toplam örnek: {total_examples:,} (QA:{len(qa_examples):,}, RAW:{len(all_encoding_data):,})")
        logger.info(f"Başarılı örnek: {len(examples):,}")
        logger.info(f"Atlanan örnek: {skipped + qa_skipped:,} (QA:{qa_skipped}, RAW:{skipped})")
        logger.info(f"Başarı oranı: {(len(examples)/total_examples*100):.1f}%")
        logger.info(f"Final bellek kullanımı: {final_memory:.2f} MB (+{memory_increase:.2f} MB)")
        logger.info(f"Ortalama hız: {total_examples/total_time:.1f} örnek/saniye")
        logger.info("=" * 50)
        
        return examples

    # --------------------------------- Yardımcılar --------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Pretokenizer → Syllabifier → (opsiyonel) temizlik; diagnostik amaçlı."""
        return self.tokenizer.tokenize(text)

    def _split_token_ids_by_length(
        self,
        token_ids: List[int],
        max_length: int,
        overlap: int = 20
    ) -> List[List[int]]:
        """
        Token ID listesini belirli bir uzunluğa göre böler (overlap ile)
        
        Args:
            token_ids: Bölünecek token ID listesi
            max_length: Her chunk'ın maksimum uzunluğu
            overlap: Chunk'lar arası örtüşme (token sayısı) - önceki chunk'ın son N token'ı
        
        Returns:
            Bölünmüş chunk'lar listesi
        """
        if not token_ids or len(token_ids) <= max_length:
            return [token_ids] if token_ids else []
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(token_ids):
            # Chunk'ın son indeksini belirle
            end_idx = min(start_idx + max_length, len(token_ids))
            chunk = token_ids[start_idx:end_idx]
            
            if chunk:
                chunks.append(chunk)
            
            # Overlap için: Bir sonraki chunk'ın başlangıcı, bu chunk'ın sonundan overlap kadar geri
            # Örnek: max_length=510, overlap=20
            # Chunk 1: [0:510] → Chunk 2: [490:1000] (490 = 510 - 20)
            if end_idx < len(token_ids):
                # Overlap: Son chunk'ın son N token'ını yeni chunk'ın başına ekle
                start_idx = end_idx - overlap
                # Güvenlik: start_idx negatif olamaz
                if start_idx < 0:
                    start_idx = 0
            else:
                break
        
        return chunks
    
    def get_vocab(self) -> Dict[str, Any]:
        """In-memory vocab bilgisi."""
        try:
            return self.tokenizer.get_vocab()
        except BPETokenError as e:
            logger.error(f"[TokenizerCore] get_vocab hatası: {e}")
            raise

    def get_merges(self) -> List[Tuple[str, str]]:
        """In-memory merges listesi."""
        try:
            return self.tokenizer.get_merges()
        except BPETokenError as e:
            logger.error(f"[TokenizerCore] get_merges hatası: {e}")
            raise

    def set_vocab(self, vocab: Dict[str, Any]) -> None:
        """Vocab'ı dışarıdan set et ve tüm bileşenlerle senkronize et."""
        try:
            self.tokenizer.set_vocab(vocab)
            self._cached_special_ids = None
        except Exception as e:
            logger.error(f"[TokenizerCore] set_vocab hatası: {e}")
            raise

    def get_vocab_size(self) -> int:
        """Modelin giriş/çıkış boyutu (embedding/linear) için kullanılır."""
        size = len(self.get_vocab())
        logger.debug(f"[TokenizerCore] Vocab size: {size}")
        return size

    def auto_update_vocab(self, tokens: List[str]) -> int:
        """Ham tokenları sözlüğe (monotonik id ile) ekler. Dönüş: eklenen sayısı. (merges değişmez)"""
        try:
            added = self.tokenizer.auto_update_vocab(tokens)
            if added:
                self._cached_special_ids = None
            return added
        except Exception as e:
            logger.error(f"[TokenizerCore] auto_update_vocab hatası: {e}")
            raise

    def summary(self) -> Dict[str, Any]:
        """Hızlı durum özeti."""
        v = self.get_vocab()
        merges = self.get_merges()
        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        return {
            "vocab_size": len(v),
            "merges_count": len(merges),
            "has_specials": all(s in v for s in specials),
            "vocab_path": self.vocab_path,
            "merges_path": self.merges_path,
        }
