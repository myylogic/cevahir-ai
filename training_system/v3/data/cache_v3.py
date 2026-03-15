# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cache_v3.py
Modül: training_system/v3/data
Görev: DataCache V3 - Strict Cache Mode. Cache olmadan eğitim BAŞLAMAZ.
       trainbpe → prepare_cache → train.py akışı zorunlu ve izoledir.
       Cache bulunamadığında anlamlı hata mesajıyla erken çıkış yapar.

MİMARİ:
- SOLID: Single Responsibility (strict cache yönetimi)
- Design Patterns: Cache Pattern + Fail-Fast Pattern
- Endüstri Standartları: MLOps pipeline isolation

KRİTİK DEĞİŞİKLİKLER (V2 → V3):
- allow_cache_key_mismatch: False (default) → hatalı cache key kabul edilmez
- allow_data_hash_mismatch: False (default) → değişmiş veri ile cache kabul edilmez
- Cache yoksa HATA fırlatır (raw data işleme YOK)
- CacheNotFoundError: Açıklayıcı hata mesajı ile erken çıkış
- Cache metadata: Human-readable JSON metadata dosyası
- Cache integrity: Her cache dosyası için SHA-256 checksum

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import os
import json
import hashlib
import pickle
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# HATA SINIFI
# ────────────────────────────────────────────────────────────────────────────

class CacheNotFoundError(RuntimeError):
    """Cache bulunamadı — eğitim başlatılamaz."""
    pass


class CacheIntegrityError(RuntimeError):
    """Cache bütünlüğü bozuk — checksum uyuşmuyor."""
    pass


# ────────────────────────────────────────────────────────────────────────────
# CACHE V3 — STRICT MODE
# ────────────────────────────────────────────────────────────────────────────

class DataCacheV3:
    """
    DataCache V3 — Strict Cache Manager.

    V2 farkları:
    - Cache yoksa HATA FIRLATIRIR (raw data işleme yok)
    - allow_cache_key_mismatch=False (kesin eşleşme zorunlu)
    - allow_data_hash_mismatch=False (veri değişikliği kabul edilmez)
    - Cache metadata JSON dosyası (human-readable)
    - SHA-256 checksum ile cache bütünlük doğrulama

    Kullanım akışı (ZORUNLU):
        1. python tokenizer_management/train_bpe.py
        2. python training_system/prepare_cache.py
        3. python training_system/train.py

    Adım 3, adım 2 çıktısı olmadan BAŞLAMAZ.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str = ".cache/preprocessed_data",
        cache_enabled: bool = True,
        strict_mode: bool = True,           # V3: Strict mode (cache zorunlu)
        verify_integrity: bool = True,      # V3: Checksum doğrulama
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.strict_mode = strict_mode
        self.verify_integrity = verify_integrity

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CacheV3] Cache dizini: {self.cache_dir} (strict={strict_mode})")

    # ──────────────────────────────────────────────────────────────────────
    # HASH HESAPLAMA
    # ──────────────────────────────────────────────────────────────────────

    def _get_data_dir_hash(self) -> str:
        """Eğitim verisi dizininin içerik hash'i (dosya adı + boyut)."""
        if not self.data_dir.exists():
            return ""

        file_infos = []
        for ext in [".json", ".txt", ".docx"]:
            for fp in sorted(self.data_dir.rglob(f"*{ext}")):
                if fp.is_file():
                    try:
                        rel = os.path.relpath(fp, self.data_dir)
                    except ValueError:
                        rel = fp.name
                    file_infos.append(f"{rel}:{fp.stat().st_size}")

        if not file_infos:
            return ""

        combined = "|".join(sorted(file_infos))
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _get_vocab_hash(self, tokenizer_core) -> str:
        """Vocab hash (vocab değişirse cache invalid)."""
        vocab = tokenizer_core.get_vocab()
        vocab_items = []
        for token, data in vocab.items():
            if isinstance(data, dict):
                vocab_items.append(f"{token}:{data.get('id', 0)}")
            elif isinstance(data, int):
                vocab_items.append(f"{token}:{data}")
        vocab_str = "|".join(sorted(vocab_items))
        return hashlib.md5(vocab_str.encode()).hexdigest()[:16]

    def _normalize_data_dir(self) -> str:
        """data_dir'ı normalize et (relative path uyumluluğu)."""
        data_dir_str = str(self.data_dir)
        try:
            if os.path.isabs(data_dir_str):
                cwd = os.getcwd()
                try:
                    data_dir_str = os.path.relpath(data_dir_str, cwd)
                except ValueError:
                    pass
        except Exception:
            pass
        return data_dir_str

    def get_cache_key(
        self,
        encode_mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        max_seq_length: int,
        vocab_hash: str,
        alignment_format: str = "autoregressive_v2",
        formatted: bool = True,
    ) -> str:
        """Cache key oluştur — parametrelere göre deterministik."""
        key_parts = [
            self._normalize_data_dir(),
            encode_mode,
            str(include_whole_words),
            str(include_syllables),
            str(include_sep),
            str(max_seq_length),
            vocab_hash,
            alignment_format,
            f"formatted_{formatted}",
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, data_hash: str) -> Path:
        return self.cache_dir / f"cached_data_{cache_key}_{data_hash}.pkl"

    def _get_metadata_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(".meta.json")

    def _get_checksum_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(".sha256")

    # ──────────────────────────────────────────────────────────────────────
    # CHECKSUM
    # ──────────────────────────────────────────────────────────────────────

    def _compute_file_checksum(self, path: Path) -> str:
        """Dosyanın SHA-256 checksum'unu hesapla."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _save_checksum(self, cache_path: Path) -> None:
        """Cache dosyası için checksum kaydet."""
        checksum = self._compute_file_checksum(cache_path)
        self._get_checksum_path(cache_path).write_text(checksum)

    def _verify_checksum(self, cache_path: Path) -> bool:
        """Cache bütünlüğünü doğrula."""
        checksum_path = self._get_checksum_path(cache_path)
        if not checksum_path.exists():
            logger.debug(f"[CacheV3] Checksum dosyası yok: {checksum_path.name}")
            return True  # Eski format — checksum yok, geç

        saved = checksum_path.read_text().strip()
        actual = self._compute_file_checksum(cache_path)
        return saved == actual

    # ──────────────────────────────────────────────────────────────────────
    # METADATA
    # ──────────────────────────────────────────────────────────────────────

    def _save_metadata(
        self,
        cache_path: Path,
        cache_key: str,
        data_hash: str,
        encode_mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        max_seq_length: int,
        alignment_format: str,
        sample_count: int,
        file_size_mb: float,
    ) -> None:
        """Human-readable metadata JSON dosyası oluştur."""
        meta = {
            "version": "v3",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_key": cache_key,
            "data_hash": data_hash,
            "data_dir": str(self.data_dir),
            "encode_mode": encode_mode,
            "include_whole_words": include_whole_words,
            "include_syllables": include_syllables,
            "include_sep": include_sep,
            "max_seq_length": max_seq_length,
            "alignment_format": alignment_format,
            "sample_count": sample_count,
            "file_size_mb": round(file_size_mb, 2),
        }
        meta_path = self._get_metadata_path(cache_path)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info(f"[CacheV3] Metadata kaydedildi: {meta_path.name}")

    # ──────────────────────────────────────────────────────────────────────
    # STRICT CACHE LOAD
    # ──────────────────────────────────────────────────────────────────────

    def load_strict(
        self,
        cache_key: str,
        data_hash: str,
    ) -> List[Tuple]:
        """
        Cache'den veriyi yükle — STRICT MODE.

        Cache bulunamazsa CacheNotFoundError fırlatır.
        V2'deki fallback mekanizması YOK.

        Args:
            cache_key: Cache anahtarı (MD5)
            data_hash: Veri dizini hash'i

        Returns:
            Formatlanmış veri listesi

        Raises:
            CacheNotFoundError: Cache dosyası bulunamadı
            CacheIntegrityError: Checksum uyuşmuyor
        """
        if not self.cache_enabled:
            raise CacheNotFoundError(
                "[CacheV3] Cache devre dışı (cache_enabled=False)!\n"
                "  Lütfen cache_enabled=True yapın ve prepare_cache.py çalıştırın."
            )

        cache_path = self._get_cache_path(cache_key, data_hash)

        if cache_path.exists():
            # Checksum doğrula
            if self.verify_integrity:
                if not self._verify_checksum(cache_path):
                    raise CacheIntegrityError(
                        f"[CacheV3] Cache bütünlüğü bozuk: {cache_path.name}\n"
                        f"  Cache'i silin ve yeniden oluşturun:\n"
                        f"  python training_system/prepare_cache.py"
                    )
            return self._load_cache_file(cache_path)

        # Tam eşleşme yok — strict mode'da HATA FIRLAT
        existing = list(self.cache_dir.glob("cached_data_*.pkl")) if self.cache_dir.exists() else []
        self._raise_cache_not_found(cache_key, data_hash, existing)

    def _load_cache_file(self, cache_path: Path) -> List[Tuple]:
        """Cache dosyasını yükle."""
        logger.info(f"[CacheV3] Yükleniyor: {cache_path.name}")
        start = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        elapsed = time.time() - start
        logger.info(f"[CacheV3] Yüklendi: {len(data):,} örnek, {elapsed:.2f}s")
        return data

    def _raise_cache_not_found(
        self,
        cache_key: str,
        data_hash: str,
        existing: List[Path],
    ) -> None:
        """CacheNotFoundError fırlat — detaylı açıklama ile."""
        lines = [
            "",
            "=" * 70,
            "[CacheV3] HATA: Cache bulunamadı! Eğitim başlatılamaz.",
            "=" * 70,
            "",
            f"  Aranan cache_key : {cache_key[:16]}...",
            f"  Aranan data_hash : {data_hash}",
            f"  Cache dizini     : {self.cache_dir}",
            "",
        ]

        if not existing:
            lines += [
                "  Cache dizininde hiç .pkl dosyası bulunamadı.",
                "",
            ]
        else:
            lines += [
                f"  Mevcut cache dosyaları ({len(existing)} adet):",
            ]
            for f in existing[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                lines.append(f"    - {f.name} ({size_mb:.2f} MB)")
                # Metadata varsa göster
                meta_path = self._get_metadata_path(f)
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        lines.append(
                            f"      max_seq={meta.get('max_seq_length')}, "
                            f"samples={meta.get('sample_count'):,}, "
                            f"created={meta.get('created_at')}"
                        )
                    except Exception:
                        pass
            if len(existing) > 5:
                lines.append(f"    ... ve {len(existing) - 5} dosya daha")
            lines.append("")
            lines += [
                "  Cache key uyuşmuyor — bu durum şu sebeplerden olabilir:",
                "    1. max_seq_length veya encode parametreleri değişti",
                "    2. vocab_hash değişti (vocab dosyası güncellendi)",
                "    3. alignment_format değişti",
                "    4. data_hash değişti (eğitim verisi değişti)",
                "",
            ]

        lines += [
            "  ÇÖZÜM: Cache'i yeniden oluşturun:",
            "    python training_system/prepare_cache.py",
            "",
            "  Eğitim akışı (ZORUNLU SIRA):",
            "    1. python tokenizer_management/train_bpe.py   [BPE eğitimi]",
            "    2. python training_system/prepare_cache.py    [Cache hazırlama]",
            "    3. python training_system/train.py            [Model eğitimi]",
            "",
            "=" * 70,
        ]

        raise CacheNotFoundError("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────────────────────

    def save(
        self,
        cache_key: str,
        data_hash: str,
        data: List[Tuple],
        encode_mode: str = "train",
        include_whole_words: bool = True,
        include_syllables: bool = False,
        include_sep: bool = False,
        max_seq_length: int = 512,
        alignment_format: str = "autoregressive_v2",
    ) -> bool:
        """
        Cache'e kaydet (atomic write + checksum + metadata).

        Args:
            cache_key: Cache anahtarı
            data_hash: Veri dizini hash'i
            data: Formatlanmış veri listesi
            ...diğer parametreler: Metadata için

        Returns:
            True ise başarılı
        """
        if not self.cache_enabled:
            return False

        cache_path = self._get_cache_path(cache_key, data_hash)

        try:
            logger.info(f"[CacheV3] Kaydediliyor: {cache_path.name}")
            start = time.time()

            # Atomic write (temp → rename)
            temp_path = cache_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(cache_path)

            elapsed = time.time() - start
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"[CacheV3] Kaydedildi: {len(data):,} örnek, "
                f"{file_size_mb:.2f} MB, {elapsed:.2f}s"
            )

            # Checksum
            if self.verify_integrity:
                self._save_checksum(cache_path)
                logger.info(f"[CacheV3] Checksum kaydedildi")

            # Metadata
            self._save_metadata(
                cache_path=cache_path,
                cache_key=cache_key,
                data_hash=data_hash,
                encode_mode=encode_mode,
                include_whole_words=include_whole_words,
                include_syllables=include_syllables,
                include_sep=include_sep,
                max_seq_length=max_seq_length,
                alignment_format=alignment_format,
                sample_count=len(data),
                file_size_mb=file_size_mb,
            )

            return True

        except Exception as e:
            logger.error(f"[CacheV3] Kaydetme hatası: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # CONVENIENCE
    # ──────────────────────────────────────────────────────────────────────

    def load_for_training(
        self,
        tokenizer_core,
        encode_mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        max_seq_length: int,
        alignment_format: str = "autoregressive_v2",
    ) -> List[Tuple]:
        """
        Eğitim için cache'den veri yükle.

        Strict mode: Cache yoksa CacheNotFoundError fırlatır.

        Returns:
            Formatlanmış veri listesi
        """
        vocab_hash = self._get_vocab_hash(tokenizer_core)
        data_hash = self._get_data_dir_hash()

        cache_key = self.get_cache_key(
            encode_mode=encode_mode,
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
            max_seq_length=max_seq_length,
            vocab_hash=vocab_hash,
            alignment_format=alignment_format,
            formatted=True,
        )

        logger.info(
            f"[CacheV3] Cache yükleniyor:\n"
            f"  cache_key={cache_key[:16]}..., data_hash={data_hash}"
        )

        return self.load_strict(cache_key, data_hash)

    def clear(self) -> int:
        """Tüm cache dosyalarını sil (pkl, meta.json, sha256)."""
        if not self.cache_dir.exists():
            return 0

        deleted = 0
        for pattern in ["cached_data_*.pkl", "cached_data_*.meta.json", "cached_data_*.sha256"]:
            for f in self.cache_dir.glob(pattern):
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"[CacheV3] Silinemedi: {f.name} — {e}")

        logger.info(f"[CacheV3] {deleted} cache dosyası silindi")
        return deleted

    def list_caches(self) -> List[Dict[str, Any]]:
        """Mevcut cache dosyalarını listele (metadata ile)."""
        if not self.cache_dir.exists():
            return []

        result = []
        for pkl_path in sorted(self.cache_dir.glob("cached_data_*.pkl")):
            info: Dict[str, Any] = {
                "file": pkl_path.name,
                "size_mb": round(pkl_path.stat().st_size / (1024 * 1024), 2),
            }
            meta_path = self._get_metadata_path(pkl_path)
            if meta_path.exists():
                try:
                    info.update(json.loads(meta_path.read_text()))
                except Exception:
                    pass
            result.append(info)
        return result
