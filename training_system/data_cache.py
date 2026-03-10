# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: data_cache.py
Modül: training_system
Görev: Preprocessed Data Cache System - Eğitim verisini cache'ler. Her epoch'ta
       dosyaları tekrar okumak/encode etmek yerine cache'den yükler. Cache key
       hesaplama, cache validation ve cache yönetimi işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache yönetimi)
- Design Patterns: Cache Pattern (data caching strategy)
- Endüstri Standartları: Training data caching best practices

KULLANIM:
- Eğitim verisini cache'lemek için
- Cache'den veri yüklemek için
- Cache validation ve yönetimi için

BAĞIMLILIKLAR:
- TokenizerCore: Tokenization işlemleri
- hashlib: Hash hesaplama
- pickle: Cache serialization

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import os
import hashlib
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """Preprocessed training data cache manager"""
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: str = ".cache/preprocessed_data",
        cache_enabled: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        
        # Cache dizinini oluştur
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DataCache] Cache dizini: {self.cache_dir}")
    
    def _get_cache_key(
        self,
        encode_mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        max_seq_length: int,
        vocab_hash: str,
        alignment_format: str = "autoregressive_v2",  # [OK] YENİ: Alignment formatı (v2 = Input'tan EOS yok)
        formatted: bool = True  # ✅ YENİ: Cache'de formatlanmış veri (BOS/EOS/PAD dahil)
    ) -> str:
        """Cache key oluştur - parametrelere göre unique key"""
        # [OK] DÜZELTME: data_dir'ı normalize et (relative path'e çevir) - cache key tutarlılığı için
        # Absolute path kullanılırsa farklı sistemlerde farklı key oluşur
        data_dir_str = str(self.data_dir)
        try:
            # Absolute path ise relative path'e çevir (cwd'ye göre)
            if os.path.isabs(data_dir_str):
                cwd = os.getcwd()
                try:
                    rel_path = os.path.relpath(data_dir_str, cwd)
                    data_dir_str = rel_path
                except ValueError:
                    # Farklı drive'larda ise absolute path kullan
                    pass
        except Exception:
            pass  # Hata durumunda orijinal path'i kullan
        
        key_parts = [
            data_dir_str,  # Normalize edilmiş path
            encode_mode,
            str(include_whole_words),
            str(include_syllables),
            str(include_sep),
            str(max_seq_length),
            vocab_hash,
            alignment_format,  # [OK] YENİ: Alignment formatı cache key'e eklendi (format değişikliklerinde invalid olur)
            f"formatted_{formatted}",  # ✅ YENİ: Formatlanmış veri flag'i (BOS/EOS/PAD dahil)
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_data_dir_hash(self) -> str:
        """Education klasöründeki dosyaların hash'ini hesapla - değişiklik kontrolü için"""
        if not self.data_dir.exists():
            return ""
        
        file_hashes = []
        file_count = 0
        
        # Tüm dosyaları tara (JSON, TXT, DOCX)
        for ext in [".json", ".txt", ".docx"]:
            for file_path in self.data_dir.rglob(f"*{ext}"):
                if file_path.is_file():
                    file_count += 1
                    # Dosya adı + boyut (mtime'i çıkardık - Colab upload'da değişebilir)
                    # Sadece dosya adı ve boyutu kullan - taşınma/upload'da mtime değişir
                    stat = file_path.stat()
                    # Relative path kullan (absolute path platform farklılıkları yaratabilir)
                    try:
                        rel_path = os.path.relpath(file_path, self.data_dir)
                    except ValueError:
                        rel_path = file_path.name
                    file_info = f"{rel_path}:{stat.st_size}"
                    file_hashes.append(file_info)
        
        if not file_hashes:
            return ""
        
        # Tüm hash'leri birleştir
        combined = "|".join(sorted(file_hashes))
        return hashlib.md5(combined.encode()).hexdigest()[:16]  # İlk 16 karakter yeterli
    
    def _get_cache_path(self, cache_key: str, data_hash: str) -> Path:
        """Cache dosya yolu"""
        cache_filename = f"cached_data_{cache_key}_{data_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def get_cached_data(
        self,
        cache_key: str,
        data_hash: str,
        allow_data_hash_mismatch: bool = True,  # [OK] YENİ: Drive'dan indirme durumunda data_hash farklı olabilir
        allow_cache_key_mismatch: bool = True  # ✅ YENİ: Eğitim sırasında cache key uyuşmasa bile cache kullan (fallback)
    ) -> Optional[List[Tuple[List[int], List[int]]]]:
        """Cache'den veri yükle"""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key, data_hash)
        
        if not cache_path.exists():
            logger.info(f"[DataCache] Cache bulunamadı: {cache_path.name}")
            # Debug: Mevcut cache dosyalarını listele ve neden eşleşmediğini göster
            if self.cache_dir.exists():
                existing_caches = list(self.cache_dir.glob("cached_data_*.pkl"))
                if existing_caches:
                    logger.info(f"[DataCache] Mevcut cache dosyaları ({len(existing_caches)} adet):")
                    matched_cache = None
                    for cache_file in existing_caches:
                        # Cache dosya adı formatı: cached_data_{cache_key}_{data_hash}.pkl
                        # cache_key = 32 karakter (MD5 hash), data_hash = 16 karakter
                        name_without_ext = cache_file.stem  # cached_data_{cache_key}_{data_hash}
                        if name_without_ext.startswith("cached_data_"):
                            rest = name_without_ext[12:]  # {cache_key}_{data_hash}
                            # Son 16 karakter data_hash (son _'dan sonra)
                            # Önceki 32 karakter cache_key (eğer varsa)
                            if "_" in rest:
                                parts = rest.rsplit("_", 1)  # Son _'dan split
                                if len(parts) == 2:
                                    file_cache_key = parts[0]
                                    file_data_hash = parts[1]
                                    
                                    # İlk 3'ünü göster
                                    if len([c for c in existing_caches if c.stem.startswith("cached_data_")]) <= 3 or cache_file == existing_caches[0]:
                                        logger.info(
                                            f"  📁 {cache_file.name}\n"
                                            f"     - Cache key: {file_cache_key[:16]}... (beklenen: {cache_key[:16]}...)\n"
                                            f"     - Data hash: {file_data_hash} (beklenen: {data_hash})"
                                        )
                                    
                                    # [OK] FALLBACK: Cache key eşleşiyorsa ama data_hash farklıysa
                                    if file_cache_key == cache_key:
                                        if file_data_hash == data_hash:
                                            # Tam eşleşme - bu olmamalı çünkü zaten kontrol ettik
                                            matched_cache = cache_file
                                            break
                                        elif allow_data_hash_mismatch:
                                            # Cache key eşleşiyor, data_hash farklı (Drive'dan indirme durumu)
                                            logger.warning(
                                                f"[DataCache] ⚠️ Cache key eşleşiyor ama data_hash farklı!\n"
                                                f"  - Cache key: [OK] Eşleşiyor\n"
                                                f"  - Data hash: {file_data_hash} (beklenen: {data_hash})\n"
                                                f"  - Bu durum Drive'dan indirme sırasında normal olabilir (dosya boyutu/metadata farklılıkları)\n"
                                                f"  - Cache kullanılıyor: {cache_file.name}"
                                            )
                                            matched_cache = cache_file
                                            break
                                else:
                                    if cache_file == existing_caches[0]:
                                        logger.info(f"  📁 {cache_file.name}")
                            else:
                                if cache_file == existing_caches[0]:
                                    logger.info(f"  📁 {cache_file.name} (eski format?)")
                        else:
                            if cache_file == existing_caches[0]:
                                logger.info(f"  📁 {cache_file.name}")
                    
                    if len(existing_caches) > 3:
                        logger.info(f"  ... ve {len(existing_caches) - 3} dosya daha")
                    
                    # Eğer cache key'e göre eşleşen cache bulunduysa, onu kullan
                    if matched_cache:
                        try:
                            logger.info(f"[DataCache] Cache'den yükleniyor (data_hash mismatch): {matched_cache.name}")
                            start_time = time.time()
                            
                            with open(matched_cache, "rb") as f:
                                cached_data = pickle.load(f)
                            
                            load_time = time.time() - start_time
                            logger.info(f"[DataCache] [OK] Cache yüklendi: {len(cached_data):,} örnek, {load_time:.2f}s")
                            
                            return cached_data
                        except Exception as e:
                            logger.warning(f"[DataCache] Cache yükleme hatası: {e}")
                            return None
                    
                    # Cache key eşleşmedi - parametreler farklı
                    # ✅ YENİ: Eğitim sırasında cache silmek yerine, mevcut cache'i kullanmayı dene (fallback)
                    if allow_cache_key_mismatch:
                        # En son oluşturulan cache'i kullan (en büyük dosya veya en yeni)
                        fallback_cache = max(existing_caches, key=lambda p: p.stat().st_mtime)  # En son değiştirilmiş
                        logger.warning(
                            f"[DataCache] ⚠️ Cache key uyuşmuyor, ama fallback aktif. "
                            f"En son cache kullanılıyor: {fallback_cache.name}\n"
                            f"Uyarı: Parametreler farklı olabilir (max_seq_length, vocab_hash, vs.)\n"
                            f"       Eğer sorun yaşarsan: python training_system/prepare_cache.py ile cache'i yeniden oluştur."
                        )
                        try:
                            logger.info(f"[DataCache] Fallback cache yükleniyor: {fallback_cache.name}")
                            start_time = time.time()
                            
                            with open(fallback_cache, "rb") as f:
                                cached_data = pickle.load(f)
                            
                            load_time = time.time() - start_time
                            logger.info(f"[DataCache] [OK] Fallback cache yüklendi: {len(cached_data):,} örnek, {load_time:.2f}s")
                            
                            return cached_data
                        except Exception as e:
                            logger.warning(f"[DataCache] Fallback cache yükleme hatası: {e}")
                            return None
                    else:
                        # Fallback kapalı - strict mode
                        logger.warning(
                            "[DataCache] ⚠️ Cache key uyuşmuyor! "
                            "Bu durum şu sebeplerden olabilir:\n"
                            "  1. alignment_format parametresi değişti\n"
                            "  2. vocab_hash değişti (vocab dosyası güncellendi)\n"
                            "  3. max_seq_length veya diğer encoding parametreleri değişti\n"
                            "  4. data_dir path'i farklı (absolute vs relative)\n"
                            f"  5. {len(existing_caches)} cache dosyası var, hangisini kullanacağı belirsiz\n"
                            "Çözüm: Cache'yi manuel olarak hazırla: python training_system/prepare_cache.py"
                        )
                else:
                    logger.debug("[DataCache] Cache dizininde hiç cache dosyası yok")
            return None
        
        try:
            logger.info(f"[DataCache] Cache'den yükleniyor: {cache_path.name}")
            start_time = time.time()
            
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            load_time = time.time() - start_time
            logger.info(f"[DataCache] [OK] Cache yüklendi: {len(cached_data):,} örnek, {load_time:.2f}s")
            
            return cached_data
        except Exception as e:
            logger.warning(f"[DataCache] Cache yükleme hatası: {e}")
            return None
    
    def save_cached_data(
        self,
        cache_key: str,
        data_hash: str,
        data: List[Tuple[List[int], List[int]]]
    ) -> bool:
        """Veriyi cache'e kaydet"""
        if not self.cache_enabled:
            return False
        
        cache_path = self._get_cache_path(cache_key, data_hash)
        
        try:
            logger.info(f"[DataCache] Cache'e kaydediliyor: {cache_path.name}")
            start_time = time.time()
            
            # Geçici dosya ile atomic write
            temp_path = cache_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.replace(cache_path)
            
            save_time = time.time() - start_time
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"[DataCache] [OK] Cache kaydedildi: {len(data):,} örnek, {file_size_mb:.2f} MB, {save_time:.2f}s")
            
            return True
        except Exception as e:
            logger.error(f"[DataCache] Cache kaydetme hatası: {e}")
            return False
    
    def get_or_process(
        self,
        tokenizer_core,
        encode_mode: str,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        max_seq_length: int,
        process_func,
        alignment_format: str = "autoregressive_v2",  # [OK] YENİ: Alignment formatı
        format_data: bool = True,  # ✅ YENİ: Cache'de formatlanmış veri sakla (BOS/EOS/PAD dahil)
        format_func = None  # ✅ YENİ: Formatlama fonksiyonu (DataPreparator._apply_autoregressive_formatting)
    ) -> Tuple[List[Tuple[List[int], List[int]]], bool]:
        """
        Cache'den yükle veya işle ve cache'e kaydet
        
        Args:
            tokenizer_core: TokenizerCore instance (vocab hash için)
            encode_mode: Encoding mode
            include_whole_words: Whole words flag
            include_syllables: Syllables flag
            include_sep: SEP flag
            max_seq_length: Max sequence length
            process_func: Veriyi işleyen fonksiyon -> List[Tuple[List[int], List[int]]]
        
        Returns:
            (processed_data, from_cache)
        """
        # Vocab hash (vocab değişirse cache invalid)
        # Tüm vocab'ı hash'le - sadece ilk 1000 token yeterli değil!
        vocab = tokenizer_core.get_vocab()
        # Vocab'ın key'lerini ve ID'lerini hash'le (daha güvenilir)
        vocab_items = []
        for token, data in vocab.items():
            if isinstance(data, dict):
                token_id = data.get('id', 0)
                vocab_items.append(f"{token}:{token_id}")
            elif isinstance(data, int):
                vocab_items.append(f"{token}:{data}")
        vocab_str = "|".join(sorted(vocab_items))
        vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:16]
        
        # Data dir hash (dosyalar değişirse cache invalid)
        data_hash = self._get_data_dir_hash()
        
        # Cache key
        cache_key = self._get_cache_key(
            encode_mode,
            include_whole_words,
            include_syllables,
            include_sep,
            max_seq_length,
            vocab_hash,
            alignment_format,  # [OK] YENİ: Alignment formatı cache key'e eklendi
            formatted=format_data  # ✅ YENİ: Formatlanmış veri flag'i
        )
        
        # Debug: Cache key parametrelerini logla
        logger.debug(
            f"[DataCache] Cache key oluşturuluyor:\n"
            f"  - encode_mode: {encode_mode}\n"
            f"  - include_whole_words: {include_whole_words}\n"
            f"  - include_syllables: {include_syllables}\n"
            f"  - include_sep: {include_sep}\n"
            f"  - max_seq_length: {max_seq_length}\n"
            f"  - vocab_hash: {vocab_hash}\n"
            f"  - alignment_format: {alignment_format}\n"
            f"  - format_data: {format_data}\n"
            f"  - data_hash: {data_hash}\n"
            f"  - cache_key (MD5): {cache_key}\n"
            f"  - Beklenen cache dosyası: cached_data_{cache_key}_{data_hash}.pkl"
        )
        
        # Cache'den yüklemeyi dene
        # ✅ YENİ: Eğitim sırasında cache key uyuşmasa bile cache kullan (fallback)
        cached_data = self.get_cached_data(cache_key, data_hash, allow_cache_key_mismatch=True)
        if cached_data is not None:
            return cached_data, True
        
        # Cache yok - işle
        logger.info(f"[DataCache] Cache yok, veri işleniyor...")
        raw_data = process_func()
        
        # ✅ YENİ: Formatlanmış veri saklanacaksa formatla
        if format_data and format_func is not None:
            logger.info(f"[DataCache] Veri formatlanıyor (BOS/EOS/PAD ekleniyor)...")
            processed_data = format_func(raw_data)
        else:
            processed_data = raw_data
        
        # Cache'e kaydet
        self.save_cached_data(cache_key, data_hash, processed_data)
        
        return processed_data, False
    
    def clear_cache(self) -> int:
        """Tüm cache dosyalarını sil"""
        if not self.cache_dir.exists():
            return 0
        
        deleted = 0
        for cache_file in self.cache_dir.glob("cached_data_*.pkl"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"[DataCache] Cache silme hatası: {cache_file.name} - {e}")
        
        logger.info(f"[DataCache] {deleted} cache dosyası silindi")
        return deleted
    
    # =========================
    # BPE Training Corpus Cache
    # =========================
    
    def _get_corpus_cache_key(
        self,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
        data_hash: str
    ) -> str:
        """BPE corpus cache key oluştur"""
        # data_dir'ı normalize et
        data_dir_str = str(self.data_dir)
        try:
            if os.path.isabs(data_dir_str):
                cwd = os.getcwd()
                try:
                    rel_path = os.path.relpath(data_dir_str, cwd)
                    data_dir_str = rel_path
                except ValueError:
                    pass
        except Exception:
            pass
        
        key_parts = [
            data_dir_str,
            "bpe_corpus",  # Cache tipi
            str(include_whole_words),
            str(include_syllables),
            str(include_sep),
            data_hash,
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_corpus_cache_path(self, cache_key: str) -> Path:
        """BPE corpus cache dosya yolu"""
        cache_filename = f"bpe_corpus_{cache_key}.pkl"
        return self.cache_dir / cache_filename
    
    def get_cached_corpus(
        self,
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
    ) -> Optional[List[str]]:
        """
        BPE training corpus'unu cache'den yükle
        
        Args:
            include_whole_words: Whole words flag
            include_syllables: Syllables flag
            include_sep: SEP flag
        
        Returns:
            Cached corpus listesi veya None
        """
        if not self.cache_enabled:
            return None
        
        # Data hash hesapla
        data_hash = self._get_data_dir_hash()
        if not data_hash:
            logger.debug("[DataCache] Data hash hesaplanamadı, corpus cache kontrol edilmiyor")
            return None
        
        # Path normalization (debug için)
        data_dir_str = str(self.data_dir)
        normalized_data_dir = data_dir_str
        try:
            if os.path.isabs(data_dir_str):
                cwd = os.getcwd()
                try:
                    rel_path = os.path.relpath(data_dir_str, cwd)
                    normalized_data_dir = rel_path
                except ValueError:
                    pass
        except Exception:
            pass
        
        # Cache key oluştur
        cache_key = self._get_corpus_cache_key(
            include_whole_words,
            include_syllables,
            include_sep,
            data_hash
        )
        
        cache_path = self._get_corpus_cache_path(cache_key)
        
        # DEBUG: Mevcut cache dosyalarını listele
        if not cache_path.exists():
            logger.info(f"[DataCache] Corpus cache bulunamadı: {cache_path.name}")
            logger.info(f"[DataCache] Aranan cache key: {cache_key}")
            logger.info(f"[DataCache] Data dir (normalized): {normalized_data_dir}")
            logger.info(f"[DataCache] Parametreler: whole_words={include_whole_words}, syllables={include_syllables}, sep={include_sep}")
            logger.info(f"[DataCache] Data hash: {data_hash[:16]}...")
            
            # Mevcut cache dosyalarını listele ve fallback kontrolü yap
            if self.cache_dir.exists():
                existing_caches = list(self.cache_dir.glob("bpe_corpus_*.pkl"))
                if existing_caches:
                    logger.info(f"[DataCache] Mevcut cache dosyaları ({len(existing_caches)} adet):")
                    for cache_file in existing_caches[:5]:  # İlk 5'ini göster
                        size_mb = cache_file.stat().st_size / (1024 * 1024)
                        logger.info(f"  - {cache_file.name} ({size_mb:.2f} MB)")
                    if len(existing_caches) > 5:
                        logger.info(f"  ... ve {len(existing_caches) - 5} dosya daha")
                    
                    # FALLBACK: Eğer tek bir cache dosyası varsa ve parametreler uyuyorsa kullan
                    # (Eski hash yöntemiyle oluşturulmuş cache'i kullanmak için)
                    if len(existing_caches) == 1 and existing_caches[0].stat().st_size > 0:
                        logger.warning(
                            f"[DataCache] ⚠️ Cache key uyuşmuyor ama tek cache dosyası var. "
                            f"Eski hash yöntemiyle oluşturulmuş olabilir. "
                            f"Cache'i yeniden oluşturmanız önerilir: python tokenizer_management/prepare_bpe_cache.py"
                        )
                else:
                    logger.info(f"[DataCache] Cache dizininde hiç cache dosyası yok: {self.cache_dir}")
            else:
                logger.info(f"[DataCache] Cache dizini mevcut değil: {self.cache_dir}")
            
            return None
        
        try:
            logger.info(f"[DataCache] Corpus cache'den yükleniyor: {cache_path.name}")
            start_time = time.time()
            
            with open(cache_path, "rb") as f:
                cached_corpus = pickle.load(f)
            
            load_time = time.time() - start_time
            logger.info(
                f"[DataCache] [OK] Corpus cache yüklendi: {len(cached_corpus):,} metin, {load_time:.2f}s"
            )
            
            return cached_corpus
        except Exception as e:
            logger.warning(f"[DataCache] Corpus cache yükleme hatası: {e}")
            return None
    
    def save_corpus(
        self,
        corpus: List[str],
        include_whole_words: bool,
        include_syllables: bool,
        include_sep: bool,
    ) -> bool:
        """
        BPE training corpus'unu cache'e kaydet
        
        Args:
            corpus: Corpus listesi
            include_whole_words: Whole words flag
            include_syllables: Syllables flag
            include_sep: SEP flag
        
        Returns:
            Başarı durumu
        """
        if not self.cache_enabled:
            return False
        
        # Data hash hesapla
        data_hash = self._get_data_dir_hash()
        if not data_hash:
            logger.debug("[DataCache] Data hash hesaplanamadı, corpus cache kaydedilmiyor")
            return False
        
        # Cache key oluştur
        cache_key = self._get_corpus_cache_key(
            include_whole_words,
            include_syllables,
            include_sep,
            data_hash
        )
        
        cache_path = self._get_corpus_cache_path(cache_key)
        
        try:
            logger.info(f"[DataCache] Corpus cache'e kaydediliyor: {cache_path.name}")
            start_time = time.time()
            
            # Geçici dosya ile atomic write
            temp_path = cache_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.replace(cache_path)
            
            save_time = time.time() - start_time
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"[DataCache] [OK] Corpus cache kaydedildi: {len(corpus):,} metin, "
                f"{file_size_mb:.2f} MB, {save_time:.2f}s"
            )
            
            return True
        except Exception as e:
            logger.error(f"[DataCache] Corpus cache kaydetme hatası: {e}")
            return False
    
    def get_or_create_corpus(
        self,
        process_func,
        include_whole_words: bool = True,
        include_syllables: bool = False,
        include_sep: bool = True,
    ) -> Tuple[List[str], bool]:
        """
        BPE training corpus'unu cache'den yükle veya oluştur ve cache'e kaydet
        
        Args:
            process_func: Corpus oluşturan fonksiyon -> List[str]
            include_whole_words: Whole words flag (cache key'e dahil)
            include_syllables: Syllables flag (cache key'e dahil)
            include_sep: SEP flag (cache key'e dahil)
        
        Returns:
            (corpus, from_cache)
        """
        # Cache'den yüklemeyi dene
        cached_corpus = self.get_cached_corpus(
            include_whole_words,
            include_syllables,
            include_sep
        )
        if cached_corpus is not None:
            return cached_corpus, True
        
        # Cache yok - corpus oluştur
        logger.info("[DataCache] Corpus cache yok, corpus oluşturuluyor...")
        corpus = process_func()
        
        # Cache'e kaydet
        self.save_corpus(
            corpus,
            include_whole_words,
            include_syllables,
            include_sep
        )
        
        return corpus, False

