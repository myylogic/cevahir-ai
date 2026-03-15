# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: prepare_cache.py
Modül: training_system
Görev: Cache Preparation Script - Colab dışında cache oluşturma scripti.
       Cache'i local bilgisayarda oluşturur, oluşturulan cache dosyasını
       Colab'a yüklenebilir hale getirir. Preprocessed data cache'i hazırlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache hazırlama scripti)
- Design Patterns: Script Pattern (standalone utility script)
- Endüstri Standartları: Cache preparation workflow

KULLANIM:
- Local bilgisayarda cache oluşturmak için
- Colab'a cache yüklemek için
- Preprocessed data cache hazırlama için

BAĞIMLILIKLAR:
- DataCache: Cache yönetimi
- TokenizerCore: Tokenization işlemleri
- Config modülleri: Yapılandırma yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""
import os
import sys
import json
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_system.data_cache import DataCache
# V3: DataCacheV3 ile checksum + metadata desteği
try:
    from training_system.v3.data.cache_v3 import DataCacheV3
    _CACHE_V3_AVAILABLE = True
except ImportError:
    DataCacheV3 = None  # type: ignore
    _CACHE_V3_AVAILABLE = False
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG, TOKENIZER_CONFIG, BPE_DETAILED_CONFIG
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareCache")


def prepare_cache(
    data_dir: str = "education",
    cache_dir: str = ".cache/preprocessed_data",
    max_seq_length: int = None,  # [OK] DÜZELTME: None ise TOKENIZER_CONFIG'ten al
    include_whole_words: bool = None,  #  YENİ: None ise BPE_DETAILED_CONFIG'ten al
    include_syllables: bool = None,  #  YENİ: None ise BPE_DETAILED_CONFIG'ten al
    include_sep: bool = None,  #  YENİ: None ise BPE_DETAILED_CONFIG'ten al
    clear_old_cache: bool = True,  # [OK] YENİ: Eski cache'i temizle
):
    """
    Cache'i hazırla
    
    Args:
        data_dir: Eğitim verisi dizini
        cache_dir: Cache dizini
        max_seq_length: Max sequence length
        include_whole_words: Whole words flag
        include_syllables: Syllables flag
        include_sep: SEP flag
    """
    logger.info("="*60)
    logger.info("CACHE HAZIRLAMA BAŞLIYOR")
    logger.info("="*60)
    
    # [OK] DÜZELTME: max_seq_length'i TOKENIZER_CONFIG'ten al (eğer None ise)
    if max_seq_length is None:
        max_seq_length = TOKENIZER_CONFIG.get("max_seq_length", 512)
        logger.info(f"  max_seq_length TOKENIZER_CONFIG'ten alındı: {max_seq_length}")
    
    #  YENİ: include_* parametrelerini BPE_DETAILED_CONFIG'ten al (eğer None ise)
    # Bu, train.py ile aynı config'i kullanmasını sağlar (cache key uyumluluğu için)
    if include_whole_words is None:
        include_whole_words = BPE_DETAILED_CONFIG.get("include_whole_words", True)
        logger.info(f"  include_whole_words BPE_DETAILED_CONFIG'ten alındı: {include_whole_words}")
    if include_syllables is None:
        include_syllables = BPE_DETAILED_CONFIG.get("include_syllables", False)
        logger.info(f"  include_syllables BPE_DETAILED_CONFIG'ten alındı: {include_syllables}")
    if include_sep is None:
        include_sep = BPE_DETAILED_CONFIG.get("include_sep", False)
        logger.info(f"  include_sep BPE_DETAILED_CONFIG'ten alındı: {include_sep}")
    
    # Config hazırla
    vocab_path = BPE_CONFIG.get("vocab_file", "data/vocab_lib/vocab.json")
    merges_path = BPE_CONFIG.get("merges_file", "data/merges_lib/merges.txt")
    
    config = {
        "data_dir": data_dir,
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "max_seq_length": max_seq_length,
        "bpe_rebuild": False,  # Vocab/merges mevcut, rebuild yok
        "train_include_whole_words": include_whole_words,
        "train_include_syllables": include_syllables,
        "train_include_sep": include_sep,
        "use_gpu": True,  # CPU'da çalış (local bilgisayar için)
    }
    
    # DataCache oluştur
    cache = DataCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        cache_enabled=True
    )
    
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Max seq length: {max_seq_length}")
    logger.info(f"Include whole words: {include_whole_words}")
    logger.info(f"Include syllables: {include_syllables}")
    logger.info(f"Include sep: {include_sep}")
    
    # [OK] Eski cache'i temizle (yeni format için)
    if clear_old_cache:
        logger.info("\n[0] Eski cache temizleniyor (yeni format ile yeniden oluşturulacak)...")
        try:
            deleted_count = cache.clear_cache()
            if deleted_count > 0:
                logger.info(f"[OK] {deleted_count} eski cache dosyası silindi")
            else:
                logger.info("  Silinecek cache dosyası yok (zaten temiz)")
        except Exception as e:
            logger.warning(f"  Cache temizleme hatası (devam ediliyor): {e}")
    
    # TokenizerCore oluştur
    logger.info("\n[1] TokenizerCore başlatılıyor...")
    try:
        tokenizer_core = TokenizerCore(config)
        tokenizer_core.finalize_vocab()
        logger.info("[OK] TokenizerCore başlatıldı")
    except Exception as e:
        logger.error(f" TokenizerCore başlatılamadı: {e}", exc_info=True)
        raise
    
    #  Vocab bilgileri (formatlama için gerekli)
    vocab = tokenizer_core.get_vocab()
    vocab_size = len(vocab)
    
    def _id_of(token: str) -> int:
        val = vocab.get(token)
        if isinstance(val, dict):
            return int(val.get("id", 0))
        return int(val or 0)
    
    BOS_ID = _id_of("<BOS>")
    EOS_ID = _id_of("<EOS>")
    PAD_ID = _id_of("<PAD>")
    UNK_ID = _id_of("<UNK>")
    SEP_ID = _id_of("<SEP>")
    
    logger.info(f"Special tokens → BOS:{BOS_ID}, EOS:{EOS_ID}, PAD:{PAD_ID}, UNK:{UNK_ID}, SEP:{SEP_ID}")
    
    #  YENİ: PAD_ID'yi global olarak sakla (overlap kontrolü için)
    global_pad_id = PAD_ID
    
    #  Formatlama fonksiyonu (cache'de tam formatlanmış veri saklamak için)
    def format_data_func(raw_data):
        """Cache'de tam formatlanmış veri saklamak için formatlama fonksiyonu"""
        logger.info(f"[Formatlama] {len(raw_data):,} örnek formatlanıyor (BOS/EOS/SEP/PAD ekleniyor, max_seq_len={max_seq_length} hizalanıyor)...")
        
        formatted_data = []
        
        def clamp_ids(seq, vocab_size, unk_id):
            """OOV'leri UNK ile değiştir"""
            return [i if 0 <= i < vocab_size else unk_id for i in seq]
        
        #  OVERLAP KONTROLÜ: Hash tracking (formatlama öncesi)
        import hashlib
        pre_format_hashes = {}  # hash -> (source_id, original_length)
        post_format_hashes = {}  # hash -> count
        
        def hash_sequence_no_pad(seq, pad_id):
            """PAD token'larını filtreleyerek hash'le"""
            clean_seq = [t for t in seq if t != pad_id]
            seq_str = str(clean_seq)
            return hashlib.sha256(seq_str.encode()).hexdigest()[:16]
        
        #  YENİ: source_id kontrolü (varsa SAKLA - overlap önleme için kritik!)
        source_id_count = 0
        no_source_id_count = 0
        for idx, item in enumerate(raw_data):
            # Veri formatı: (inp_ids, tgt_ids) veya (inp_ids, tgt_ids, source_id)
            source_id = None  # Default
            if len(item) == 3:
                inp_ids, tgt_ids, source_id = item  # source_id var - SAKLA!
                source_id_count += 1
            elif len(item) == 2:
                inp_ids, tgt_ids = item  # source_id yok
                no_source_id_count += 1
            else:
                logger.warning(f"[!] Örnek {idx} atlandı: Geçersiz format (len={len(item)})")
                continue
            
            #  DEBUG: İlk 5 örnek için source_id bilgisi
            if idx < 5:
                logger.debug(f"[DEBUG] Örnek {idx}: source_id={source_id if source_id is not None else 'YOK'}")
            
            #  OVERLAP KONTROLÜ: Formatlama öncesi hash (PAD yok, zaten formatlanmamış)
            # Formatlama öncesi veri zaten PAD içermez, direkt hash'leyebiliriz
            pre_seq_str = str(list(inp_ids))
            pre_hash = hashlib.sha256(pre_seq_str.encode()).hexdigest()[:16]
            if pre_hash not in pre_format_hashes:
                pre_format_hashes[pre_hash] = []
            pre_format_hashes[pre_hash].append((source_id, len(inp_ids), idx))
            
            try:
                #  BASİT VE DOĞRU AUTOREGRESSIVE FORMAT
                # Input: [BOS, t1, t2, ..., tN]
                # Target: [t1, t2, ..., tN, EOS]
                # Autoregressive: Input[i] → Target[i] (BOS → t1, t1 → t2, ..., tN → EOS)
                
                # 1. BOS/EOS ekle
                seq_in = [BOS_ID] + list(inp_ids)
                seq_tgt = list(tgt_ids) + [EOS_ID]  # tgt_ids = inp_ids (autoregressive)
                
                # 2. Truncate: max_seq_length'e kadar kes (EOS'u MUTLAKA koru)
                if len(seq_in) > max_seq_length:
                    seq_in = seq_in[:max_seq_length]
                if len(seq_tgt) > max_seq_length:
                    # EOS'u koru: son pozisyonda EOS olmalı
                    seq_tgt = seq_tgt[:max_seq_length-1] + [EOS_ID]
                
                # 3. Alignment: Input ve Target aynı uzunlukta olmalı
                # Autoregressive için: len(seq_in) == len(seq_tgt)
                current_len = max(len(seq_in), len(seq_tgt))
                
                # Input'u uzat (son token'ı tekrarla)
                if len(seq_in) < current_len:
                    if len(seq_in) > 0:
                        seq_in = seq_in + [seq_in[-1]] * (current_len - len(seq_in))
                    else:
                        seq_in = [BOS_ID] * current_len
                
                # Target'i uzat (EOS'u koru, sonra PAD ekle)
                if len(seq_tgt) < current_len:
                    # EOS'u kontrol et
                    if len(seq_tgt) > 0 and seq_tgt[-1] == EOS_ID:
                        # EOS var, sadece PAD ekle
                        seq_tgt = seq_tgt + [PAD_ID] * (current_len - len(seq_tgt))
                    else:
                        # EOS yok, ekle ve PAD ekle
                        seq_tgt = seq_tgt + [EOS_ID] + [PAD_ID] * (current_len - len(seq_tgt) - 1)
                
                # 4. Padding: max_seq_length'e kadar PAD ekle
                if len(seq_in) < max_seq_length:
                    seq_in += [PAD_ID] * (max_seq_length - len(seq_in))
                
                if len(seq_tgt) < max_seq_length:
                    # EOS'u kontrol et
                    if len(seq_tgt) > 0 and seq_tgt[-1] == EOS_ID:
                        # EOS var, sadece PAD ekle
                        seq_tgt += [PAD_ID] * (max_seq_length - len(seq_tgt))
                    else:
                        # EOS yok, ekle ve PAD ekle
                        seq_tgt = seq_tgt + [EOS_ID] + [PAD_ID] * (max_seq_length - len(seq_tgt) - 1)
                
                # 5. FINAL CHECK: Target'ın içeriği ve EOS'u kontrol et
                # Son PAD olmayan token'ı bul
                last_non_pad_pos = None
                for i in range(len(seq_tgt) - 1, -1, -1):
                    if seq_tgt[i] != PAD_ID:
                        last_non_pad_pos = i
                        break
                
                # Eğer target tamamen PAD ise, içerik kaybolmuş - YENİDEN OLUŞTUR
                if last_non_pad_pos is None:
                    # İçerik kaybolmuş, yeniden oluştur
                    seq_tgt = list(tgt_ids)[:max_seq_length-1] + [EOS_ID]
                    if len(seq_tgt) < max_seq_length:
                        seq_tgt += [PAD_ID] * (max_seq_length - len(seq_tgt))
                    # Input'u da yeniden oluştur
                    seq_in = [BOS_ID] + list(inp_ids)[:max_seq_length-1]
                    if len(seq_in) < max_seq_length:
                        seq_in += [PAD_ID] * (max_seq_length - len(seq_in))
                # EOS yoksa ekle
                elif seq_tgt[last_non_pad_pos] != EOS_ID:
                    # EOS yok, ekle (son PAD olmayan token'dan sonra)
                    seq_tgt = seq_tgt[:last_non_pad_pos+1] + [EOS_ID] + seq_tgt[last_non_pad_pos+2:]
                    # Input'u da uzat
                    if len(seq_in) < len(seq_tgt):
                        seq_in = seq_in + [seq_in[-1] if len(seq_in) > 0 else BOS_ID] * (len(seq_tgt) - len(seq_in))
                    # Tekrar truncate ve padding
                    if len(seq_tgt) > max_seq_length:
                        seq_tgt = seq_tgt[:max_seq_length]
                    if len(seq_in) > max_seq_length:
                        seq_in = seq_in[:max_seq_length]
                    if len(seq_tgt) < max_seq_length:
                        seq_tgt += [PAD_ID] * (max_seq_length - len(seq_tgt))
                    if len(seq_in) < max_seq_length:
                        seq_in += [PAD_ID] * (max_seq_length - len(seq_in))
                
                # OOV clamp
                seq_in = clamp_ids(seq_in, vocab_size, UNK_ID)
                seq_tgt = clamp_ids(seq_tgt, vocab_size, UNK_ID)
                
                #  OVERLAP KONTROLÜ: Formatlama sonrası hash (PAD olmadan)
                post_hash = hash_sequence_no_pad(seq_in, pad_id=PAD_ID)
                if post_hash not in post_format_hashes:
                    post_format_hashes[post_hash] = []
                post_format_hashes[post_hash].append((source_id, len(seq_in), idx))
                
                #  DEDUPLICATION: Aynı source_id içinde duplicate chunk'ları filtrele
                # Eğer bu hash + source_id kombinasyonu daha önce görüldüyse, atla
                # Not: Hash collision tespiti için post_format_hashes kullanıyoruz
                # Ama duplicate'leri filtrelemek için ayrı bir set kullanıyoruz
                hash_key = (post_hash, source_id) if source_id is not None else post_hash
                
                # İlk görülen hash_key'leri takip et
                if not hasattr(format_data_func, '_seen_hashes'):
                    format_data_func._seen_hashes = set()
                
                if hash_key not in format_data_func._seen_hashes:
                    format_data_func._seen_hashes.add(hash_key)
                    #  DÜZELTME: source_id'yi SAKLA (overlap önleme için kritik!)
                    if source_id is not None:
                        formatted_data.append((seq_in, seq_tgt, source_id))  # source_id ile
                    else:
                        formatted_data.append((seq_in, seq_tgt))  # source_id yok (geriye dönük uyumluluk)
                else:
                    # Duplicate chunk - atla (aynı source_id içinde)
                    #  INFO seviyesine çıkar (debug için önemli)
                    if not hasattr(format_data_func, '_dup_count'):
                        format_data_func._dup_count = 0
                    format_data_func._dup_count += 1
                    if format_data_func._dup_count <= 10:  # İlk 10 duplicate'i göster
                        logger.info(f"[DEDUP] Örnek {idx} atlandı: Duplicate chunk (source_id={source_id}, hash={post_hash[:8]}...)")
                    continue  # Bu duplicate chunk'ı atla
                
            except Exception as e:
                logger.warning(f"[!] Örnek {idx} atlandı: {e}")
                continue
        
        #  DEDUPLICATION istatistikleri
        total_processed = source_id_count + no_source_id_count
        total_duplicates = total_processed - len(formatted_data)
        logger.info(f"[Formatlama]  {len(formatted_data):,} örnek formatlandı (hazır)")
        if total_duplicates > 0:
            logger.info(f"[Formatlama] 🗑️  {total_duplicates:,} duplicate chunk filtrelendi ({total_duplicates/total_processed*100:.2f}%)")
        logger.info(f"[Formatlama] 📊 source_id istatistikleri:")
        logger.info(f"  - source_id olan: {source_id_count:,} ({source_id_count/total_processed*100:.1f}%)")
        logger.info(f"  - source_id olmayan: {no_source_id_count:,} ({no_source_id_count/total_processed*100:.1f}%)")
        
        # Clean up temporary attribute
        if hasattr(format_data_func, '_seen_hashes'):
            delattr(format_data_func, '_seen_hashes')
        
        #  DEBUG: Formatlanmış veride source_id kontrolü
        if formatted_data:
            first_fmt = formatted_data[0]
            has_source_id_in_formatted = len(first_fmt) == 3
            logger.info(f"[Formatlama] 📊 Formatlanmış veri formatı:")
            logger.info(f"  - İlk örnek formatı: {len(first_fmt)} eleman")
            logger.info(f"  - source_id kaydedildi mi? {has_source_id_in_formatted}")
            if has_source_id_in_formatted:
                _, _, sid = first_fmt
                logger.info(f"  - İlk örnek source_id: {sid}")
        
        #  OVERLAP ANALİZİ: Formatlama öncesi ve sonrası hash karşılaştırması
        logger.info(f"\n[OVERLAP ANALİZİ] Formatlama sürecinde hash kontrolü:")
        
        # Pre-format hash collisions (aynı içerik farklı source_id'lerde)
        pre_collisions = {h: info_list for h, info_list in pre_format_hashes.items() if len(info_list) > 1}
        if pre_collisions:
            logger.warning(f"    Formatlama ÖNCESİ: {len(pre_collisions)} hash collision tespit edildi!")
            collision_count = sum(len(info_list) for info_list in pre_collisions.values())
            logger.warning(f"  - Toplam collision örnek sayısı: {collision_count}")
            # İlk 5 collision'ı göster
            for i, (h, info_list) in enumerate(list(pre_collisions.items())[:5]):
                source_ids_in_collision = [info[0] for info in info_list]
                unique_source_ids = set(source_ids_in_collision)
                logger.warning(f"    Collision {i+1}: hash={h}, {len(info_list)} örnek, {len(unique_source_ids)} farklı source_id: {sorted(list(unique_source_ids))[:5]}")
        
        # Post-format hash collisions
        post_collisions = {h: info_list for h, info_list in post_format_hashes.items() if len(info_list) > 1}
        if post_collisions:
            logger.warning(f"    Formatlama SONRASI: {len(post_collisions)} hash collision tespit edildi!")
            collision_count = sum(len(info_list) for info_list in post_collisions.values())
            logger.warning(f"  - Toplam collision örnek sayısı: {collision_count}")
            
            #  ANALİZ: Aynı source_id içinde collision var mı?
            same_source_collision_count = 0
            different_source_collision_count = 0
            
            # İlk 5 collision'ı göster
            for i, (h, info_list) in enumerate(list(post_collisions.items())[:5]):
                source_ids_in_collision = [info[0] for info in info_list]
                unique_source_ids = set(source_ids_in_collision)
                logger.warning(f"    Collision {i+1}: hash={h}, {len(info_list)} örnek, {len(unique_source_ids)} farklı source_id: {sorted(list(unique_source_ids))[:5]}")
                
                #  DÜZELTME: Aynı source_id içinde collision var mı kontrol et
                same_source_collisions = {}
                for source_id, length, idx in info_list:
                    if source_id not in same_source_collisions:
                        same_source_collisions[source_id] = []
                    same_source_collisions[source_id].append((length, idx))
                
                #  MANTIK DÜZELTMESİ: Aynı source_id içinde birden fazla örnek var mı?
                has_same_source_collision = any(len(items) > 1 for items in same_source_collisions.values())
                
                if has_same_source_collision:
                    # Aynı source_id içinde collision (truncation nedeniyle olabilir)
                    same_source_collision_count += 1
                    logger.warning(f"        AYNI source_id içinde collision var! (truncation nedeniyle olabilir)")
                    for sid, items in same_source_collisions.items():
                        if len(items) > 1:
                            lengths = [item[0] for item in items]
                            logger.warning(f"        source_id={sid}: {len(items)} örnek, uzunluklar: {lengths}")
                
                if len(unique_source_ids) > 1:
                    # Farklı source_id'lerde collision
                    different_source_collision_count += 1
            
            #  ÖNEMLİ: Farklı source_id'lerdeki collision'lar (duplicate content)
            if different_source_collision_count > 0:
                logger.warning(f"    {different_source_collision_count} collision FARKLI source_id'lerde!")
                logger.warning(f"      Bu, farklı dosyalarda aynı içeriğin olduğunu gösterir (kaynak veri sorunu)")
                logger.warning(f"      Bu duplicate'ler filtrelenmedi (farklı source_id olduğu için)")
                logger.warning(f"      Split sırasında bu duplicate'ler farklı setlere gidebilir (overlap riski!)")
        
        if not pre_collisions and not post_collisions:
            logger.info(f"   Formatlama öncesi/sonrası hash collision yok!")
        
        return formatted_data
    
    # Veri işleme fonksiyonu
    def process_data():
        """Veriyi işle (formatlama yapmadan)"""
        logger.info("\n[2] Veri işleniyor (encode ediliyor)...")
        try:
            raw_data = tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=include_whole_words,
                include_syllables=include_syllables,
                include_sep=include_sep,
                include_source_id=True,  #  KRİTİK: source_id ekle (overlap önleme için)
            )
            logger.info(f"[OK] {len(raw_data):,} örnek encode edildi")
            
            #  DEBUG: raw_data'da source_id kontrolü
            if raw_data:
                first_raw = raw_data[0]
                has_source_id_raw = len(first_raw) == 3
                logger.info(f"[DEBUG] 📊 raw_data formatı:")
                logger.info(f"  - İlk örnek formatı: {len(first_raw)} eleman")
                logger.info(f"  - source_id var mı? {has_source_id_raw}")
                if has_source_id_raw:
                    _, _, sid_raw = first_raw
                    logger.info(f"  - İlk örnek source_id: {sid_raw}")
                    # source_id istatistikleri
                    source_id_set = set()
                    for item in raw_data:
                        if len(item) == 3:
                            _, _, sid = item
                            source_id_set.add(sid)
                    logger.info(f"  - Unique source_id sayısı: {len(source_id_set):,}")
                    logger.info(f"  - İlk 5 source_id: {sorted(list(source_id_set))[:5]}")
                else:
                    logger.warning(f"    source_id YOK! tokenizer_core.load_training_data() source_id döndürmüyor!")
            
            return raw_data
        except Exception as e:
            logger.error(f" Veri işleme hatası: {e}", exc_info=True)
            raise
    
    # Cache'den yükle veya işle (formatlanmış veri ile)
    logger.info("\n[3] Cache kontrolü yapılıyor...")
    try:
        formatted_data, from_cache = cache.get_or_process(
            tokenizer_core=tokenizer_core,
            encode_mode="train",
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
            max_seq_length=max_seq_length,
            process_func=process_data,
            alignment_format="autoregressive_v2",
            format_data=True,  #  Cache'de formatlanmış veri sakla
            format_func=format_data_func  #  Formatlama fonksiyonu
        )
        
        #  DEBUG: Cache'den yüklenen veride source_id kontrolü
        if from_cache:
            logger.info(f"[DEBUG] 📦 Cache'den yüklendi")
        else:
            logger.info(f"[DEBUG] 📝 Yeni cache oluşturuldu")
        
        if formatted_data:
            first_item = formatted_data[0]
            has_source_id = len(first_item) == 3
            logger.info(f"[DEBUG] 📊 Cache'deki formatlanmış veri:")
            logger.info(f"  - Toplam örnek: {len(formatted_data):,}")
            logger.info(f"  - İlk örnek formatı: {len(first_item)} eleman")
            logger.info(f"  - source_id var mı? {has_source_id}")
            if has_source_id:
                _, _, sid = first_item
                logger.info(f"  - İlk örnek source_id: {sid}")
                # İlk 5 örnek için source_id'leri göster
                logger.info(f"  - İlk 5 örnek source_id'leri:")
                for i in range(min(5, len(formatted_data))):
                    if len(formatted_data[i]) == 3:
                        _, _, sid_val = formatted_data[i]
                        logger.info(f"    [{i}] source_id={sid_val}")
            else:
                logger.warning(f"    source_id YOK! Overlap önlenemez!")
        
        if from_cache:
            logger.info("[OK] Cache'den yüklendi (formatlanmış, hazır)")
        else:
            logger.info("[OK] Cache oluşturuldu ve kaydedildi (formatlanmış, hazır)")
        
        logger.info(f"[OK] Toplam {len(formatted_data):,} örnek hazır (BOS/EOS/SEP/PAD eklenmiş, max_seq_len={max_seq_length} hizalanmış)")
        
        #  NOT: Split eğitim sırasında yapılacak (data_preparator._split_train_val içinde)
        # Cache hazırlanırken split yapılmıyor, sadece formatlanmış veri saklanıyor
        # Overlap kontrolü: source_id bazlı split ile önleniyor (tokenizer_core.load_training_data'da eklendi)
        
    except Exception as e:
        logger.error(f" Cache işleme hatası: {e}", exc_info=True)
        raise
    
    # V3: Yeni oluşturulan cache dosyaları için checksum + metadata kaydet
    if _CACHE_V3_AVAILABLE and not from_cache:
        logger.info("\n[3.5] V3 integrity dosyaları oluşturuluyor (checksum + metadata)...")
        try:
            cache_v3 = DataCacheV3(
                data_dir=data_dir, cache_dir=cache_dir,
                cache_enabled=True, strict_mode=False, verify_integrity=True,
            )
            for pkl_file in Path(cache_dir).glob("cached_data_*.pkl"):
                sha_path = pkl_file.with_suffix(".sha256")
                meta_path = pkl_file.with_suffix(".meta.json")
                if not sha_path.exists():
                    cache_v3._save_checksum(pkl_file)
                    logger.info(f"[V3] Checksum: {sha_path.name}")
                if not meta_path.exists():
                    stem = pkl_file.stem
                    rest = stem[12:] if stem.startswith("cached_data_") else stem
                    parts = rest.rsplit("_", 1)
                    ck = parts[0] if len(parts) == 2 else rest
                    dh = parts[1] if len(parts) == 2 else ""
                    cache_v3._save_metadata(
                        cache_path=pkl_file, cache_key=ck, data_hash=dh,
                        encode_mode="train",
                        include_whole_words=include_whole_words,
                        include_syllables=include_syllables,
                        include_sep=include_sep,
                        max_seq_length=max_seq_length,
                        alignment_format="autoregressive_v2",
                        sample_count=len(formatted_data),
                        file_size_mb=pkl_file.stat().st_size / (1024 * 1024),
                    )
                    logger.info(f"[V3] Metadata: {meta_path.name}")
            logger.info("[V3] Integrity dosyaları hazır")
        except Exception as e:
            logger.warning(f"[V3] Integrity oluşturma hatası (devam ediliyor): {e}")

    # Cache dosyalarını listele
    logger.info("\n[4] Cache dosyaları:")
    cache_path = Path(cache_dir)
    if cache_path.exists():
        cache_files = list(cache_path.glob("cached_data_*.pkl"))
        for cache_file in cache_files:
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            meta_ok = "OK" if cache_file.with_suffix(".meta.json").exists() else "?"
            sha_ok = "OK" if cache_file.with_suffix(".sha256").exists() else "?"
            logger.info(
                f"   📁 {cache_file.name} ({file_size_mb:.2f} MB) "
                f"[checksum={sha_ok}, meta={meta_ok}]"
            )

    logger.info("\n" + "="*60)
    logger.info("[OK] CACHE HAZIRLAMA TAMAMLANDI!")
    logger.info("="*60)
    logger.info("\n📤 Colab'a yükleme:")
    logger.info(f"   1. Cache dizini: {cache_dir}")
    logger.info(f"   2. Tüm .pkl dosyalarını Colab'a yükle")
    logger.info(f"   3. Colab'ta aynı cache_dir'i kullan")
    logger.info(f"   4. Cache otomatik kullanılacak (hızlı başlangıç)")
    
    return cache_dir


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache hazırlama scripti")
    parser.add_argument("--data-dir", type=str, default="education", help="Eğitim verisi dizini")
    parser.add_argument("--cache-dir", type=str, default=".cache/preprocessed_data", help="Cache dizini")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max sequence length (None ise TOKENIZER_CONFIG'ten alınır)")
    #  YENİ: None default - BPE_DETAILED_CONFIG'ten otomatik alınır
    # argparse'da None default için özel bir yaklaşım kullanıyoruz
    def str_to_bool_or_none(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif v.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            return None
    
    parser.add_argument("--include-whole-words", type=str_to_bool_or_none, default=None, nargs='?', const=True, help="Include whole words (None ise BPE_DETAILED_CONFIG'ten alınır)")
    parser.add_argument("--include-syllables", type=str_to_bool_or_none, default=None, nargs='?', const=True, help="Include syllables (None ise BPE_DETAILED_CONFIG'ten alınır)")
    parser.add_argument("--include-sep", type=str_to_bool_or_none, default=None, nargs='?', const=True, help="Include SEP (None ise BPE_DETAILED_CONFIG'ten alınır)")
    parser.add_argument("--no-clear-cache", action="store_true", default=False, help="Eski cache'i temizleme (default: temizler)")
    
    args = parser.parse_args()
    
    try:
        cache_dir = prepare_cache(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            max_seq_length=args.max_seq_length,
            include_whole_words=args.include_whole_words,
            include_syllables=args.include_syllables,
            include_sep=args.include_sep,
            clear_old_cache=not args.no_clear_cache,  # --no-clear-cache varsa False
        )
        print(f"\n[OK] Başarılı! Cache dizini: {cache_dir}")
        return 0
    except Exception as e:
        logger.error(f" Hata: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

