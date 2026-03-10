#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: test_oov_fallback_debug.py
Modül: tokenizer_management
Görev: OOV (Out-of-Vocabulary) Syllable Fallback Mekanizması Debug Testi -
       Vocab'da karakterlerin olup olmadığını ve OOV fallback mekanizmasının
       çalışıp çalışmadığını test eder. Debugging ve troubleshooting için.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (OOV fallback test scripti)
- Design Patterns: Script Pattern (standalone test tool)
- Endüstri Standartları: OOV handling validation

KULLANIM:
- OOV fallback mekanizmasını test etmek için
- Vocab karakter kontrolü için
- Debugging ve troubleshooting için

BAĞIMLILIKLAR:
- TokenizerCore: Tokenization işlemleri
- get_turkish_config, get_bpe_detailed_config: Config yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import sys
import os
import json
import logging

# Proje root'unu path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import get_turkish_config, get_bpe_detailed_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vocab_characters():
    """Vocab'da karakterlerin olup olmadığını kontrol et"""
    vocab_path = "data/vocab_lib/vocab.json"
    
    if not os.path.exists(vocab_path):
        logger.error(f"Vocab dosyası bulunamadı: {vocab_path}")
        return False
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Türkçe karakterler
    turkish_chars = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z']
    turkish_chars_with_w = [c + '</w>' for c in turkish_chars]
    
    # Kontrol et
    missing_chars = []
    found_chars = []
    
    for char in turkish_chars:
        if char not in vocab:
            missing_chars.append(char)
        else:
            found_chars.append(char)
    
    for char_w in turkish_chars_with_w:
        if char_w not in vocab:
            missing_chars.append(char_w)
        else:
            found_chars.append(char_w)
    
    logger.info(f"Vocab'da bulunan karakterler: {len(found_chars)}/{len(turkish_chars) + len(turkish_chars_with_w)}")
    if missing_chars:
        logger.warning(f"Vocab'da eksik karakterler: {missing_chars[:10]}... (toplam {len(missing_chars)})")
    else:
        logger.info("Tüm Türkçe karakterler vocab'da mevcut!")
    
    return len(missing_chars) == 0

def test_oov_fallback():
    """OOV syllable fallback mekanizmasını test et"""
    logger.info("=" * 80)
    logger.info("OOV SYLLABLE FALLBACK TESTİ")
    logger.info("=" * 80)
    
    # Config
    config = {
        "vocab_path": "data/vocab_lib/vocab.json",
        "merges_path": "data/merges_lib/merges.txt",
        **get_bpe_detailed_config(),
        **get_turkish_config(),
    }
    
    # TokenizerCore başlat
    tokenizer = TokenizerCore(config=config)
    
    # Test metinleri (vocab'da olmayan kelimeler)
    test_texts = [
        "xylophone",  # İngilizce, vocab'da olmayabilir
        "çokuzgaralı",  # Türkçe, vocab'da olmayabilir
        "supercalifragilisticexpialidocious",  # Çok uzun kelime
        "test123",  # Sayı içeren
    ]
    
    logger.info("\nTest metinleri encode ediliyor...")
    logger.info("include_syllables=False, use_syllables_for_oov=True olmalı")
    
    for text in test_texts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test metni: '{text}'")
        logger.info(f"{'='*60}")
        
        try:
            tokens, token_ids = tokenizer.encode(
                text,
                mode="inference",
                include_whole_words=True,
                include_syllables=False,  # OOV fallback için False olmalı
                include_sep=True
            )
            
            # UNK sayısını kontrol et
            unk_id = tokenizer._special_ids().get("<UNK>")
            if unk_id is not None:
                unk_count = sum(1 for tid in token_ids if tid == unk_id)
                unk_ratio = unk_count / len(token_ids) if token_ids else 0.0
                
                logger.info(f"Token sayısı: {len(tokens)}")
                logger.info(f"Token ID sayısı: {len(token_ids)}")
                logger.info(f"UNK sayısı: {unk_count}")
                logger.info(f"UNK oranı: {unk_ratio:.2%}")
                logger.info(f"Token'lar: {tokens[:20]}...")  # İlk 20 token
                
                if unk_count > 0:
                    logger.warning(f"UNK token bulundu! OOV fallback çalışmamış olabilir.")
                else:
                    logger.info("UNK token yok! OOV fallback başarılı olabilir.")
            else:
                logger.warning("UNK ID bulunamadı!")
                
        except Exception as e:
            logger.error(f"Encode hatası: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("OOV FALLBACK DEBUG TESTİ")
    logger.info("=" * 80)
    
    # 1. Vocab karakter kontrolü
    logger.info("\n1. VOCAB KARAKTER KONTROLÜ")
    logger.info("-" * 80)
    vocab_ok = test_vocab_characters()
    
    # 2. OOV fallback testi
    logger.info("\n2. OOV SYLLABLE FALLBACK TESTİ")
    logger.info("-" * 80)
    test_oov_fallback()
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST TAMAMLANDI")
    logger.info("=" * 80)
