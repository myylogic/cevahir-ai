#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: check_vocab.py
Modül: tokenizer_management
Görev: Vocab kontrol scripti - Vocab dosyasında basit kelimelerin ve karakterlerin
       olup olmadığını kontrol eder. Vocab içeriğini analiz eder ve raporlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (vocab kontrol scripti)
- Design Patterns: Script Pattern (standalone utility script)
- Endüstri Standartları: Vocab validation ve debugging

KULLANIM:
- Vocab dosyasını kontrol etmek için
- Debugging ve analiz için
- Standalone script olarak çalıştırılır

BAĞIMLILIKLAR:
- json: Vocab dosyası okuma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import json
import sys

vocab_path = "data/vocab_lib/vocab.json"

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

print(f"Vocab boyutu: {len(vocab)}")
print("\n" + "="*80)
print("KARAKTERLER VOCAB'DA MI?")
print("="*80)

chars = ['b', 'u', 'i', 'r', ' ', 't', 'e', 's', 'c', 'ü', 'm', 'l']
for c in chars:
    in_vocab = c in vocab
    print(f"'{c}': {in_vocab}")

print("\n" + "="*80)
print("BASİT KELİMELER VOCAB'DA MI?")
print("="*80)

words = ['bu</w>', 'bir</w>', 'test</w>', 'yapay</w>', 'zeka</w>']
for word in words:
    in_vocab = word in vocab
    if in_vocab:
        word_id = vocab[word].get('id', 'N/A')
        print(f"'{word}': EVET (ID: {word_id})")
    else:
        print(f"'{word}': HAYIR")

print("\n" + "="*80)
print("VOCAB'DAKİ İLK 30 TOKEN")
print("="*80)
for i, token in enumerate(list(vocab.keys())[:30]):
    token_id = vocab[token].get('id', 'N/A')
    print(f"{i+1:2d}. '{token}' (ID: {token_id})")

print("\n" + "="*80)
print("VOCAB'DA 'b' İLE BAŞLAYAN TOKEN'LAR (İLK 20)")
print("="*80)
b_tokens = [t for t in vocab.keys() if t.startswith('b')][:20]
for token in b_tokens:
    token_id = vocab[token].get('id', 'N/A')
    print(f"  '{token}' (ID: {token_id})")

