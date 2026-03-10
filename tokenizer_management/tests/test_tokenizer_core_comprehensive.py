#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TokenizerCore Kapsamlı Test Dosyası
===================================
TokenizerCore'un tüm fonksiyonlarını test eder ve kritik sorunları tespit eder.

Kullanım:
    python tokenizer_management/tests/test_tokenizer_core_comprehensive.py
    python tokenizer_management/tests/test_tokenizer_core_comprehensive.py --test-encode
    python tokenizer_management/tests/test_tokenizer_core_comprehensive.py --test-training-data
    python tokenizer_management/tests/test_tokenizer_core_comprehensive.py --test-all
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ✅ Windows Unicode desteği için
if os.name == 'nt':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError


class Colors:
    """Terminal renkleri"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Başlık yazdır"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_success(text: str):
    """Başarı mesajı"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Hata mesajı"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    """Uyarı mesajı"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """Bilgi mesajı"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


# ============================================================================
# TEST 1: ENCODE/DECODE FONKSİYONLARI
# ============================================================================

def test_encode_decode(tokenizer: TokenizerCore):
    """Test 1: Encode/Decode fonksiyonları"""
    print_header("TEST 1: ENCODE/DECODE FONKSİYONLARI")
    
    test_cases = [
        "Merhaba dünya",
        "Bu bir test metnidir.",
        "Türkçe karakterler: ığüşöçİĞÜŞÖÇ",
        "Sayılar: 123 456 789",
        "Özel karakterler: !@#$%^&*()",
        "",  # Boş string
    ]
    
    all_passed = True
    
    print_info("Tekil encode/decode testleri...")
    for i, text in enumerate(test_cases, 1):
        try:
            # Encode
            tokens, token_ids = tokenizer.encode(text, mode="inference")
            
            # Kontrol: Boş string için özel kontrol
            if text == "":
                if len(token_ids) == 0:
                    print_success(f"Test {i}: Boş string doğru işlendi (token_ids boş)")
                else:
                    print_error(f"Test {i}: Boş string için token_ids boş olmalı ama {len(token_ids)} token var")
                    all_passed = False
                continue
            
            # Kontrol: Token IDs geçerli mi?
            vocab_size = tokenizer.get_vocab_size()
            invalid_ids = [tid for tid in token_ids if tid < 0 or tid >= vocab_size]
            if invalid_ids:
                print_error(f"Test {i}: Geçersiz token ID'ler: {invalid_ids}")
                all_passed = False
                continue
            
            # Decode
            decoded_text = tokenizer.decode(token_ids, remove_specials=True)
            
            # Kontrol: Decode sonucu mantıklı mı?
            if len(decoded_text) == 0 and len(text) > 0:
                print_warning(f"Test {i}: Decode sonucu boş ama orijinal metin boş değil")
                print_info(f"  Orijinal: '{text[:50]}'")
                print_info(f"  Token IDs: {token_ids[:10]}...")
            else:
                print_success(f"Test {i}: Encode/Decode başarılı")
                print_info(f"  Orijinal: '{text[:50]}'")
                print_info(f"  Decoded: '{decoded_text[:50]}'")
                print_info(f"  Token sayısı: {len(token_ids)}")
        
        except Exception as e:
            print_error(f"Test {i}: Hata - {e}")
            all_passed = False
    
    print_info("\nBatch encode/decode testleri...")
    try:
        # Batch encode
        batch_results = tokenizer.batch_encode(
            test_cases,
            mode="inference",
            skip_invalid=True
        )
        
        print_success(f"Batch encode başarılı: {len(batch_results)} sonuç")
        
        # Batch decode
        all_token_ids = [ids for _, ids in batch_results]
        decoded_texts = tokenizer.batch_decode(
            all_token_ids,
            remove_specials=True,
            skip_invalid=True
        )
        
        print_success(f"Batch decode başarılı: {len(decoded_texts)} sonuç")
        
    except Exception as e:
        print_error(f"Batch encode/decode hatası: {e}")
        all_passed = False
    
    return all_passed


def test_encode_modes(tokenizer: TokenizerCore):
    """Test 2: Farklı encode modları"""
    print_header("TEST 2: ENCODE MODLARI (TRAIN vs INFERENCE)")
    
    test_text = "Bu bir test metnidir."
    all_passed = True
    
    modes = ["train", "inference"]
    include_options = [
        {"include_whole_words": True, "include_syllables": False, "include_sep": False},
        {"include_whole_words": True, "include_syllables": True, "include_sep": False},
        {"include_whole_words": False, "include_syllables": True, "include_sep": False},
        {"include_whole_words": True, "include_syllables": False, "include_sep": True},
    ]
    
    for mode in modes:
        print_info(f"\nMode: {mode}")
        for i, opts in enumerate(include_options, 1):
            try:
                tokens, token_ids = tokenizer.encode(
                    test_text,
                    mode=mode,
                    **opts
                )
                
                print_success(f"  Config {i}: {len(token_ids)} token")
                print_info(f"    Options: {opts}")
                print_info(f"    İlk 10 token: {tokens[:10]}")
                
            except Exception as e:
                print_error(f"  Config {i}: Hata - {e}")
                all_passed = False
    
    return all_passed


# ============================================================================
# TEST 3: VOCAB İŞLEMLERİ
# ============================================================================

def test_vocab_operations(tokenizer: TokenizerCore):
    """Test 3: Vocab işlemleri"""
    print_header("TEST 3: VOCAB İŞLEMLERİ")
    
    all_passed = True
    
    try:
        # Vocab alma
        vocab = tokenizer.get_vocab()
        vocab_size = tokenizer.get_vocab_size()
        
        print_success(f"Vocab boyutu: {vocab_size:,}")
        
        # Special token kontrolü
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        missing_specials = []
        
        for token in special_tokens:
            if token not in vocab:
                missing_specials.append(token)
        
        if missing_specials:
            print_error(f"Eksik special token'lar: {missing_specials}")
            all_passed = False
        else:
            print_success("Tüm special token'lar mevcut")
        
        # Special token ID'leri
        special_ids = {}
        for token in special_tokens:
            token_info = vocab.get(token)
            if isinstance(token_info, dict):
                token_id = token_info.get("id")
                if token_id is not None:
                    special_ids[token] = token_id
                    print_info(f"  {token}: ID={token_id}")
        
        # Summary
        summary = tokenizer.summary()
        print_info(f"\nSummary:")
        print_info(f"  Vocab size: {summary['vocab_size']:,}")
        print_info(f"  Merges count: {summary['merges_count']:,}")
        print_info(f"  Has specials: {summary['has_specials']}")
        
    except Exception as e:
        print_error(f"Vocab işlemleri hatası: {e}")
        all_passed = False
    
    return all_passed


# ============================================================================
# TEST 4: LOAD_TRAINING_DATA - KRİTİK TEST
# ============================================================================

def test_load_training_data(tokenizer: TokenizerCore):
    """Test 4: load_training_data - KRİTİK TEST"""
    print_header("TEST 4: LOAD_TRAINING_DATA - KRİTİK TEST")
    
    all_passed = True
    critical_issues = []
    
    try:
        print_info("load_training_data() çağrılıyor...")
        raw_data = tokenizer.load_training_data(
            encode_mode="train",
            include_whole_words=True,
            include_syllables=False,
            include_sep=False,
        )
        
        print_success(f"✅ {len(raw_data):,} örnek yüklendi")
        
        if len(raw_data) == 0:
            print_error("❌ Hiç örnek yüklenmedi!")
            return False
        
        # İlk 10 örneği detaylı kontrol et
        print_info("\n📊 İlk 10 örneği detaylı kontrol ediliyor...\n")
        
        for i, (inp_ids, tgt_ids) in enumerate(raw_data[:10], 1):
            print(f"Örnek {i}:")
            print(f"  inp_ids (ilk 20): {inp_ids[:20]}")
            print(f"  tgt_ids (ilk 20): {tgt_ids[:20]}")
            print(f"  Uzunluk: inp={len(inp_ids)}, tgt={len(tgt_ids)}")
            
            # KRİTİK KONTROL 1: inp_ids ve tgt_ids eşit mi?
            if inp_ids == tgt_ids:
                print_warning(f"  ⚠️  UYARI: inp_ids ve tgt_ids AYNI!")
                print_warning(f"     Bu autoregressive training için YANLIŞ olabilir!")
                critical_issues.append(f"Örnek {i}: inp_ids == tgt_ids (aynı)")
            else:
                print_success(f"  ✅ inp_ids != tgt_ids (farklı)")
                # Farkları göster
                diffs = [j for j, (a, b) in enumerate(zip(inp_ids, tgt_ids)) if a != b]
                if diffs:
                    print_info(f"  Fark pozisyonları (ilk 10): {diffs[:10]}")
            
            # KRİTİK KONTROL 2: Uzunluk kontrolü
            if len(inp_ids) != len(tgt_ids):
                print_error(f"  ❌ HATA: Uzunluklar farklı!")
                critical_issues.append(f"Örnek {i}: Uzunluk farkı (inp={len(inp_ids)}, tgt={len(tgt_ids)})")
                all_passed = False
            else:
                print_success(f"  ✅ Uzunluklar eşit")
            
            # KRİTİK KONTROL 3: Autoregressive format kontrolü
            # Autoregressive için: tgt_ids[i] == inp_ids[i+1] olmalı (i < len(tgt_ids))
            if len(inp_ids) > 1 and len(tgt_ids) > 0:
                matches = 0
                mismatches = []
                for j in range(min(len(tgt_ids), len(inp_ids) - 1)):
                    if tgt_ids[j] == inp_ids[j+1]:
                        matches += 1
                    else:
                        mismatches.append((j, inp_ids[j+1], tgt_ids[j]))
                
                if matches == len(tgt_ids):
                    print_success(f"  ✅ Autoregressive format DOĞRU (target[i] = input[i+1])")
                else:
                    print_warning(f"  ⚠️  Autoregressive format YANLIŞ: {matches}/{len(tgt_ids)} eşleşme")
                    if mismatches:
                        print_info(f"  İlk 5 uyuşmazlık: {mismatches[:5]}")
                    critical_issues.append(f"Örnek {i}: Autoregressive format yanlış")
            
            # KRİTİK KONTROL 4: BOS/EOS kontrolü
            # TokenizerCore'dan gelen veri BOS/EOS içermemeli (TrainingService'te eklenecek)
            vocab = tokenizer.get_vocab()
            BOS_ID = vocab.get("<BOS>", {}).get("id", 1) if isinstance(vocab.get("<BOS>"), dict) else vocab.get("<BOS>", 1)
            EOS_ID = vocab.get("<EOS>", {}).get("id", 2) if isinstance(vocab.get("<EOS>"), dict) else vocab.get("<EOS>", 2)
            
            if BOS_ID in inp_ids or BOS_ID in tgt_ids:
                print_warning(f"  ⚠️  UYARI: BOS token TokenizerCore'dan geliyor (beklenmiyor)")
                critical_issues.append(f"Örnek {i}: BOS token TokenizerCore'dan geliyor")
            
            if EOS_ID in inp_ids or EOS_ID in tgt_ids:
                print_warning(f"  ⚠️  UYARI: EOS token TokenizerCore'dan geliyor (beklenmiyor)")
                critical_issues.append(f"Örnek {i}: EOS token TokenizerCore'dan geliyor")
            
            print("---")
        
        # Özet
        print_header("KRİTİK SORUN ÖZETİ")
        if critical_issues:
            print_error(f"❌ {len(critical_issues)} kritik sorun tespit edildi:")
            for issue in critical_issues:
                print_error(f"  - {issue}")
            all_passed = False
        else:
            print_success("✅ Tüm kritik kontroller başarılı!")
        
        # İstatistikler
        print_info("\n📊 İstatistikler:")
        total_samples = len(raw_data)
        same_samples = sum(1 for inp, tgt in raw_data if inp == tgt)
        diff_samples = total_samples - same_samples
        
        print_info(f"  Toplam örnek: {total_samples:,}")
        print_info(f"  Aynı (inp==tgt): {same_samples:,} ({same_samples/total_samples*100:.1f}%)")
        print_info(f"  Farklı (inp!=tgt): {diff_samples:,} ({diff_samples/total_samples*100:.1f}%)")
        
        if same_samples > 0:
            print_warning(f"  ⚠️  {same_samples} örnekte inp_ids == tgt_ids (autoregressive için yanlış!)")
            all_passed = False
        
    except TokenizerCoreError as e:
        print_error(f"TokenizerCoreError: {e}")
        all_passed = False
    except Exception as e:
        print_error(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


# ============================================================================
# TEST 5: ENCODE/DECODE ROUNDTRIP
# ============================================================================

def test_encode_decode_roundtrip(tokenizer: TokenizerCore):
    """Test 5: Encode/Decode roundtrip (encode → decode → encode)"""
    print_header("TEST 5: ENCODE/DECODE ROUNDTRIP")
    
    test_cases = [
        "Merhaba dünya",
        "Bu bir test metnidir.",
        "Türkçe karakterler: ığüşöçİĞÜŞÖÇ",
    ]
    
    all_passed = True
    
    for i, original_text in enumerate(test_cases, 1):
        try:
            # Encode
            tokens1, token_ids1 = tokenizer.encode(original_text, mode="inference")
            
            # Decode
            decoded_text = tokenizer.decode(token_ids1, remove_specials=True)
            
            # Tekrar encode
            tokens2, token_ids2 = tokenizer.encode(decoded_text, mode="inference")
            
            # Kontrol: Token ID'ler aynı mı?
            if token_ids1 == token_ids2:
                print_success(f"Test {i}: Roundtrip başarılı (token IDs aynı)")
            else:
                print_warning(f"Test {i}: Roundtrip sonucu farklı token IDs")
                print_info(f"  Orijinal: '{original_text[:50]}'")
                print_info(f"  Decoded: '{decoded_text[:50]}'")
                print_info(f"  İlk encode: {token_ids1[:10]}")
                print_info(f"  İkinci encode: {token_ids2[:10]}")
        
        except Exception as e:
            print_error(f"Test {i}: Hata - {e}")
            all_passed = False
    
    return all_passed


# ============================================================================
# TEST 6: BATCH İŞLEMLER
# ============================================================================

def test_batch_operations(tokenizer: TokenizerCore):
    """Test 6: Batch işlemleri"""
    print_header("TEST 6: BATCH İŞLEMLER")
    
    test_texts = [
        "Merhaba dünya",
        "Bu bir test metnidir.",
        "Türkçe karakterler: ığüşöçİĞÜŞÖÇ",
        "Sayılar: 123 456 789",
        "Özel karakterler: !@#$%^&*()",
    ]
    
    all_passed = True
    
    try:
        # Batch encode
        batch_results = tokenizer.batch_encode(
            test_texts,
            mode="inference",
            skip_invalid=True
        )
        
        print_success(f"Batch encode: {len(batch_results)}/{len(test_texts)} başarılı")
        
        # Batch decode
        all_token_ids = [ids for _, ids in batch_results]
        decoded_texts = tokenizer.batch_decode(
            all_token_ids,
            remove_specials=True,
            skip_invalid=True
        )
        
        print_success(f"Batch decode: {len(decoded_texts)}/{len(batch_results)} başarılı")
        
        # Kontrol: Sonuç sayıları eşleşiyor mu?
        if len(batch_results) == len(test_texts) and len(decoded_texts) == len(batch_results):
            print_success("✅ Tüm batch işlemleri başarılı")
        else:
            print_warning(f"⚠️  Bazı örnekler atlandı: {len(batch_results)}/{len(test_texts)}")
            all_passed = False
        
    except Exception as e:
        print_error(f"Batch işlemleri hatası: {e}")
        all_passed = False
    
    return all_passed


# ============================================================================
# ANA TEST FONKSİYONU
# ============================================================================

def run_all_tests(config: Dict[str, Any]):
    """Tüm testleri çalıştır"""
    print_header("TOKENIZERCORE KAPSAMLI TEST SUITE")
    
    # TokenizerCore oluştur
    try:
        print_info("TokenizerCore başlatılıyor...")
        tokenizer = TokenizerCore(config)
        tokenizer.finalize_vocab()
        print_success("TokenizerCore başlatıldı")
    except Exception as e:
        print_error(f"TokenizerCore başlatılamadı: {e}")
        return False
    
    results = {}
    
    # Test 1: Encode/Decode
    results["encode_decode"] = test_encode_decode(tokenizer)
    
    # Test 2: Encode Modları
    results["encode_modes"] = test_encode_modes(tokenizer)
    
    # Test 3: Vocab İşlemleri
    results["vocab_operations"] = test_vocab_operations(tokenizer)
    
    # Test 4: Load Training Data (KRİTİK)
    results["load_training_data"] = test_load_training_data(tokenizer)
    
    # Test 5: Roundtrip
    results["roundtrip"] = test_encode_decode_roundtrip(tokenizer)
    
    # Test 6: Batch İşlemler
    results["batch_operations"] = test_batch_operations(tokenizer)
    
    # Özet
    print_header("TEST ÖZETİ")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = "✅ BAŞARILI" if passed else "❌ BAŞARISIZ"
        print(f"  {test_name:30s}: {status}")
    
    print(f"\n{'='*80}")
    print(f"Toplam: {total_tests} test")
    print(f"Başarılı: {passed_tests} test")
    print(f"Başarısız: {failed_tests} test")
    print(f"{'='*80}")
    
    if failed_tests == 0:
        print_success("\n✅ Tüm testler başarılı!")
        return True
    else:
        print_error(f"\n❌ {failed_tests} test başarısız!")
        return False


def main():
    parser = argparse.ArgumentParser(description="TokenizerCore kapsamlı test suite")
    parser.add_argument("--test-all", action="store_true", help="Tüm testleri çalıştır")
    parser.add_argument("--test-encode", action="store_true", help="Sadece encode/decode testleri")
    parser.add_argument("--test-training-data", action="store_true", help="Sadece load_training_data testi (KRİTİK)")
    parser.add_argument("--data-dir", type=str, default="education", help="Data dizini")
    parser.add_argument("--vocab-path", type=str, default="data/vocab_lib/vocab.json", help="Vocab dosyası")
    parser.add_argument("--merges-path", type=str, default="data/merges_lib/merges.txt", help="Merges dosyası")
    
    args = parser.parse_args()
    
    config = {
        "data_dir": args.data_dir,
        "vocab_path": args.vocab_path,
        "merges_path": args.merges_path,
        "bpe_rebuild": False,
        "device": "cpu",
    }
    
    if args.test_all:
        return 0 if run_all_tests(config) else 1
    elif args.test_training_data:
        # Sadece kritik test
        tokenizer = TokenizerCore(config)
        tokenizer.finalize_vocab()
        result = test_load_training_data(tokenizer)
        return 0 if result else 1
    elif args.test_encode:
        # Sadece encode testleri
        tokenizer = TokenizerCore(config)
        tokenizer.finalize_vocab()
        result1 = test_encode_decode(tokenizer)
        result2 = test_encode_modes(tokenizer)
        return 0 if (result1 and result2) else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
