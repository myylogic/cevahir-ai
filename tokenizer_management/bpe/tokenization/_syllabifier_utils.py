# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: _syllabifier_utils.py
Modül: tokenizer_management/bpe/tokenization
Görev: Syllabifier yardımcı fonksiyonları - Türkçe heceleme işlemleri için
       utility fonksiyonları. Diakritik kaldırma, heceleme algoritması ve
       Türkçe karakter işleme fonksiyonları içerir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (syllabifier utility fonksiyonları)
- Design Patterns: Utility Pattern (yardımcı fonksiyonlar)
- Endüstri Standartları: Türkçe heceleme algoritmaları

KULLANIM:
- Syllabifier sınıfı tarafından kullanılır
- Türkçe heceleme işlemleri için
- Diakritik kaldırma ve karakter normalizasyonu için

BAĞIMLILIKLAR:
- unicodedata: Unicode karakter analizi
- torch: Tensor işlemleri (opsiyonel)

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
from typing import List, Set, Optional
import torch

logger = logging.getLogger(__name__)

def strip_diacritics(text: str) -> str:
    """
    Türkçe harfleri Latin ASCII'ye çevirir, diakritikleri kaldırır.
    Örn: 'İıŞşĞğÜüÖöÇç' -> 'IiSsGgUuOoCc'
    """
    tr_map = {
        "ç": "c", "Ç": "C",
        "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S",
        "ü": "u", "Ü": "U",
    }
    decomposed = unicodedata.normalize("NFKD", text)
    out = []
    for ch in decomposed:
        if ch in tr_map:
            out.append(tr_map[ch])
        elif not unicodedata.combining(ch):
            out.append(ch)
    return "".join(out)

def is_vowel(ch: str, vowels: Set[str]) -> bool:
    return ch in vowels

def optimize_syllables(raw_sylls: List[str], vowels: Set[str]) -> List[str]:
    """
    Tek ünsüzlü heceleri önceki hecenin coda'sına ekler.
    """
    optimized: List[str] = []
    buffer = ''
    for syl in raw_sylls:
        if len(syl) == 1 and syl not in vowels and buffer and buffer[-1] in vowels:
            buffer += syl
        else:
            if buffer:
                optimized.append(buffer)
            buffer = syl
    if buffer:
        optimized.append(buffer)
    return optimized

def split_into_syllables_impl(
    word: str,
    vowels: Set[str],
    valid_initial_clusters: Set[str],
    use_gpu: bool = False,
    device: Optional[torch.device] = None
) -> List[str]:
    """
    Türkçe heceleme kurallarıyla ayırır.
    """
    if not word or not isinstance(word, str):
        return []
    w = word  # casefold() KULLANMA! Harfler kaybolur!
    n = len(w)
    
    if use_gpu and device:
        with torch.amp.autocast('cuda'):
            try:
                chars = torch.tensor([ord(c) for c in w], device=device, dtype=torch.long)
                vowels_tensor = torch.tensor([ord(v) for v in vowels], device=device)
                is_vowel_tensor = torch.isin(chars, vowels_tensor)
                clusters_tensor = torch.tensor([ord(c) for cluster in valid_initial_clusters for c in cluster], device=device)
                
                raw_sylls = []
                i = 0
                first = True
                
                while i < n:
                    onset = ''
                    if first:
                        while i < n and not is_vowel_tensor[i].item():
                            onset += w[i]
                            i += 1
                    else:
                        temp_tensor = chars[i:i+2]
                        temp_str = ''.join(chr(c.item()) for c in temp_tensor)
                        if i + 1 < n and (len(temp_str) == 1 or temp_str in valid_initial_clusters):
                            onset += w[i]
                            i += 1
                        else:
                            break
                    if i < n and is_vowel_tensor[i].item():
                        nucleus = w[i]
                        i += 1
                    else:
                        if onset:
                            raw_sylls.append(onset)
                        break
                    coda = ''
                    while i < n and not is_vowel_tensor[i].item() and len(coda) < 2 and (
                        i + 1 == n or (i + 1 < n and not is_vowel_tensor[i+1].item())
                    ):
                        coda += w[i]
                        i += 1
                    raw_sylls.append(onset + nucleus + coda)
                    first = False
                
                if i < n:
                    raw_sylls.append(w[i:])
                return optimize_syllables(raw_sylls, vowels)
            except Exception as e:
                logger.error(f"[Syllabifier] GPU hatası: {e}")
                return _split_into_syllables_impl_cpu(w, vowels, valid_initial_clusters)
    else:
        return _split_into_syllables_impl_cpu(w, vowels, valid_initial_clusters)

def _split_into_syllables_impl_cpu(
    word: str,
    vowels: Set[str],
    valid_initial_clusters: Set[str]
) -> List[str]:
    n = len(word)
    i = 0
    raw_sylls: List[str] = []
    first = True

    while i < n:
        onset = ''
        # Onset
        if first:
            while i < n and not is_vowel(word[i], vowels):
                onset += word[i]
                i += 1
        else:
            while i < n and not is_vowel(word[i], vowels):
                temp = onset + word[i]
                if len(temp) == 1 or temp in valid_initial_clusters:
                    onset = temp
                    i += 1
                else:
                    break
        # Nucleus
        if i < n and is_vowel(word[i], vowels):
            nucleus = word[i]
            i += 1
        else:
            if onset:
                raw_sylls.append(onset)
            break
        # Coda
        coda = ''
        while i < n and not is_vowel(word[i], vowels) and len(coda) < 2 and (
            i + 1 == n or (i + 1 < n and not is_vowel(word[i+1], vowels))
        ):
            coda += word[i]
            i += 1
        raw_sylls.append(onset + nucleus + coda)
        first = False

    if i < n:
        raw_sylls.append(word[i:])
    return optimize_syllables(raw_sylls, vowels)

def syllabify_word(word: str, vowels: Set[str], allowed_onsets: Set[str]) -> List[str]:
    """
    Tek bir kelimeyi hecelerine ayırır. Noktalama sona eklenir.
    """
    if not isinstance(word, str):
        raise TypeError(f"Kelime str olmalı, got {type(word)}")
    w = word.strip()
    if not w:
        return []
    # Noktalama
    if not w[-1].isalnum():
        return syllabify_word(w[:-1], vowels, allowed_onsets) + [w[-1]]
    return split_into_syllables_impl(w, vowels, allowed_onsets)