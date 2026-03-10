# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_manager_utils.py
Modül: tokenizer_management/bpe
Görev: BPE Manager yardımcı fonksiyonları - Vocab ve merges işlemleri için
       utility fonksiyonları. Default vocab oluşturma, vocab normalizasyonu,
       valid ID kontrolü, JSON okuma/yazma işlemleri.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (utility fonksiyonları)
- Design Patterns: Utility Pattern (yardımcı fonksiyonlar)
- Endüstri Standartları: Vocab yönetimi, JSON serialization

KULLANIM:
- BPEManager, BPEEncoder, BPEDecoder, BPETrainer tarafından kullanılır
- Vocab normalizasyonu için
- Default vocab oluşturma için
- Valid ID kontrolü için

BAĞIMLILIKLAR:
- json: JSON serialization
- DEFAULT_SPECIALS: Özel token tanımları

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
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing import Final

logger = logging.getLogger(__name__)

# ---- Şema ve sabitler --------------------------------------------------------

# Varsayılan özel token ID'leri (sabit ve benzersiz tutulur)
DEFAULT_SPECIALS: Final[Dict[str, int]] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "<SEP>": 4,
}

# ---- Yardımcılar: Dosya Sistemi & JSON ---------------------------------------

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def read_json(path: str) -> Any:
    """
    JSON dosyasını okur. Hata durumlarını üst katmanda ele almak üzere aynen iletir.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    """
    JSON dosyasını atomik olarak yazar: .tmp dosyasına yazıp os.replace ile taşır.
    Böylece yarım yazım/bozuk dosya riski minimize edilir.
    """
    _ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

# ---- Vocab Normalizasyonu & Doğrulama ----------------------------------------

def _normalize_token_map(token_map: Dict[str, Union[int, dict]]) -> Dict[str, dict]:
    """
    İçsel yardımcı: {token: int|{id:...}} -> {token: {"id": int, "total_freq": int, "positions": list}}
    """
    normalized: Dict[str, dict] = {}
    for tok, props in token_map.items():
        if isinstance(props, int):
            normalized[tok] = {"id": int(props), "total_freq": 0, "positions": []}
        elif isinstance(props, dict) and isinstance(props.get("id"), int):
            # total_freq / positions yoksa ekle
            nid = int(props["id"])
            tf = int(props.get("total_freq", 0))
            pos = props.get("positions", [])
            if not isinstance(pos, list):
                pos = []
            normalized[tok] = {"id": nid, "total_freq": tf, "positions": pos}
        else:
            raise ValueError(f"Geçersiz vocab girdisi: {tok} → {props!r}")
    return normalized

def normalize_vocab(raw: Dict[str, Union[int, dict]]) -> Dict[str, dict]:
    """
    Geriye dönük uyumlu normalizasyon:
    - Eski şema: {token: int | {id: ...}}
    - Yeni şema: {"version": 1, "specials": {...}, "tokens": {...}}
    ÇIKTI: {token: {"id": int, "total_freq": int, "positions": list}}
    """
    if not isinstance(raw, dict):
        raise ValueError("Vocab verisi dict olmalıdır.")

    # Yeni şema?
    if "version" in raw and "tokens" in raw and isinstance(raw["tokens"], dict):
        tokens_part = raw["tokens"]
        specials_part = raw.get("specials", {})
        merged: Dict[str, Union[int, dict]] = dict(tokens_part)

        # Specials içindekileri de token haritasına dahil et (varsa override etme, eksikse ekle)
        for tok, id_val in specials_part.items():
            if tok not in merged:
                merged[tok] = {"id": int(id_val), "total_freq": 0, "positions": []}
            else:
                # Eğer merged[tok] int ise; dict'e çevir
                if isinstance(merged[tok], int):
                    merged[tok] = {"id": int(merged[tok]), "total_freq": 0, "positions": []}

        out = _normalize_token_map(merged)

    else:
        # Eski şema
        out = _normalize_token_map(raw)

    validate_vocab(out)  # Tutarlılık kontrolü
    return out

def validate_vocab(vocab: Dict[str, dict]) -> None:
    """
    - Tüm girdilerde 'id' int ve benzersiz olmalı
    - ID’ler negatif olmamalı
    - Token anahtarları boş olmamalı
    """
    seen_ids: Set[int] = set()
    for tok, meta in vocab.items():
        if not isinstance(tok, str) or (not tok.strip() and not tok.startswith(' ')):
            raise ValueError(f"Boş veya geçersiz token anahtarı: {tok!r}")

        if not isinstance(meta, dict) or "id" not in meta:
            raise ValueError(f"Token meta geçersiz: {tok} → {meta!r}")

        tid = meta["id"]
        if not isinstance(tid, int):
            raise ValueError(f"Token ID int olmalı: {tok} → {tid!r}")
        if tid < 0:
            raise ValueError(f"Token ID negatif olamaz: {tok} → {tid}")
        if tid in seen_ids:
            raise ValueError(f"Çiftlenmiş ID tespit edildi: id={tid} (token={tok})")
        seen_ids.add(tid)

def get_valid_ids(vocab: Dict[str, dict]) -> Set[int]:
    """Geçerli ID kümesini döndürür (üyelik kontrollü doğrulama için)."""
    return {meta["id"] for meta in vocab.values()}

def next_id(vocab: Dict[str, dict]) -> int:
    """Monotonik artan yeni ID üretir (maks + 1)."""
    if not vocab:
        return max(DEFAULT_SPECIALS.values()) + 1
    return max(meta["id"] for meta in vocab.values()) + 1

# ---- Base Alphabet (Endüstri Standardı) --------------------------------------

def get_base_alphabet() -> List[str]:
    """
    Base alphabet: Tüm temel karakterler (endüstri standardı - GPT, BERT gibi).
    
    ENDÜSTRİ STANDARDI:
    - Base alphabet karakterleri vocab başlangıcında otomatik eklenir
    - Karakterler SADECE tek karakter formatında ("a", "a</w>" değil)
    - </w> ayrı bir token olarak eklenir
    - "a</w>" merge sonucu oluşur (vocab'a sonradan eklenir)
    
    Returns:
        List[str]: Base alphabet karakterleri (harfler, rakamlar, noktalama, boşluk, word boundary)
    """
    try:
        from tokenizer_management.config import TURKISH_CHARACTERS, ADDITIONAL_CHARACTERS
    except ImportError:
        # Fallback: Manuel karakter listesi
        TURKISH_CHARACTERS = [
            "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h",
            "ı", "i", "j", "k", "l", "m", "n", "o", "ö", "p",
            "r", "s", "ş", "t", "u", "ü", "v", "y", "z"
        ]
        ADDITIONAL_CHARACTERS = ["w", "x", "q"]
    
    # Harfler (Türkçe + ekstra) - KÜÇÜK VE BÜYÜK HARFLER
    # ✅ CASE-PRESERVING: Altyapı büyük/küçük harf kullanımını korumayı destekliyor
    lowercase_letters = TURKISH_CHARACTERS + ADDITIONAL_CHARACTERS
    
    # Türkçe karakterlerin büyük harflerini oluştur (Türkçe özel kurallarla)
    # ÖNEMLİ: Türkçe'de İ→i, I→ı dönüşümü var, bu yüzden manuel mapping gerekli
    turkish_uppercase_mapping = {
        "a": "A", "b": "B", "c": "C", "ç": "Ç", "d": "D", "e": "E", "f": "F",
        "g": "G", "ğ": "Ğ", "h": "H", "ı": "I", "i": "İ", "j": "J", "k": "K",
        "l": "L", "m": "M", "n": "N", "o": "O", "ö": "Ö", "p": "P", "r": "R",
        "s": "S", "ş": "Ş", "t": "T", "u": "U", "ü": "Ü", "v": "V", "y": "Y", "z": "Z",
        "w": "W", "x": "X", "q": "Q"
    }
    uppercase_letters = [turkish_uppercase_mapping.get(char, char.upper()) for char in lowercase_letters]
    
    # Arapça harfler (küçük ve büyük - Arapça'da büyük/küçük yok ama Unicode'da var)
    arabic_letters = [
        # Küçük harfler (Arapça'da resmi olarak yok ama Unicode'da mevcut)
        "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش",
        "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
        # Ekstra karakterler
        "ة", "ى", "ئ", "ء", "ؤ", "آ", "أ", "إ",
    ]
    
    # Kiril alfabesi (Rusça, Bulgarca, Sırpça vb. için)
    cyrillic_letters = [
        # Küçük harfler
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м",
        "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
        # Büyük harfler
        "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й", "К", "Л", "М",
        "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
    ]
    
    # Tüm harfler
    letters = lowercase_letters + uppercase_letters + arabic_letters + cyrillic_letters
    
    # Rakamlar
    digits = list("0123456789")
    
    # Noktalama işaretleri (temel)
    basic_punct = [".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", '"', "'"]
    
    # Ek özel karakterler (URL, kod, matematik vb. için)
    # Endüstri standardı: Tüm yaygın karakterler base alphabet'te olmalı
    special_chars = [
        "/",  # URL, path separator
        "\\", # Windows path separator
        "&",  # HTML entities, URL params
        "@",  # Email, mentions
        "#",  # Hashtag, hex
        "*",  # Wildcard, markdown
        "+",  # Plus, URL encoding
        "=",  # Equality, URL params
        "<",  # Comparison, HTML
        ">",  # Comparison, HTML
        "|",  # Pipe, logical OR
        "%",  # Percent, URL encoding
        "$",  # Currency, variables
        "^",  # XOR, caret
        "~",  # Tilde, approximation
    ]
    
    # ============================================================================
    # MATEMATİK/FİZİK/KİMYA SEMBOLLERİ (Endüstri Standardı - GPT-4, Claude gibi)
    # ============================================================================
    
    # Yunan Harfleri (Küçük)
    greek_lowercase = [
        "α",  # alpha
        "β",  # beta
        "γ",  # gamma
        "δ",  # delta
        "ε",  # epsilon
        "ζ",  # zeta
        "η",  # eta
        "θ",  # theta
        "ι",  # iota
        "κ",  # kappa
        "λ",  # lambda
        "μ",  # mu
        "ν",  # nu
        "ξ",  # xi
        "ο",  # omicron
        "π",  # pi
        "ρ",  # rho
        "σ",  # sigma
        "τ",  # tau
        "υ",  # upsilon
        "φ",  # phi
        "χ",  # chi
        "ψ",  # psi
        "ω",  # omega
    ]
    
    # Yunan Harfleri (Büyük)
    greek_uppercase = [
        "Α",  # Alpha
        "Β",  # Beta
        "Γ",  # Gamma
        "Δ",  # Delta
        "Ε",  # Epsilon
        "Ζ",  # Zeta
        "Η",  # Eta
        "Θ",  # Theta
        "Ι",  # Iota
        "Κ",  # Kappa
        "Λ",  # Lambda
        "Μ",  # Mu
        "Ν",  # Nu
        "Ξ",  # Xi
        "Ο",  # Omicron
        "Π",  # Pi
        "Ρ",  # Rho
        "Σ",  # Sigma
        "Τ",  # Tau
        "Υ",  # Upsilon
        "Φ",  # Phi
        "Χ",  # Chi
        "Ψ",  # Psi
        "Ω",  # Omega
    ]
    
    # Matematik Sembolleri
    math_symbols = [
        "∞",  # sonsuz (infinity)
        "±",  # artı/eksi (plus-minus)
        "×",  # çarpı (multiplication)
        "÷",  # bölü (division)
        "∑",  # toplam (summation)
        "∏",  # çarpım (product)
        "∫",  # integral
        "∮",  # kapalı integral (contour integral)
        "√",  # karekök (square root)
        "∛",  # küp kök (cube root)
        "∜",  # dördüncü kök (fourth root)
        "≈",  # yaklaşık (approximately equal)
        "≠",  # eşit değil (not equal)
        "≤",  # küçük eşit (less than or equal)
        "≥",  # büyük eşit (greater than or equal)
        "≪",  # çok küçük (much less than)
        "≫",  # çok büyük (much greater than)
        "≡",  # denk (equivalent)
        "∝",  # orantılı (proportional to)
        "∈",  # eleman (element of)
        "∉",  # eleman değil (not element of)
        "⊂",  # alt küme (subset)
        "⊃",  # üst küme (superset)
        "⊆",  # alt küme veya eşit (subset or equal)
        "⊇",  # üst küme veya eşit (superset or equal)
        "∪",  # birleşim (union)
        "∩",  # kesişim (intersection)
        "∖",  # fark (set difference)
        "∇",  # gradyan (nabla/del)
        "∂",  # kısmi türev (partial derivative)
        "∆",  # değişim (increment/Laplace operator)
        "∃",  # var (there exists)
        "∀",  # her (for all)
        "∅",  # boş küme (empty set)
        "ℕ",  # doğal sayılar (natural numbers)
        "ℤ",  # tam sayılar (integers)
        "ℚ",  # rasyonel sayılar (rational numbers)
        "ℝ",  # gerçek sayılar (real numbers)
        "ℂ",  # karmaşık sayılar (complex numbers)
        "ℵ",  # aleph (cardinal number)
    ]
    
    # Fizik Sembolleri
    physics_symbols = [
        "ℏ",  # Planck sabiti (reduced Planck constant)
        "°",  # derece (degree)
        "′",  # dakika (prime/minute)
        "″",  # saniye (double prime/second)
        "‴",  # üçüncü (triple prime)
        "℧",  # mho (conductance unit)
        "Ω",  # ohm (resistance unit - alternatif)
        "℃",  # Celsius
        "℉",  # Fahrenheit
        "K",  # Kelvin
    ]
    
    # Kimya Sembolleri
    chemistry_symbols = [
        "→",  # reaksiyon (reaction arrow)
        "←",  # geri reaksiyon (reverse reaction)
        "↔",  # denge (equilibrium)
        "⇌",  # denge (reversible reaction)
        "⇄",  # denge (reversible reaction - alternatif)
        "↑",  # gaz (gas/gas evolution)
        "↓",  # çökelti (precipitate)
        "⇈",  # çift ok yukarı (double arrow up)
        "⇊",  # çift ok aşağı (double arrow down)
        "⇅",  # çift ok yukarı-aşağı (double arrow up-down)
    ]
    
    # Alt/Üst İndis Sembolleri
    subscript_superscript = [
        # Üst indis (superscript)
        "⁰",  # 0
        "¹",  # 1
        "²",  # 2
        "³",  # 3
        "⁴",  # 4
        "⁵",  # 5
        "⁶",  # 6
        "⁷",  # 7
        "⁸",  # 8
        "⁹",  # 9
        "⁺",  # +
        "⁻",  # -
        "⁼",  # =
        "⁽",  # (
        "⁾",  # )
        "ⁿ",  # n
        "ⁱ",  # i
        "ᵃ",  # a
        "ᵇ",  # b
        "ᵈ",  # d
        "ᵉ",  # e
        "ᵍ",  # g
        "ʰ",  # h
        "ⁱ",  # i
        "ʲ",  # j
        "ᵏ",  # k
        "ˡ",  # l
        "ᵐ",  # m
        "ⁿ",  # n
        "ᵒ",  # o
        "ᵖ",  # p
        "ʳ",  # r
        "ˢ",  # s
        "ᵗ",  # t
        "ᵘ",  # u
        "ᵛ",  # v
        "ʷ",  # w
        "ˣ",  # x
        "ʸ",  # y
        "ᶻ",  # z
        # Alt indis (subscript)
        "₀",  # 0
        "₁",  # 1
        "₂",  # 2
        "₃",  # 3
        "₄",  # 4
        "₅",  # 5
        "₆",  # 6
        "₇",  # 7
        "₈",  # 8
        "₉",  # 9
        "₊",  # +
        "₋",  # -
        "₌",  # =
        "₍",  # (
        "₎",  # )
        "ₐ",  # a
        "ₑ",  # e
        "ₕ",  # h
        "ᵢ",  # i
        "ⱼ",  # j
        "ₖ",  # k
        "ₗ",  # l
        "ₘ",  # m
        "ₙ",  # n
        "ₒ",  # o
        "ₚ",  # p
        "ᵣ",  # r
        "ₛ",  # s
        "ₜ",  # t
        "ᵤ",  # u
        "ᵥ",  # v
        "ₓ",  # x
    ]
    
    # Tüm matematik/fizik/kimya sembolleri
    scientific_symbols = (
        greek_lowercase + 
        greek_uppercase + 
        math_symbols + 
        physics_symbols + 
        chemistry_symbols + 
        subscript_superscript
    )
    
    # Tüm noktalama ve özel karakterler
    punct = basic_punct + special_chars + scientific_symbols
    
    # Boşluk
    space = [" "]
    
    # Word boundary token (ayrı bir token olarak)
    word_boundary = ["</w>"]
    
    # ÖNEMLİ: Sadece tek karakterler eklenir, char + </w> formatı YOK!
    return letters + digits + punct + space + word_boundary

# ---- Varsayılan Vocab & Token Temizliği --------------------------------------

def default_vocab(specials: Optional[Dict[str, int]] = None, include_base_alphabet: bool = True) -> Dict[str, dict]:
    """
    Varsayılan vocab'ı üretir (endüstri standardı - GPT, BERT gibi).
    
    ENDÜSTRİ STANDARDI:
    1. Special tokenlar (sabit ID'ler: 0-4)
    2. Base alphabet (otomatik eklenir: 5+)
    
    Args:
        specials: Özel token haritası (None ise DEFAULT_SPECIALS kullanılır)
        include_base_alphabet: Base alphabet'i otomatik ekle (True: endüstri standardı)
    
    Returns:
        Dict[str, dict]: Normalize edilmiş vocab (special tokenlar + base alphabet)
    
    Example:
        >>> vocab = default_vocab()
        >>> vocab["<BOS>"]["id"]  # 2
        >>> vocab["a"]["id"]      # 5 (otomatik eklenmiş)
        >>> vocab["</w>"]["id"]   # 53 (otomatik eklenmiş)
    """
    sp = dict(specials) if specials is not None else dict(DEFAULT_SPECIALS)

    # Doğrula (benzersizlik vs.)
    if len(set(sp.values())) != len(sp):
        raise ValueError("Special token ID'leri benzersiz olmalıdır.")
    if any((not isinstance(k, str) or (not k.strip() and k != " ") or not isinstance(v, int) or v < 0) for k, v in sp.items()):
        raise ValueError("Special token haritası geçersiz değer içeriyor.")

    # 1. Special tokenları ekle
    vocab: Dict[str, dict] = {}
    for tok, tid in sp.items():
        vocab[tok] = {"id": int(tid), "total_freq": 0, "positions": []}

    # 2. Base alphabet'i otomatik ekle (endüstri standardı)
    if include_base_alphabet:
        base_alphabet = get_base_alphabet()
        # Sonraki ID'yi hesapla (special tokenlardan sonra)
        next_char_id = max(meta["id"] for meta in vocab.values()) + 1
        
        for char in base_alphabet:
            # Special tokenlar ile çakışmasın
            if char not in vocab:
                vocab[char] = {"id": next_char_id, "total_freq": 0, "positions": []}
                next_char_id += 1
        
        base_start_id = max(sp.values()) + 1
        base_end_id = next_char_id - 1
        logger.info(f"[default_vocab] Base alphabet otomatik eklendi: {len(base_alphabet)} karakter (ID: {base_start_id} - {base_end_id})")

    validate_vocab(vocab)
    return vocab

def clean_tokens(tokens: List[str]) -> List[str]:
    """
    Token listesini güvenle temizler:
    - None/boş girişlerde [] döner (exception fırlatmaz)
    - Sadece str olanları alır, strip eder, boşları atar
    - Sıra korunarak yinelenenleri kaldırır
    """
    if not isinstance(tokens, list) or len(tokens) == 0:
        return []
    cleaned = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
    # Sırayı koruyarak tekilleştir
    return list(dict.fromkeys(cleaned))

# ---- (İsteğe bağlı) Vocab’i versiyonlu JSON’a dönüştürme ---------------------

def to_versioned_vocab(
    vocab: Dict[str, dict],
    *,
    version: int = 1,
    specials_map: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Normalleştirilmiş vocab’ı yeni şemaya dönüştürür:
    {
      "version": 1,
      "specials": {"<BOS>":2, ...},
      "tokens": {"merhaba</w>":{"id":10,"total_freq":12,"positions":[]}, ...}
    }
    """
    validate_vocab(vocab)

    # specials_map verilmişse kullan; yoksa DEFAULT_SPECIALS içerisinden vocab’ta mevcut olanları tahmin et
    if specials_map is None:
        # Heuristik: DEFAULT_SPECIALS'ta tanımlı olanlar ve ID’si vocab’ta eşleşenler
        reverse: Dict[int, str] = {}
        for t, meta in vocab.items():
            reverse[meta["id"]] = t
        detected: Dict[str, int] = {}
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            if sp_tok in vocab and vocab[sp_tok]["id"] == sp_id:
                detected[sp_tok] = sp_id
        specials_map = detected

    tokens_obj: Dict[str, Any] = {}
    for tok, meta in vocab.items():
        tokens_obj[tok] = {
            "id": int(meta["id"]),
            "total_freq": int(meta.get("total_freq", 0)),
            "positions": list(meta.get("positions", [])),
        }

    return {
        "version": int(version),
        "specials": dict(specials_map),
        "tokens": tokens_obj,
    }
