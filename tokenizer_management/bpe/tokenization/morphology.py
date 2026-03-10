# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: morphology.py
Modül: tokenizer_management/bpe/tokenization
Görev: Morphology sınıfı - Türkçe morfoloji analizi. Kelimeleri kök ve ek
       olarak ayırır, ünlü uyumu kurallarını uygular, Türkçe dilbilgisi
       kurallarına göre morfem analizi yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (morfoloji analizi)
- Design Patterns: Rule-based Pattern (Türkçe morfoloji kuralları)
- Endüstri Standartları: Türkçe dilbilgisi kuralları, morfem analizi

KULLANIM:
- BPEManager tokenization pipeline'ında kullanılır (opsiyonel)
- Kelime → kök + ek ayrıştırması için
- Türkçe morfoloji analizi için

BAĞIMLILIKLAR:
- get_turkish_config: Türkçe karakter yapılandırması
- _morphology_utils: Morfoloji yardımcı fonksiyonları
- MorphologyError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import List, Dict, Set, Union, Optional

from tokenizer_management.config import get_turkish_config
from tokenizer_management.bpe.tokenization._morphology_utils import (
    split_into_syllables_impl,
    split_morpheme_impl,
    check_vowel_harmony_impl,
    _normalize_root,
)

logger = logging.getLogger(__name__)

class MorphologyError(Exception):
    """Türkçe kök-ek ayrımı sırasında oluşan özel istisna."""
    pass

class Morphology:
    """
    Türkçe kelimeler üzerinde rule-based, tamamen algoritmik kök çıkarımı.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        # Config merge (turkish + BPE detailed)
        from tokenizer_management.config import BPE_DETAILED_CONFIG
        cfg = get_turkish_config()
        if config:
            cfg.update(config)
        cfg.update(BPE_DETAILED_CONFIG)  # BPE parametreleri için
        self.cfg = cfg
        self.config = cfg

        self.vowels: Set[str] = set(cfg.get("vowels", []))
        self.consonants: Set[str] = set(cfg.get("consonants", []))
        self.valid_initial_consonant_clusters: Set[str] = set(
            cfg.get("valid_initial_consonant_clusters", [])
        )

        # Suffix listesi (en uzun ek en önce)
        raw_suffixes = cfg.get("suffixes", {})
        if isinstance(raw_suffixes, list):
            raw_suffixes = {"default": raw_suffixes}
        if not isinstance(raw_suffixes, dict):
            raise MorphologyError(f"suffixes config must be dict or list, got: {type(raw_suffixes).__name__}")
        flat = {
            s.casefold()
            for lst in raw_suffixes.values()
            for s in (lst if isinstance(lst, (list, tuple)) else [])
            if isinstance(s, str) and len(s) > 0
        }
        self.suffixes: List[str] = sorted(flat, key=len, reverse=True)
        self.root_overrides: Dict[str, str] = cfg.get("root_overrides", {})

        default_map = {"p": "b", "ç": "c", "t": "d", "k": ["ğ", "g"]}
        cs_map: Dict[str, Union[str, List[str]]] = cfg.get("consonant_softening", default_map)
        self.soft_to_hard: Dict[str, str] = {}
        for hard, soft in cs_map.items():
            if isinstance(soft, list):
                for s in soft:
                    self.soft_to_hard[s] = hard
            else:
                self.soft_to_hard[soft] = hard

        self.vowel_harmony_rules: Dict[str, List[str]] = cfg.get(
            "vowel_harmony_rules", {"front_vowels": ["e", "i", "ö", "ü"], "back_vowels": ["a", "ı", "o", "u"]}
        )
        self.vowel_harmony_exceptions: Set[str] = set(cfg.get("vowel_harmony_exceptions", []))

        # Kök bulma için ekler (config'ten)
        negative_suffixes_list = cfg.get("negative_suffixes", ["ma", "me"])
        question_suffixes_list = cfg.get("question_particles", ["mi", "mı", "mu", "mü"])
        personal_suffixes_list = cfg.get("personal_suffixes", ["im", "ım", "um", "üm", "sin", "sın", "sun", "sün", "miz", "nız", "niz", "muz", "müz", "lar", "ler", "yim", "yım", "yum", "yüm"])
        
        # List veya set'ten set'e çevir
        self.negative_suffixes = set(negative_suffixes_list) if not isinstance(negative_suffixes_list, set) else negative_suffixes_list
        self.question_suffixes = set(question_suffixes_list) if not isinstance(question_suffixes_list, set) else question_suffixes_list
        self.personal_suffixes = set(personal_suffixes_list) if not isinstance(personal_suffixes_list, set) else personal_suffixes_list

        logger.info(
            "[+] Morphology başlatıldı: %d ünlü, %d ünsüz, %d suffix, %d onset clusters",
            len(self.vowels), len(self.consonants), len(self.suffixes), len(self.valid_initial_consonant_clusters)
        )

    def split_into_syllables(self, word: str) -> List[str]:
        """Harf kümesi ve kurallara göre heceleme yapar."""
        return split_into_syllables_impl(
            word,
            self.vowels,
            self.valid_initial_consonant_clusters
        )


    def _split_morpheme(self, token: str) -> List[str]:
        """
        Token'i kök + ek zinciri olarak ayırır.
        Kök üzerinde TDK uyumlu normalizasyon uygular.
        Reflexive fiilleri ve root_overrides'u kontrol eder.

        Args:
            token (str): Input token to split.

        Returns:
            List[str]: List containing the root and suffixes.

        Raises:
            TypeError: If token is None, not a string, or empty.
        """
        if token is None or not isinstance(token, str) or token.strip() == "":
            logger.debug(f"[!] Geçersiz token: {token!r}, TypeError fırlatılıyor")
            raise TypeError("Geçersiz kelime.")

        t = token.lower()

        # 1) TDK exceptions check
        tdk_exceptions = self.cfg.get("tdk_exceptions", {})
        if t in tdk_exceptions:
            return tdk_exceptions[t]

        # 2) Root override check
        override = self.root_overrides.get(t)
        if override:
            return [override] if isinstance(override, str) else override

        # 3) Reflexive verbs check
        reflexive_verbs = self.cfg.get("reflexive_verbs", [])
        for rv in reflexive_verbs:
            if t.startswith(rv):
                suffix_part = t[len(rv):]
                if suffix_part == "" or any(suffix_part.endswith(suf) for suf in self.suffixes):
                    suffixes = []
                    remainder = suffix_part
                    sorted_sufs = sorted(self.suffixes, key=len, reverse=True)
                    while remainder and len(remainder) >= 2:
                        matched = False
                        for suf in sorted_sufs:
                            if remainder.endswith(suf):
                                suffixes.insert(0, suf)
                                remainder = remainder[:-len(suf)]
                                matched = True
                                break
                        if not matched:
                            break
                    normalized_root = _normalize_root(rv, token, suffixes, self.suffixes)
                    return [normalized_root] + suffixes

        # 4) Normal morpheme splitting
        try:
            parts = split_morpheme_impl(
                token,
                self.suffixes,
                self.cfg.get("consonant_softening", {}),
                reflexive_verbs,
                self.cfg.get("question_particles", ["mi", "mı", "mu", "mü"])
            )
            if not parts:
                return [token]
            normalized_root = _normalize_root(parts[0], token, parts[1:], self.suffixes)
            return [normalized_root] + parts[1:]
        except Exception as e:
            logger.debug(f"[!] Morfem ayrımı hatası: {token}, hata: {str(e)}")
            return [token]


    def _strip_suffix_chain(self, t: str) -> str:
        """
        Ek zincirini sökerek minimum uzunluk ve ünlü sayısı koşulunu sağlayan kökü bulur.
        """
        original = t
        current = t
        for suf in self.suffixes:
            if current.endswith(suf) and len(current) - len(suf) >= 2:
                current = current[:-len(suf)]
                # Yumuşama düzeltmesi
                for soft, hard in self.soft_to_hard.items():
                    if current.endswith(soft):
                        alt = current[:-1] + hard
                        if len(alt) >= 2 and alt.lower() in original.lower():
                            current = alt
                        break
        # Ünlü ve uzunluk kontrolü
        if len(current) < 2 or sum(1 for c in current if c in self.vowels) == 0:
            return original
        return current

    def find_root(self, token: str) -> str:
        """
        Tamamen algoritmik kök çıkarımı:
        - TDK istisnaları ile özel kelime kökü kontrolü
        - root_overrides ile istisna kelime kökü
        - Bileşik fiil veya parçalı yapı ayrımı (ör. "yardım etti" → "yardım et")
        - Suffix zinciri çözümü
        - Yumuşama, olumsuzluk, zaman ve soru ekleri ayrımı
        - Kısaltma (örneğin, TBMM'den → TBMM)

        Args:
            token (str): Kökü çıkarılacak kelime.

        Returns:
            str: Normalleştirilmiş kök.

        Raises:
            MorphologyError: Eğer token None, string değil veya boşsa.
        """
        if not isinstance(token, str) or not token.strip():
            raise MorphologyError(f"Geçersiz token: {token!r}")

        t = token.lower().strip()
        min_len = self.cfg.get("text_processing", {}).get("min_word_length", 4)

        # Config'den gerekli yapılandırmaları al
        root_overrides = self.root_overrides
        negative_suffixes = self.negative_suffixes
        question_suffixes = self.cfg.get("question_particles", ["mi", "mı", "mu", "mü"])
        personal_suffixes = self.personal_suffixes
        time_suffixes_list = self.cfg.get("time_suffixes", ["dı", "di", "du", "dü", "ti", "tu", "tü"])
        time_suffixes = set(time_suffixes_list) if not isinstance(time_suffixes_list, set) else time_suffixes_list
        zamirler = self.cfg.get("zamirler", ["ben", "sen", "biz", "siz", "onlar"])
        compound_verbs = self.cfg.get("compound_verbs", ["et", "etmek", "edi", "ett"])  # Bileşik fiiller için

        # --- 1) TDK istisnaları kontrolü ---
        tdk_exceptions = self.cfg.get("tdk_exceptions", {})
        if t in tdk_exceptions:
            parts = tdk_exceptions[t]
            if len(parts) > 1 and parts[1] in compound_verbs:
                return f"{parts[0]} {parts[1]}"  # Bileşik fiil kökü, örn: "yardım et"
            return parts[0]  # Normal kök

        # --- 2) Root override kontrolü ---
        override = root_overrides.get(t)
        if override is not None:
            return override

        # --- 3) Kısaltma/abbreviation ---
        if "'" in t:
            return t.split("'")[0]

        # --- 4) Bileşik fiil/parçalı yapı ---
        if " " in t and len(t.split()) == 2:
            toks = t.split()
            root1 = self.find_root(toks[0])
            root2 = self.find_root(toks[1])
            if root2 in compound_verbs:  # "etmek" fiili için özel düzeltme
                root2 = "et"
            return f"{root1} {root2}".strip()

        # --- 5) Zamir eki kontrolü ---
        for suf in zamirler:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                base = t[:-len(suf)]
                if base in zamirler:
                    return base

        # --- 6) Soru eki ve fiil çekimi ---
        for suf in question_suffixes:
            if t.endswith(f" {suf}") or t.endswith(suf):
                t2 = t.replace(f" {suf}", "").replace(suf, "").strip()
                parts = self._split_morpheme(t2)
                if len(parts) > 1:
                    candidate = parts[0]
                    if candidate.endswith(("t", "d")) and any(
                        suf in personal_suffixes | time_suffixes for suf in parts[1:]
                    ):
                        candidate = candidate[:-1]
                    if len(candidate) >= 2 and check_vowel_harmony_impl(
                        candidate, self.vowels, self.vowel_harmony_rules, self.vowel_harmony_exceptions
                    ):
                        return candidate
                return self.find_root(t2)

        # --- 7) Olumsuzluk eki kontrolü ---
        for suf in negative_suffixes:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                base = t[:-len(suf)]
                if base.endswith(("madı", "medi")):
                    return self.find_root(base[:-3])
                return self.find_root(base)

        # --- 8) Morfem zinciri ile ayrıştırma ---
        parts = self._split_morpheme(t)
        if len(parts) > 1:
            candidate = parts[0]
            if len(candidate) >= 2 and check_vowel_harmony_impl(
                candidate, self.vowels, self.vowel_harmony_rules, self.vowel_harmony_exceptions
            ):
                if candidate.endswith(("t", "d")) and any(
                    suf in personal_suffixes | time_suffixes for suf in parts[1:]
                ):
                    candidate = candidate[:-1]
                return candidate

        # --- 9) Fallback: suffix zincirini sıyırma ---
        root = self._strip_suffix_chain(t)
        if len(root) >= 2 and check_vowel_harmony_impl(
            root, self.vowels, self.vowel_harmony_rules, self.vowel_harmony_exceptions
        ):
            if root.endswith(("t", "d")) and any(
                t.endswith(suf) for suf in personal_suffixes | time_suffixes
            ):
                root = root[:-1]
            return root

        # --- 10) Uzun kelime veya istisna ---
        if len(t) > min_len:
            return t

        return root
    
    def _check_vowel_harmony(self, root: str) -> bool:
        """Ünlü uyumunu kontrol eder."""
        return check_vowel_harmony_impl(
            root,
            self.vowels,
            self.vowel_harmony_rules,
            self.vowel_harmony_exceptions
        )

    def analyze(self, tokens: List[str]) -> List[str]:
        """
        Token listesi üzerinden kök çıkarımı (tamamen algoritmik).
        Zincir, kısa token birleştirme, ek/olumsuzluk/soru ayrımı, bileşik fiil kontrolü yapılır.
        """
        if tokens is None:
            raise MorphologyError("Token listesi None olamaz.")
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise MorphologyError(f"Token listesi geçersiz tipte: {type(tokens).__name__}")
        if not tokens:
            raise MorphologyError("Token listesi boş.")

        results: List[str] = []
        i = 0
        N = len(tokens)
        while i < N:
            normalized = tokens[i].lower().strip()
            if not normalized:
                results.append(tokens[i])
                i += 1
                continue

            # Uzun kelime/bileşik fiil/kısaltma/özel durumlarda tek başına işle
            if (
                len(normalized) > 4
                or "'" in normalized
                or " " in normalized
            ):
                root = self.find_root(normalized)
                results.append(root)
                i += 1
                continue

            # Kısa token birleştir
            found = False
            for j in range(N, i+1, -1):
                if j - i < 2:
                    continue
                combined = "".join(tokens[i:j]).lower()
                root = self.find_root(combined)
                if (
                    root != combined
                    and len(root) >= 2
                    and sum(1 for c in root if c in self.vowels) >= 1
                ):
                    logger.debug(f"[+] Kombine kök: {tokens[i:j]} -> {combined} -> {root}")
                    results.append(root)
                    i = j
                    found = True
                    break
            if found:
                continue

            # Tek token kök bul
            root = self.find_root(normalized)
            results.append(root)
            i += 1

        logger.debug(f"[+] Analiz sonucu: {tokens} -> {results}")
        return results

    def reset(self) -> None:
        """Config’u yeniden yükler."""
        self.__init__()