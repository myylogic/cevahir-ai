# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: constitutional_principles.py
Modül: cognitive_management/v2/config
Görev: Constitutional AI Principles - Constitutional principles for AI safety and
       alignment. Phase 7.2: Advanced Critic System Enhancement. Default
       constitutional principles, principle management ve principle retrieval
       işlemlerini yapar. Akademik referans: Anthropic (2023).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (constitutional principles)
- Design Patterns: Principles Pattern (constitutional principles)
- Endüstri Standartları: Constitutional AI best practices

KULLANIM:
- Constitutional principles için
- AI safety için
- AI alignment için

BAĞIMLILIKLAR:
- Modül içi bağımlılıklar

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import re
from typing import Iterable, List, Optional, Tuple


# Default Constitutional Principles
# Based on Anthropic's Constitutional AI and best practices
#
# Each principle appears once, with its English and Turkish wording. Evaluating
# both wordings of the same principle doubled the model calls without adding a
# new check, so the critic now selects the wording(s) a response actually needs.

SUPPORTED_PRINCIPLE_LANGUAGES: Tuple[str, ...] = ("en", "tr")

CONSTITUTIONAL_PRINCIPLES: Tuple[Tuple[str, str], ...] = (
    # (English, Türkçe)
    # Core Principles
    ("Be helpful, harmless, and honest.",
     "Yardımcı, zararsız ve dürüst ol."),
    
    # Safety Principles
    ("Avoid harmful, illegal, or unethical content.",
     "Zararlı, yasadışı veya etik dışı içerikten kaçın."),
    ("Do not provide information that could cause harm.",
     "Zarara neden olabilecek bilgi verme."),
    
    # Privacy and Autonomy
    ("Respect user privacy and autonomy.",
     "Kullanıcı gizliliğini ve özerkliğini saygı göster."),
    ("Do not invade privacy or share personal information.",
     "Gizliliği ihlal etme veya kişisel bilgi paylaşma."),
    
    # Accuracy and Honesty
    ("Provide accurate information and acknowledge uncertainty.",
     "Doğru bilgi ver ve belirsizliği kabul et."),
    ("Clearly distinguish facts from opinions.",
     "Gerçekleri görüşlerden açıkça ayır."),
    ("Admit when you don't know something.",
     "Bir şeyi bilmediğinde kabul et."),
    
    # Helpfulness
    ("Be as helpful as possible while staying safe and honest.",
     "Güvenli ve dürüst kalırken mümkün olduğunca yardımcı ol."),
    ("Provide clear and concise responses.",
     "Net ve özlü cevaplar ver."),
    
    # Fairness
    ("Treat all users fairly and without bias.",
     "Tüm kullanıcılara adil ve önyargısız davran."),
    ("Avoid discriminatory or prejudiced content.",
     "Ayrımcı veya önyargılı içerikten kaçın."),
    
    # Ethical Guidelines
    ("Follow ethical guidelines in all responses.",
     "Tüm cevaplarda etik kurallara uy."),
    ("Consider the potential consequences of your responses.",
     "Cevaplarının potansiyel sonuçlarını düşün."),
)

# Historical flat list (both wordings, 28 entries, original order); kept for
# callers that import it. The critic no longer evaluates this list as-is.
DEFAULT_CONSTITUTIONAL_PRINCIPLES: List[str] = [
    text for pair in CONSTITUTIONAL_PRINCIPLES for text in pair
]

_TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")
_TURKISH_WORDS = frozenset((
    "ve", "bir", "bu", "şu", "için", "ile", "da", "de", "ki", "mi", "çok",
    "ama", "gibi", "değil", "olarak", "daha", "nasıl", "neden", "evet", "hayır",
))
_ENGLISH_WORDS = frozenset((
    "the", "and", "is", "are", "was", "of", "to", "in", "it", "that", "this",
    "with", "for", "you", "not", "what", "how", "why", "yes", "have",
))


def detect_principle_language(text: str) -> Optional[str]:
    """
    Return "tr" or "en" for the dominant language of `text`, or None.

    A cheap lexical check (Turkish-specific letters and common function words),
    not a language-identification model; ties go to Turkish, the system's
    primary language.
    """
    if not text:
        return None
    words = re.findall(r"[^\W\d_]+", text.lower())
    turkish = sum(word in _TURKISH_WORDS for word in words)
    if any(char in _TURKISH_LETTERS for char in text):
        turkish += 2
    english = sum(word in _ENGLISH_WORDS for word in words)
    if turkish == english == 0:
        return None
    return "tr" if turkish >= english else "en"


def get_principles(
    custom_principles: Tuple[str, ...] = (),
    languages: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Get constitutional principles.
    
    Args:
        custom_principles: Custom principles to add (each kept once, as given)
        languages: Wordings to include, e.g. ("tr",). None keeps the historical
            list with both wordings of every principle.
        
    Returns:
        Each default principle once per requested language, then the custom
        principles.
    """
    if languages is None:
        principles = list(DEFAULT_CONSTITUTIONAL_PRINCIPLES)
    else:
        wanted = []
        for language in languages:
            if language not in SUPPORTED_PRINCIPLE_LANGUAGES:
                raise ValueError(f"Unsupported principle language: {language!r}")
            if language not in wanted:
                wanted.append(language)
        indices = [SUPPORTED_PRINCIPLE_LANGUAGES.index(language) for language in wanted]
        principles = [pair[i] for pair in CONSTITUTIONAL_PRINCIPLES for i in indices]
    
    for principle in custom_principles:
        if principle not in principles:
            principles.append(principle)
    
    return principles


__all__ = [
    "CONSTITUTIONAL_PRINCIPLES",
    "DEFAULT_CONSTITUTIONAL_PRINCIPLES",
    "SUPPORTED_PRINCIPLE_LANGUAGES",
    "detect_principle_language",
    "get_principles",
]

