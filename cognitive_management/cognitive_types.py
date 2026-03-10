# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cognitive_types.py
Modül: cognitive_management
Görev: Cognitive Types - Bilişsel yönetim katmanı için tip tanımları. Mode, ToolDecision,
       DecodingConfig, PolicyOutput, CognitiveState, CognitiveInput, CognitiveOutput,
       ThoughtCandidate ve diğer tip tanımlarını içerir. Dataclass tabanlı, tip güvenli
       yapı sağlar.

MİMARİ:
- SOLID Prensipleri: Type definitions (tip tanımları)
- Design Patterns: Type Pattern (tip tanımları)
- Endüstri Standartları: Type safety best practices

KULLANIM:
- Tip tanımları için
- Dataclass tanımları için
- Type hints için

BAĞIMLILIKLAR:
- dataclasses: Dataclass tanımları
- typing: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

# === Mod kipleri ve kararlar ===
Mode = Literal["direct", "think1", "debate2", "tot"]  # Phase 4: Added "tot" for Tree of Thoughts
ToolDecision = Literal["none", "maybe", "must"]

# === Decoding/Üretim ayarları ===
@dataclass
class DecodingConfig:
    """
    Metin üretim düğmeleri. ModelManager.generate(...) / backend tarafından tüketilir.
    """
    max_new_tokens: int = 256
    min_new_tokens: Optional[int] = None  # [OK] YENİ: Minimum token sayısı (EOS erken gelmesin)
    temperature: float = 0.7
    top_p: float = 0.9
    # testler top_k alanını geçiriyor → opsiyonel olarak ekliyoruz
    top_k: Optional[int] = 0
    # backend'de kullanıyoruz; burada da dursun
    repetition_penalty: float = 1.1

# === Politika çıkışı ===
@dataclass
class PolicyOutput:
    """
    PolicyRouter kararının bütünsel çıktısı.
    """
    mode: Mode
    tool: ToolDecision
    decoding: DecodingConfig
    inner_steps: int = 0  # think/debate için iç düşünce adım sayısı

# === İç ses adayları ===
@dataclass
class ThoughtCandidate:
    text: str
    score: float

# === Bilişsel durum ===
@dataclass
class CognitiveState:
    """
    Chat/oturum sırasında kümülatif durum.
    - history: {"role": "user"|"assistant"|"system_summary", "content": "..."} listesi
    - step: kaçıncı bilişsel tur
    - last_entropy: bir önceki belirsizlik kestirimi (opsiyonel)
    - last_mode: son karar verilen kip (opsiyonel)
    """
    history: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    last_entropy: Optional[float] = None
    last_mode: Optional[Mode] = None

# === İstek/veri girişi ===
@dataclass
class CognitiveInput:
    """
    Kullanıcıdan/üst katmandan gelen ham istek.
    - user_message: zorunlu
    - system_prompt: sistem davranışını sabitleyen üst yönerge (opsiyonel)
    - metadata: sinyaller, tip etiketleri, risk bayrakları (opsiyonel)
    """
    user_message: str
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# === Nihai çıktı ===
@dataclass
class CognitiveOutput:
    """
    CognitiveManager.handle(...) dönüşü.
    """
    text: str
    used_mode: Mode
    tool_used: Optional[str] = None
    revised_by_critic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)  # Phase 5.3: Tracing metadata

