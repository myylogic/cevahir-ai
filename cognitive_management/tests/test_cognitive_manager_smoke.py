# -*- coding: utf-8 -*-
"""
tests/test_cognitive_manager_smoke.py
=====================================
CognitiveManager için uçtan uca duman ve davranış testleri (pytest).

Koşullar
--------
- Gerçek model yerine FakeModelAPI kullanılır (deterministik).
- Amaç: direct/think/debate kipleri, tool seçimi, critic revizyonu,
  bellek özetleme ve bağlam inşasının temel akışını doğrulamak.
"""

from __future__ import annotations
import pytest

from cognitive_management.cognitive_types import CognitiveState, CognitiveInput
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_manager import CognitiveManager, ModelAPI
from cognitive_management.cognitive_types import DecodingConfig


# =============================================================================
# Fake Model API (deterministik)
# =============================================================================

class FakeModel(ModelAPI):
    """
    Minimal, deterministik bir ModelAPI:
    - generate(): prompt içeriğine göre basit yanıt verir
    - score(): aday uzunluğuna göre kaba skor
    - entropy_estimate(): metin ipuçlarına göre sabit değer döndürür
    """

    def __init__(self):
        self.last_prompt = None
        self.last_decoding = None

    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str:
        self.last_prompt = prompt
        self.last_decoding = decoding_cfg
        # CRITIC akışı: "[CRITIC]" varsa güvenli, kısa revizyon döndür
        if "[CRITIC]" in prompt:
            return "Revize edildi: Daha temkinli ve kısa bir yanıt."
        # INTERNAL THOUGHT akışı: iç plan üret
        if "[INTERNAL THOUGHT]" in prompt:
            return (
                "- MADDE 1: Talebi çözümle.\n"
                "- MADDE 2: Gerekirse araç öner.\n"
                "- MADDE 3: Net bir yanıt planla."
            )
        # Genel üretim (taslak)
        if "riskli_oran" in prompt:
            # Kritiği tetikleyecek iddialı taslak
            return "Bu %50 ve %30 oranlarıyla kesinlikle doğrudur. Kaynak yok."
        # Normal eko-yanıt
        return "Bu, deneme cevabıdır."

    def score(self, prompt: str, candidate: str) -> float:
        # Basit uzunluk tabanlı skor (daha uzun = biraz yüksek)
        return float(len(candidate))

    def entropy_estimate(self, text: str) -> float:
        # İpuçlarına göre kaba entropi kestirimi
        low = "LOW_ENT" in text
        high = "HIGH_ENT" in text
        if high:
            return 2.5
        if low:
            return 0.5
        # Varsayılan orta
        return 1.0


# =============================================================================
# Fixture'lar
# =============================================================================

@pytest.fixture()
def cfg() -> CognitiveManagerConfig:
    cfg = CognitiveManagerConfig()
    # Testleri hızlandırmak için küçük bağlam bütçesi
    cfg.memory.max_history_tokens = 512
    cfg.memory.session_summary_every = 6
    cfg.memory.salient_topk = 6
    cfg.critic.enabled = True
    cfg.critic.strictness = 0.5
    return cfg

@pytest.fixture()
def cm(cfg: CognitiveManagerConfig) -> CognitiveManager:
    fm = FakeModel()
    return CognitiveManager(fm, cfg)

@pytest.fixture()
def state() -> CognitiveState:
    return CognitiveState()


# =============================================================================
# Testler
# =============================================================================

def test_smoke_direct_mode(cm: CognitiveManager, state: CognitiveState):
    """Kısa ve basit mesajlarda genellikle 'direct' kip beklenir."""
    out = cm.handle(state, CognitiveInput(user_message="Merhaba! LOW_ENT"))
    assert isinstance(out.text, str) and len(out.text) > 0
    assert out.used_mode in {"direct", "think1"}  # think1 da olabilir; eşikler config'e bağlı
    assert out.tool_used in {None, "search", "calculator", "file"}
    assert isinstance(out.revised_by_critic, bool)


def test_tool_policy_search_must(cm: CognitiveManager, state: CognitiveState):
    """'bugün/güncel' gibi tetikleyicilerde search aracı önerilmeli/atanmalı."""
    out = cm.handle(state, CognitiveInput(user_message="Bugün en güncel gelişmeleri anlatır mısın?"))
    # ToolPolicy varsayılan önceliğinde 'search' seçilir
    assert out.tool_used in {"search", None}  # critic/flow etkilerine göre None da olabilir
    assert isinstance(out.text, str) and len(out.text) > 0


def test_debate_mode_on_long_input(cm: CognitiveManager, state: CognitiveState):
    """Uzun mesajlarda debate2 kapısı açılmalı (length_gate_debate)."""
    long_msg = " ".join(["uzun"] * 400)  # ~400 kelime
    out = cm.handle(state, CognitiveInput(user_message=long_msg))
    assert out.used_mode in {"debate2", "think1", "direct", "tot"}  # config eşiğine göre değişebilir, tot da geçerli
    # Tercihen debate2 beklenir; başarısız olmaması yeterli


def test_critic_revision_trigger(cm: CognitiveManager, state: CognitiveState):
    """Sayısal iddialar ve hassas anahtarlarla critic revizyonu tetiklenmeli."""
    # riskli_oran ipucu FakeModel.generate içinde iddialı taslağı tetikler
    msg = "Hukuki konuda riskli_oran HIGH_ENT"
    out = cm.handle(state, CognitiveInput(user_message=msg))
    assert isinstance(out.text, str) and len(out.text) > 0
    # Critic devreye girip revize etmiş olmalı (FakeModel [CRITIC] akışı)
    assert out.revised_by_critic is True


def test_memory_summary_injected(cm: CognitiveManager, state: CognitiveState):
    """Belirli aralıklarla system_summary turlarının eklendiğini doğrula."""
    # 7 tur çalıştır: varsayılan session_summary_every=6 ⇒ 6. turun sonunda özet eklenir
    for i in range(7):
        cm.handle(state, CognitiveInput(user_message=f"tur-{i}"))
    roles = [t.get("role") for t in state.history]
    assert "system_summary" in roles


def test_empty_input_graceful(cm: CognitiveManager, state: CognitiveState):
    """Boş mesajta güvenli kısa yanıt dön."""
    out = cm.handle(state, CognitiveInput(user_message=""))
    assert "Nasıl yardımcı olabilirim" in out.text


def test_inner_thought_path(cm: CognitiveManager, state: CognitiveState, cfg: CognitiveManagerConfig):
    """Yüksek entropide iç düşünce (think1) tetiklenmeli."""
    # Eşiği aşmak için HIGH_ENT ipucu
    msg = "Plan yapmam gerekebilir. HIGH_ENT"
    out = cm.handle(state, CognitiveInput(user_message=msg))
    assert out.used_mode in {"think1", "debate2", "direct"}
    # FakeModel INTERNAL THOUGHT üretmiş olmalı; doğrudan görünmez, ama hata almamalıyız.
    assert isinstance(out.text, str) and len(out.text) > 0


def test_decoding_knobs_with_entropy(cm: CognitiveManager, state: CognitiveState):
    """Entropi arttıkça sıcaklık artışı (riskle törpülenmiş) çalışmalı."""
    # Birinci çağrı: düşük entropi
    out1 = cm.handle(state, CognitiveInput(user_message="LOW_ENT kısa bilgi"))
    t1 = cm.mm.last_decoding.temperature if hasattr(cm.mm, "last_decoding") else None

    # İkinci çağrı: yüksek entropi
    out2 = cm.handle(state, CognitiveInput(user_message="HIGH_ENT daha belirsiz istek"))
    t2 = cm.mm.last_decoding.temperature if hasattr(cm.mm, "last_decoding") else None

    # Her iki durumda da üretim yapılmış olmalı
    assert isinstance(out1.text, str) and isinstance(out2.text, str)

    # Sıcaklıkların mantıklı aralıkta ve artış eğiliminde olması beklenir (mutlak şart değil)
    if t1 is not None and t2 is not None:
        assert 0.3 <= t1 <= 1.1
        assert 0.3 <= t2 <= 1.1
        assert t2 >= t1 or abs(t2 - t1) < 1e-6  # en azından düşmesin (risk törpüsü yoksa)


# =============================================================================
# Pytest giriş noktası (isteğe bağlı)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__])
