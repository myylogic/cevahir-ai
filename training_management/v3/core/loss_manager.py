"""
Cevahir V3 - Bileşik Kayıp Fonksiyonu Yöneticisi
==================================================
Bu modül, Cevahir Türkçe dil modelinin eğitiminde kullanılan
bileşik kayıp fonksiyonu sistemini implemente eder.

Desteklenen Kayıp Fonksiyonları:
    1. CrossEntropyLoss       - Etiket yumuşatma ile standart çapraz entropi
    2. EntropyRegularizationLoss - Çıkış dağılımını düzenlileştirme (Pereyra et al. 2017)
    3. FocalLoss              - Zor örneklere odaklanma (Lin et al. 2017 RetinaNet)
    4. AuxiliaryLoss          - MoE+MoD yardımcı kayıpları (3-demet model çıktısı)

Referanslar:
    - Pereyra et al. (2017): "Regularizing Neural Networks by Penalizing Confident
      Output Distributions" (Entropi Düzenlileştirme)
    - Lin et al. (2017): "Focal Loss for Dense Object Detection" (FocalLoss)
    - Müller et al. (2019): "When Does Label Smoothing Help?" (Etiket Yumuşatma)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

__all__ = [
    "CompositeLossManager",
    "LossConfig",
    "LossOutput",
    "compute_accuracy",
    "compute_perplexity",
    "compute_entropy",
]


# ---------------------------------------------------------------------------
# Paylaşımlı Arayüz Tanımları
# ---------------------------------------------------------------------------

class LossOutput(TypedDict):
    """
    CompositeLossManager.compute() tarafından döndürülen kayıp çıktısı.

    Alanlar:
        total          : Geriye yayılım için kullanılan toplam kayıp tensörü
        ce             : Çapraz entropi kaybı (Python float)
        entropy_reg    : Entropi düzenlileştirme kaybı (Python float)
        focal          : Focal kayıp değeri (Python float)
        auxiliary      : MoE/MoD yardımcı kayıpları toplamı (Python float)
        label_smoothing: Etiket yumuşatma katsayısı (config'den)
    """
    total: torch.Tensor
    ce: float
    entropy_reg: float
    focal: float
    auxiliary: float
    label_smoothing: float


# ---------------------------------------------------------------------------
# Konfigürasyon Veri Sınıfı
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """
    CompositeLossManager için tüm parametreleri içeren konfigürasyon sınıfı.

    Parametre Açıklamaları:
        label_smoothing       : CE kaybında etiket yumuşatma katsayısı [0, 1)
        entropy_coeff         : Entropi düzenlileştirme ağırlığı (Pereyra 2017)
        focal_gamma           : Focal loss odak parametresi γ (Lin 2017)
        focal_weight          : Focal loss'un toplam kayıptaki ağırlığı
        auxiliary_weight      : Yardımcı kayıpların toplam kayıptaki ağırlığı
        pad_token_id          : Dolgu token ID'si (kayıp hesabında atlanır)
        eos_token_id          : Cümle sonu token ID'si
        eos_weight            : EOS token'ına uygulanan sınıf ağırlığı (< 1.0 azaltır)
        min_response_tokens   : İlk bu kadar pozisyonda EOS maskeli (erken bitiş engeli)
        vocab_size            : Model kelime dağarcığı boyutu
        use_focal             : Focal loss aktif mi?
        use_entropy_reg       : Entropi düzenlileştirme aktif mi?
        use_auxiliary         : Yardımcı kayıplar aktif mi?
        class_weights         : Özel sınıf ağırlıkları (vocab_size boyutunda)
    """
    # Temel parametreler
    label_smoothing: float = 0.1
    pad_token_id: int = 0
    eos_token_id: int = 2
    vocab_size: int = 32000

    # Entropi düzenlileştirme (Pereyra et al. 2017)
    entropy_coeff: float = 0.01
    use_entropy_reg: bool = True

    # Focal loss (Lin et al. 2017)
    focal_gamma: float = 2.0
    focal_weight: float = 0.5
    use_focal: bool = True

    # Yardımcı kayıplar (MoE + MoD)
    auxiliary_weight: float = 0.1
    use_auxiliary: bool = True

    # EOS token kontrolü
    eos_weight: float = 0.3           # EOS sınıf ağırlığı (1.0 = normal)
    min_response_tokens: int = 10     # İlk N pozisyonda EOS yasaklı

    # Özel sınıf ağırlıkları (None ise otomatik oluşturulur)
    class_weights: Optional[List[float]] = field(default=None)


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar (Modül Seviyesi)
# ---------------------------------------------------------------------------

def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
) -> float:
    """
    Token-seviyesinde doğruluk hesaplar (dolgu tokenları hariç).

    Args:
        logits  : Model çıktıları [batch, seq_len, vocab_size]
        targets : Hedef token ID'leri [batch, seq_len]
        pad_id  : Hesaplamaya dahil edilmeyecek dolgu token ID'si

    Returns:
        Dolgu dışı token doğruluğu [0.0, 1.0]
    """
    with torch.no_grad():
        # En yüksek olasılıklı token tahminleri
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]

        # Dolgu tokenlarını maske dışında bırak
        non_pad_mask = targets != pad_id      # [batch, seq_len]

        # Doğru tahmin edilen non-pad tokenlar
        correct = (predictions == targets) & non_pad_mask
        total_non_pad = non_pad_mask.sum().item()

        if total_non_pad == 0:
            return 0.0

        return correct.sum().item() / total_non_pad


def compute_perplexity(loss: float) -> float:
    """
    Kayıp değerinden perplexity (şaşkınlık) hesaplar.

    Perplexity = exp(CE_loss), dil modellerinin standart değerlendirme metriği.
    Düşük perplexity → model dil dağılımını daha iyi öğrenmiş demektir.

    Args:
        loss: Ortalama çapraz entropi kaybı (negatif log-likelihood per token)

    Returns:
        Perplexity değeri. Çok büyük değerler (>10000) NaN/Inf'e karşı kırpılır.
    """
    try:
        # Taşma engellemek için max değer sınırı
        clamped_loss = min(loss, math.log(10000))
        return math.exp(clamped_loss)
    except (OverflowError, ValueError):
        logger.warning(f"Perplexity hesaplanamadı, loss={loss:.4f}")
        return float("inf")


def compute_entropy(logits: torch.Tensor) -> float:
    """
    Model çıkış dağılımının Shannon entropisi H(p) = -Σ p·log(p).

    Yüksek entropi → model emin değil (dağılım düz)
    Düşük entropi → model emin (dağılım sivri)

    Türkçe dil modellemesinde entropi izleme:
    - Eğitim başlarında yüksek entropi beklenir
    - Aşırı düşük entropi → model collapse belirtisi
    - Aşırı yüksek entropi → yetersiz öğrenme

    Args:
        logits: Model çıktıları [batch, seq_len, vocab_size] veya [batch, vocab_size]

    Returns:
        Ortalama Shannon entropisi (nats cinsinden)
    """
    # ⚠️  MEMORY-SAFE: chunk bazlı hesaplama
    # Naif softmax(logits) → (B*T, V) float32 tensör ~ 16-18 GB ile OOM.
    # log_softmax chunk'ı: sadece (512, V) tutulur, hemen silinir.
    _CHUNK = 512
    with torch.no_grad():
        flat = logits.reshape(-1, logits.shape[-1])  # (N, V)
        _ents: list = []
        for _s in range(0, flat.shape[0], _CHUNK):
            _lp = torch.log_softmax(flat[_s : _s + _CHUNK].float(), dim=-1)
            _ents.append(-(_lp.exp() * _lp).sum(dim=-1))
            del _lp
        return torch.cat(_ents).mean().item()


# ---------------------------------------------------------------------------
# Focal Loss İmplementasyonu
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss - Zor Sınıflandırma Örneklerine Odaklanma
    =====================================================
    Lin et al. (2017) "Focal Loss for Dense Object Detection" (RetinaNet)

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Geleneksel çapraz entropiye kıyasla:
    - Kolay örnekler (yüksek p_t): (1-p_t)^γ faktörü kaybı küçültür
    - Zor örnekler (düşük p_t): faktör 1'e yakın, standart CE gibi davranır
    - γ=0: standart çapraz entropiye eşdeğer

    Türkçe dil modellemesinde önemi:
    - Yaygın tokenlar (ve, bir, bu...) kolay tahmin edilir → ağırlıkları azalır
    - Nadir/zor Türkçe ekler, uzun kelimeler daha fazla ağırlık alır
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma       : Odak parametresi γ ≥ 0. Büyüdükçe kolay örnekler daha az ağırlık alır.
            weight      : Sınıf ağırlıkları [vocab_size] tensörü
            ignore_index: Bu ID'ye sahip tokenlar kayıp hesabından atlanır
            reduction   : 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits : [batch * seq_len, vocab_size] veya [batch, seq_len, vocab_size]
            targets: [batch * seq_len] veya [batch, seq_len]

        Returns:
            Scalar kayıp tensörü (reduction='mean' varsayılan)
        """
        # Boyutları düzleştir: [N, vocab_size] ve [N]
        if logits.dim() == 3:
            batch, seq_len, vocab = logits.shape
            logits = logits.reshape(batch * seq_len, vocab)
            targets = targets.reshape(batch * seq_len)

        # Geçerli (ignore edilmeyen) tokenları maskele
        valid_mask = targets != self.ignore_index
        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask]

        if logits_valid.numel() == 0:
            return logits.new_tensor(0.0)

        # Log-softmax ile sayısal kararlı log-olasılıklar
        log_probs = F.log_softmax(logits_valid.float(), dim=-1)  # [N_valid, vocab]

        # Hedef tokenların log-olasılıklarını topla
        # gather: her token için doğru sınıfın log-p'sini seç
        log_p_t = log_probs.gather(
            dim=1,
            index=targets_valid.unsqueeze(1)
        ).squeeze(1)  # [N_valid]

        # p_t = exp(log_p_t)
        p_t = log_p_t.exp()

        # Focal ağırlık: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma  # [N_valid]

        # Focal loss: -(1-p_t)^γ · log(p_t)
        focal_loss = -focal_weight * log_p_t  # [N_valid]

        # Sınıf ağırlığı uygula (EOS down-weighting vb.)
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_valid]  # [N_valid]
            focal_loss = focal_loss * class_w

        # Azaltma
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------------------------------------------------------
# Entropi Düzenlileştirme Kaybı
# ---------------------------------------------------------------------------

class EntropyRegularizationLoss(nn.Module):
    """
    Entropi Düzenlileştirme Kaybı (Confidence Penalty)
    ===================================================
    Pereyra et al. (2017) "Regularizing Neural Networks by Penalizing
    Confident Output Distributions"

    Formül: L_total = L_CE - β · H(p)
    H(p) = -Σ p · log(p)  (Shannon entropisi)

    Amaç: Modelin çıkış dağılımını çok sivri yapmasını (overconfidence)
    engellemek. Eğitim seti üzerinde aşırı uyuma karşı düzenlileştirici.

    Türkçe dil modellemesinde önemi:
    - Türkçe'de eklemeli yapı nedeniyle belirsizlik doğal → entropi yararlı
    - Çok düşük entropi → model bağlamı dikkate almıyor (dejenere çözüm)
    - entropy_coeff tipik aralık: [0.001, 0.1]
    """

    def __init__(self, entropy_coeff: float = 0.01):
        """
        Args:
            entropy_coeff: β katsayısı. Büyüdükçe dağılımlar daha düz tutulur.
        """
        super().__init__()
        self.entropy_coeff = entropy_coeff

    def forward(
        self,
        logits: torch.Tensor,
        ignore_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Entropi düzenlileştirme kaybını hesaplar (negatif → minimize edilince entropi artar).

        Args:
            logits      : [..., vocab_size] model çıktıları
            ignore_mask : True olan pozisyonlar atlanır [batch, seq_len]

        Returns:
            Skalar: -entropy_coeff * H(p)  (bu değer toplam kayıptan çıkarılacak)
        """
        # ⚠️  MEMORY-SAFE: chunk bazlı hesaplama — gradient akışı torch.cat ile korunur.
        # probs (B*T,V) + log_probs (B*T,V) aynı anda ~35 GB → OOM.
        # Her chunk yalnızca (512, V) tutar, hemen silinir.
        _CHUNK = 512
        flat = logits.reshape(-1, logits.shape[-1])   # (N, V)
        _ents: list = []
        for _s in range(0, flat.shape[0], _CHUNK):
            _lp = torch.log_softmax(flat[_s : _s + _CHUNK].float(), dim=-1)
            _ents.append(-(_lp.exp() * _lp).sum(dim=-1))  # (chunk,)
            del _lp
        entropy = torch.cat(_ents)                    # (N,) — gradient korunur

        # Orijinal şekle geri döndür ve maske uygula
        orig_shape = logits.shape[:-1]                # [batch] veya [batch, seq_len]
        entropy = entropy.reshape(orig_shape)

        # Maske varsa yalnızca geçerli pozisyonlar
        if ignore_mask is not None and entropy.dim() > 0:
            valid_mask = ~ignore_mask
            if valid_mask.any():
                entropy = entropy[valid_mask]

        mean_entropy = entropy.mean()

        # Kayıp olarak negatif entropi: minimize ederken entropi maksimize edilir
        # total_loss -= entropy_coeff * H(p)
        # → loss'a şu eklenir: -entropy_coeff * H(p)
        return -self.entropy_coeff * mean_entropy


# ---------------------------------------------------------------------------
# Ana Bileşik Kayıp Yöneticisi
# ---------------------------------------------------------------------------

class CompositeLossManager:
    """
    Cevahir V3 Bileşik Kayıp Fonksiyonu Yöneticisi
    ================================================
    Birden fazla kayıp fonksiyonunu tek bir arayüzde birleştirir.

    Kayıp bileşenleri:
        1. CrossEntropyLoss    : Temel dil modeli kaybı (etiket yumuşatmalı)
        2. FocalLoss           : Zor tokenlar için ek odaklanma
        3. EntropyRegularization: Overconfidence engelleyici
        4. AuxiliaryLoss       : MoE yönlendirme + MoD seçim kayıpları

    Toplam kayıp:
        L = CE + w_focal·FL + (-β·H(p)) + w_aux·L_aux

    EOS Token Kontrolü:
        - EOS tokenları düşük sınıf ağırlığı (eos_weight < 1.0) alır
        - İlk min_response_tokens pozisyonda EOS logit'leri -inf yapılır
          (model çok erken cümle bitiremez)

    Kullanım:
        config = LossConfig(vocab_size=32000, eos_token_id=2)
        manager = CompositeLossManager(config, device='cuda')

        output: LossOutput = manager.compute(logits, targets, aux_loss=0.05)
        output['total'].backward()
    """

    def __init__(
        self,
        config: LossConfig,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            config: Tüm kayıp parametrelerini içeren LossConfig nesnesi
            device: Hesaplama cihazı ('cpu', 'cuda', 'mps')
        """
        self.config = config
        self.device = torch.device(device)

        # Sınıf ağırlıklarını oluştur veya kullan
        self._class_weights = self._build_class_weights()

        # Çapraz entropi kaybı (etiket yumuşatmalı)
        self._ce_loss = nn.CrossEntropyLoss(
            weight=self._class_weights,
            ignore_index=config.pad_token_id,
            label_smoothing=config.label_smoothing,
            reduction="mean",
        )

        # Focal loss
        self._focal_loss = FocalLoss(
            gamma=config.focal_gamma,
            weight=self._class_weights,
            ignore_index=config.pad_token_id,
        ) if config.use_focal else None

        # Entropi düzenlileştirme
        self._entropy_reg = EntropyRegularizationLoss(
            entropy_coeff=config.entropy_coeff,
        ) if config.use_entropy_reg else None

        logger.info(
            f"CompositeLossManager başlatıldı | "
            f"label_smooth={config.label_smoothing}, "
            f"focal={'ON' if config.use_focal else 'OFF'} γ={config.focal_gamma}, "
            f"entropy_reg={'ON' if config.use_entropy_reg else 'OFF'} β={config.entropy_coeff}, "
            f"aux={'ON' if config.use_auxiliary else 'OFF'} w={config.auxiliary_weight}"
        )

    # ------------------------------------------------------------------
    # Dahili Yardımcılar
    # ------------------------------------------------------------------

    def _build_class_weights(self) -> Optional[torch.Tensor]:
        """
        Token sınıf ağırlık vektörü oluşturur.
        - Dolgu token: 0.0 (kayıp hesabından dışlanır)
        - EOS token  : config.eos_weight (varsayılan 0.3, azaltılmış)
        - Diğerleri  : 1.0

        EOS down-weighting nedeni: Model, her zaman en kısa yanıtı üretmeye
        yönelmesin. EOS'u erken tahmin etmek kolay (yüksek frekans) ama
        içerik üretimi için zararlı.
        """
        cfg = self.config

        if cfg.class_weights is not None:
            # Kullanıcı tanımlı ağırlıklar
            weights = torch.tensor(cfg.class_weights, dtype=torch.float32)
            assert weights.shape[0] == cfg.vocab_size, (
                f"class_weights boyutu {weights.shape[0]} != vocab_size {cfg.vocab_size}"
            )
        else:
            # Varsayılan: hepsi 1.0
            weights = torch.ones(cfg.vocab_size, dtype=torch.float32)

        # Dolgu token ağırlığı sıfır (CE ignore_index ile zaten atlanıyor,
        # ama weight=0 da ek güvence sağlar)
        weights[cfg.pad_token_id] = 0.0

        # EOS token ağırlığını azalt
        weights[cfg.eos_token_id] = cfg.eos_weight

        return weights.to(self.device)

    def _apply_min_response_mask(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        min_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        İlk min_tokens pozisyonda EOS hedeflerini PAD (ignore_index) olarak işaretler.

        Modelin çok kısa yanıtlar üretmesini engeller; Türkçe cevaplar için
        minimum uzunluk garantisi sağlar.

        [PERF FIX] Önceki uygulama logits tensörünü (.clone() ile) kopyalıyordu:
            - logits: [batch, seq_len, vocab_size] ≈ 64 × 512 × 32000 × 2 bayt ≈ 4 GB
            - Yalnızca tek bir sütun (EOS token) değiştiriliyordu
        Yeni uygulama targets tensörünü kopyalar:
            - targets: [batch, seq_len] ≈ 64 × 512 × 8 bayt ≈ 256 KB — 16.000× daha küçük
            - EOS olan hedef pozisyonları pad_token_id (ignore_index) olarak işaretlenir
            - CrossEntropyLoss(ignore_index=pad_token_id) bu pozisyonları zaten atlar
        Davranış farkı: Logit maskeleme (logit → -inf) yerine hedef maskeleme
        (target → ignore) — her ikisi de modelin o pozisyonda EOS'u doğru tahmin
        etmesini kayıp hesabının dışında tutar.

        Args:
            logits    : [batch, seq_len, vocab_size] — değiştirilmeden döndürülür
            targets   : [batch, seq_len] — EOS pozisyonları maskelenir
            min_tokens: EOS'un yasaklı olduğu ilk N pozisyon sayısı

        Returns:
            (logits, masked_targets): logits değişmez, targets'ta ilk N EOS → pad_id
        """
        if min_tokens <= 0:
            return logits, targets

        seq_len = targets.shape[1]
        mask_len = min(min_tokens, seq_len)
        if mask_len == 0:
            return logits, targets

        # [B, mask_len] — yalnızca küçük dilimdeki EOS pozisyonlarını işaretle
        masked_targets = targets.clone()
        eos_positions = masked_targets[:, :mask_len] == self.config.eos_token_id
        masked_targets[:, :mask_len][eos_positions] = self.config.pad_token_id
        return logits, masked_targets

    def _extract_aux_loss(
        self,
        model_output: Union[torch.Tensor, Tuple],
        aux_loss_override: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Model çıktısından logits ve yardımcı kaybı çıkarır.

        Cevahir modeli 3-demet döndürebilir: (logits, moe_loss, mod_loss)
        Bu fonksiyon bu demet yapısını ele alır.

        Args:
            model_output    : Tensor veya (logits, aux1, aux2) demetleri
            aux_loss_override: Dışarıdan geçirilen yardımcı kayıp (öncelikli)

        Returns:
            (logits_tensor, aux_loss_scalar veya None)
        """
        if isinstance(model_output, torch.Tensor):
            # Düz tensor çıktı: yardımcı kayıp yok
            return model_output, None

        if isinstance(model_output, (tuple, list)):
            logits = model_output[0]
            if len(model_output) >= 3:
                # (logits, moe_loss, mod_loss) → topla
                aux_losses = [
                    x for x in model_output[1:]
                    if isinstance(x, torch.Tensor)
                ]
                if aux_losses:
                    combined_aux = sum(aux_losses)
                    return logits, combined_aux.item()
            elif len(model_output) == 2 and isinstance(model_output[1], torch.Tensor):
                return logits, model_output[1].item()
            return logits, None

        return model_output, None

    # ------------------------------------------------------------------
    # Ana Hesaplama Arayüzü
    # ------------------------------------------------------------------

    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_loss: Optional[Union[torch.Tensor, float]] = None,
    ) -> LossOutput:
        """
        Tüm kayıp bileşenlerini hesaplar ve LossOutput döndürür.

        Hesaplama Adımları:
            1. EOS minimum yanıt maskesi uygula
            2. Çapraz entropi kaybını hesapla
            3. Focal kaybı hesapla (config'de aktifse)
            4. Entropi düzenlileştirmeyi hesapla (aktifse)
            5. Yardımcı kaybı ekle (aktifse)
            6. Toplam kaybı birleştir

        Args:
            logits : Model çıktıları [batch, seq_len, vocab_size]
            targets: Hedef token ID'leri [batch, seq_len]
            aux_loss: MoE/MoD yardımcı kaybı (tensor veya float, opsiyonel)

        Returns:
            LossOutput TypedDict (backward için 'total' alanını kullan)

        Örnek:
            output = manager.compute(logits, targets, aux_loss=moe_loss)
            output['total'].backward()
            print(f"CE: {output['ce']:.4f}, Focal: {output['focal']:.4f}")
        """
        cfg = self.config

        # Cihaz uyumluluğunu sağla
        logits = logits.to(self.device)
        targets = targets.to(self.device)

        # ---------------------------------------------------
        # Adım 1: EOS minimum yanıt maskesi (target-space)
        # ---------------------------------------------------
        if cfg.min_response_tokens > 0:
            logits, targets = self._apply_min_response_mask(
                logits, targets, cfg.min_response_tokens
            )

        # ---------------------------------------------------
        # Adım 2: Çapraz Entropi Kaybı
        # ---------------------------------------------------
        # CE, [batch, vocab, seq_len] biçimi bekler (PyTorch convention)
        if logits.dim() == 3:
            # [batch, seq, vocab] → [batch, vocab, seq]
            ce_logits = logits.permute(0, 2, 1)
        else:
            ce_logits = logits

        ce_value = self._ce_loss(ce_logits, targets)
        total_loss = ce_value

        # ---------------------------------------------------
        # Adım 3: Focal Loss
        # ---------------------------------------------------
        focal_value = 0.0
        if cfg.use_focal and self._focal_loss is not None:
            fl = self._focal_loss(logits, targets)
            focal_value = fl.item()
            total_loss = total_loss + cfg.focal_weight * fl

        # ---------------------------------------------------
        # Adım 4: Entropi Düzenlileştirme
        # ---------------------------------------------------
        entropy_reg_value = 0.0
        if cfg.use_entropy_reg and self._entropy_reg is not None:
            # Dolgu tokenları entropi hesabından çıkar
            pad_mask = (targets == cfg.pad_token_id)
            er = self._entropy_reg(logits, ignore_mask=pad_mask)
            entropy_reg_value = er.item()
            # er zaten negatif (= -β·H), toplam kayıptan çıkarılmak yerine ekliyoruz
            total_loss = total_loss + er

        # ---------------------------------------------------
        # Adım 5: Yardımcı Kayıplar (MoE + MoD)
        # ---------------------------------------------------
        aux_value = 0.0
        if cfg.use_auxiliary and aux_loss is not None:
            if isinstance(aux_loss, torch.Tensor):
                aux_scalar = aux_loss.to(self.device)
                aux_value = aux_scalar.item()
                total_loss = total_loss + cfg.auxiliary_weight * aux_scalar
            elif isinstance(aux_loss, (int, float)):
                aux_value = float(aux_loss)
                # Sayısal yardımcı kayıp tensor'a dönüştür
                aux_tensor = torch.tensor(
                    aux_value, dtype=total_loss.dtype, device=self.device,
                    requires_grad=False
                )
                total_loss = total_loss + cfg.auxiliary_weight * aux_tensor

        return LossOutput(
            total=total_loss,
            ce=ce_value.item(),
            entropy_reg=entropy_reg_value,
            focal=focal_value,
            auxiliary=aux_value,
            label_smoothing=cfg.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Metrik Yardımcı Fonksiyonlar (Sarıcı Metotlar)
    # ------------------------------------------------------------------

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> float:
        """
        Token-seviyesinde doğruluk hesaplar.

        Args:
            logits : [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            pad_id : Atlanacak dolgu token ID'si (None → config'den alınır)

        Returns:
            Dolgu dışı doğruluk oranı [0.0, 1.0]
        """
        _pad_id = pad_id if pad_id is not None else self.config.pad_token_id
        return compute_accuracy(logits, targets, pad_id=_pad_id)

    def compute_perplexity(self, loss: float) -> float:
        """
        Kayıp değerinden perplexity hesaplar.

        Args:
            loss: Ortalama çapraz entropi kaybı

        Returns:
            Perplexity değeri
        """
        return compute_perplexity(loss)

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Çıkış dağılımının Shannon entropisi H(p).

        Args:
            logits: [..., vocab_size] model çıktıları

        Returns:
            Ortalama entropi (nats)
        """
        return compute_entropy(logits)

    def update_class_weights(self, new_weights: torch.Tensor) -> None:
        """
        Eğitim sırasında sınıf ağırlıklarını güncellemek için (curriculum vb.)

        Args:
            new_weights: [vocab_size] yeni ağırlık tensörü
        """
        assert new_weights.shape[0] == self.config.vocab_size
        self._class_weights = new_weights.to(self.device)
        # CE ve Focal loss ağırlıklarını güncelle
        self._ce_loss.weight = self._class_weights
        if self._focal_loss is not None:
            self._focal_loss.weight = self._class_weights
        logger.debug("Sınıf ağırlıkları güncellendi.")

    def get_loss_weights_summary(self) -> Dict[str, float]:
        """
        Mevcut kayıp ağırlıklarının özeti (debug için).

        Returns:
            Kayıp bileşen ağırlıklarını içeren sözlük
        """
        return {
            "label_smoothing": self.config.label_smoothing,
            "focal_weight": self.config.focal_weight if self.config.use_focal else 0.0,
            "entropy_coeff": self.config.entropy_coeff if self.config.use_entropy_reg else 0.0,
            "auxiliary_weight": self.config.auxiliary_weight if self.config.use_auxiliary else 0.0,
            "eos_weight": self.config.eos_weight,
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CompositeLossManager("
            f"label_smoothing={cfg.label_smoothing}, "
            f"focal={'ON' if cfg.use_focal else 'OFF'}(γ={cfg.focal_gamma}), "
            f"entropy_reg={'ON' if cfg.use_entropy_reg else 'OFF'}(β={cfg.entropy_coeff}), "
            f"auxiliary={'ON' if cfg.use_auxiliary else 'OFF'}(w={cfg.auxiliary_weight}), "
            f"eos_weight={cfg.eos_weight}, "
            f"min_resp_tokens={cfg.min_response_tokens}"
            f")"
        )
