"""
Cevahir V3 - Gradyan Yöneticisi
=================================
Bu modül, Cevahir Türkçe dil modelinin eğitiminde gradyan hesaplama,
kırpma ve gürültü ekleme işlemlerini yönetir.

Desteklenen Gradyan Teknikleri:
    1. Standart Gradyan Kırpma    : Küresel L2 norm kırpma (max_norm)
    2. Uyarlamalı Gradyan Kırpma  : AGC - Brock et al. 2021 (NFNets)
    3. Gradyan Gürültüsü          : Neelakantan et al. 2015
    4. Gradyan Sağlık Tespiti     : NaN/Inf/Patlama/Ölü gradyan tespiti

Referanslar:
    - Pascanu et al. (2013): "On the difficulty of training recurrent neural networks"
      (Gradyan kırpma motivasyonu)
    - Brock et al. (2021): "High-Performance Large-Scale Image Recognition Without
      Normalization" (NFNets, Adaptive Gradient Clipping)
    - Neelakantan et al. (2015): "Adding Gradient Noise Improves Learning for Very
      Deep Networks" (Gradyan gürültüsü)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["GradientManager"]


class GradientManager:
    """
    Cevahir V3 Gradyan Yöneticisi
    ==============================
    Model parametrelerinin gradyanlarını hesaplar, kırpar ve düzenler.

    Temel Operasyonlar:
        - clip_gradients          : Standart küresel L2 norm kırpma
        - adaptive_clip_gradients : AGC (parametre norm bazlı uyarlamalı kırpma)
        - calculate_gradient_norm : Tüm gradyanların L2 norm'u
        - detect_gradient_issues  : NaN/Inf/patlama/ölü gradyan tespiti
        - inject_gradient_noise   : Neelakantan et al. 2015 gürültü enjeksiyonu

    Kullanım:
        gm = GradientManager(max_norm=1.0, explosion_threshold=10.0)

        # İleri + geri geçişten sonra:
        norm_before = gm.clip_gradients(model, max_norm=1.0)
        issues = gm.detect_gradient_issues(model)
        if not issues['has_nan']:
            optimizer.step()
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        agc_lambda: float = 0.01,
        agc_eps: float = 1e-3,
        explosion_threshold: float = 10.0,
        noise_eta: float = 0.01,
    ):
        """
        Args:
            max_norm            : Varsayılan küresel L2 norm kırpma eşiği
            agc_lambda          : AGC kırpma faktörü λ (Brock et al. 2021)
            agc_eps             : AGC'de sıfır bölme koruması için epsilon
            explosion_threshold : Bu değerin üzerindeki norm → patlama uyarısı
            noise_eta           : Gradyan gürültüsü ölçek parametresi η
        """
        self.max_norm = max_norm
        self.agc_lambda = agc_lambda
        self.agc_eps = agc_eps
        self.explosion_threshold = explosion_threshold
        self.noise_eta = noise_eta

        logger.debug(
            f"GradientManager başlatıldı | max_norm={max_norm}, "
            f"agc_lambda={agc_lambda}, explosion_threshold={explosion_threshold}"
        )

    # ------------------------------------------------------------------
    # Parametre Yardımcıları
    # ------------------------------------------------------------------

    @staticmethod
    def _trainable_params_with_grad(
        model: nn.Module,
    ) -> Iterator[nn.Parameter]:
        """
        Eğitilebilir ve gradyanı hesaplanmış parametreleri verir.

        Args:
            model: PyTorch nn.Module modeli

        Yields:
            Gradyanı olan eğitilebilir parametre tensörleri
        """
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                yield p

    @staticmethod
    def _all_trainable_params(
        model: nn.Module,
    ) -> Iterator[nn.Parameter]:
        """
        Tüm eğitilebilir parametreleri (gradyan olup olmadığından bağımsız) verir.

        Args:
            model: PyTorch nn.Module modeli

        Yields:
            Eğitilebilir parametre tensörleri
        """
        for p in model.parameters():
            if p.requires_grad:
                yield p

    # ------------------------------------------------------------------
    # 1. Standart Gradyan Kırpma
    # ------------------------------------------------------------------

    def clip_gradients(
        self,
        model: nn.Module,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
    ) -> float:
        """
        Küresel L2 norm gradyan kırpma (standart yöntem).

        Pascanu et al. (2013) tarafından önerilen yöntem:
            - Tüm parametre gradyanlarının birleşik L2 norm'unu hesapla
            - Norm max_norm'u aşıyorsa gradyanları ölçekle

        Türkçe dil modellemesinde önemi:
            - Transformer'larda attention katmanlarında patlayan gradyanları önler
            - max_norm=1.0 derin modeller için yaygın varsayılan değer

        Args:
            model    : Eğitilen PyTorch modeli
            max_norm : Maksimum izin verilen gradyan norm'u (None → self.max_norm)
            norm_type: Norm tipi (2.0 → L2, 1.0 → L1, float('inf') → max)

        Returns:
            Kırpma öncesi gradyan norm'u (float). Bu değer izleme için kaydedilebilir.

        Örnek:
            norm = gm.clip_gradients(model, max_norm=0.5)
            logger.info(f"Kırpma öncesi norm: {norm:.4f}")
        """
        _max_norm = max_norm if max_norm is not None else self.max_norm

        # Gradyanı olan parametreleri topla
        params_with_grad = [
            p for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        if not params_with_grad:
            logger.debug("clip_gradients: Hesaplanmış gradyan bulunamadı")
            return 0.0

        # [PERF FIX] Önceki kod önce _compute_total_norm() ile norm hesaplıyor,
        # ardından clip_grad_norm_() ile tekrar hesaplatıyordu (2 tam iterasyon).
        # clip_grad_norm_() zaten norm'u döndürür — tek geçişte hem kırp hem ölç.
        total_norm_tensor = nn.utils.clip_grad_norm_(
            params_with_grad, _max_norm, norm_type=norm_type
        )
        total_norm = total_norm_tensor.item()  # Tek GPU→CPU senkronizasyon noktası

        if total_norm > _max_norm * 2:
            logger.debug(
                f"Gradyan kırpıldı: {total_norm:.4f} → {_max_norm:.4f}"
            )

        return total_norm

    def _compute_total_norm(
        self,
        parameters: List[nn.Parameter],
        norm_type: float = 2.0,
    ) -> float:
        """
        Verilen parametrelerin birleşik L_p norm'unu hesaplar.

        Args:
            parameters: Parametre listesi
            norm_type : Norm tipi

        Returns:
            Toplam gradyan norm'u (float)
        """
        if norm_type == float("inf"):
            norms = [p.grad.detach().abs().max().item() for p in parameters]
            return max(norms) if norms else 0.0
        else:
            norms = [
                p.grad.detach().norm(norm_type).item() ** norm_type
                for p in parameters
            ]
            return sum(norms) ** (1.0 / norm_type)

    # ------------------------------------------------------------------
    # 2. Uyarlamalı Gradyan Kırpma (AGC)
    # ------------------------------------------------------------------

    def adaptive_clip_gradients(
        self,
        model: nn.Module,
        agc_lambda: Optional[float] = None,
        agc_eps: Optional[float] = None,
    ) -> float:
        """
        Uyarlamalı Gradyan Kırpma (AGC) - Brock et al. 2021 (NFNets).

        Standart kırpmadan farkı:
        - Küresel bir eşik yerine HER parametre için bağımsız eşik kullanır
        - Eşik: λ · ||W||_F / ||∇W||_F
          (parametre Frobenius norm'unun λ katı / gradyan norm'u)

        Avantajları:
        - Büyük parametreler → büyük gradyan toleransı
        - Küçük parametreler → küçük gradyan toleransı
        - Batch normalizasyonu olmadan stabiliteyi artırır (NFNets'in temel fikri)

        Formül (parametre p için):
            grad_norm  = ||∇p||_F
            param_norm = max(||p||_F, eps)
            clip_factor = min(1, λ · param_norm / grad_norm)
            ∇p ← ∇p · clip_factor

        Args:
            model      : PyTorch modeli
            agc_lambda : AGC kırpma faktörü λ (None → self.agc_lambda)
            agc_eps    : Sıfır bölme koruması (None → self.agc_eps)

        Returns:
            Kırpma öncesi maksimum parametre-gradyan norm oranı

        Referans:
            Brock, A., et al. (2021). High-Performance Large-Scale Image Recognition
            Without Normalization. ICML 2021.
        """
        _lambda = agc_lambda if agc_lambda is not None else self.agc_lambda
        _eps = agc_eps if agc_eps is not None else self.agc_eps

        max_ratio = 0.0
        clipped_count = 0

        for param in model.parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Parametre ve gradyan Frobenius norm'ları
            grad_norm = param.grad.detach().norm(2).item()
            param_norm = param.detach().norm(2).item()

            # Sıfır parametre veya gradyan durumu
            if grad_norm == 0.0:
                continue

            # Etkin parametre normu (sıfır bölmeyi engelle)
            effective_param_norm = max(param_norm, _eps)

            # Oran izleme (max değer)
            ratio = grad_norm / effective_param_norm
            max_ratio = max(max_ratio, ratio)

            # AGC kırpma faktörü: min(1, λ · ||p|| / ||∇p||)
            clip_threshold = _lambda * effective_param_norm
            if grad_norm > clip_threshold:
                # Ölçekleme faktörü: threshold / grad_norm
                scale = clip_threshold / grad_norm
                param.grad.detach().mul_(scale)
                clipped_count += 1

        if clipped_count > 0:
            logger.debug(
                f"AGC: {clipped_count} parametre kırpıldı, "
                f"maks oran={max_ratio:.4f}, λ={_lambda}"
            )

        return max_ratio

    # ------------------------------------------------------------------
    # 3. Gradyan Norm Hesaplama
    # ------------------------------------------------------------------

    def calculate_gradient_norm(
        self,
        model: nn.Module,
        norm_type: float = 2.0,
    ) -> float:
        """
        Model parametrelerinin toplam gradyan L2 norm'unu hesaplar.

        Eğitim izleme için temel metrik:
        - Çok yüksek → patlayan gradyan (kırpma gerekli)
        - Çok düşük → kaybolan gradyan (mimari/öğrenme oranı sorunu)
        - Stabil → sağlıklı eğitim

        Args:
            model     : PyTorch modeli
            norm_type : Norm tipi (varsayılan L2 = 2.0)

        Returns:
            Toplam gradyan norm'u. Gradyan yoksa 0.0 döner.
        """
        params = list(self._trainable_params_with_grad(model))
        if not params:
            return 0.0

        return self._compute_total_norm(params, norm_type)

    # ------------------------------------------------------------------
    # 4. Gradyan Sağlık Tespiti
    # ------------------------------------------------------------------

    def detect_gradient_issues(
        self,
        model: nn.Module,
        explosion_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Model gradyanlarında sayısal ve istatistiksel sorunları tespit eder.

        Tespit Edilen Sorunlar:
            - has_nan       : NaN içeren gradyanlar (genellikle loss NaN olduğunda)
            - has_inf       : Sonsuz değerli gradyanlar (taşma durumu)
            - has_explosion : Norm patlama eşiğini aşıyorsa
            - max_grad      : Görülen en büyük gradyan değeri
            - total_norm    : Toplam L2 gradyan norm'u
            - dead_ratio    : Sıfır gradyanlı parametre oranı (gradyan ölümü)

        Gradyan Ölümü (dead_ratio):
            Eğer parametrenin tüm gradyanları sıfırsa, o parametre öğrenmiyor
            demektir. Yüksek dead_ratio → mimari/aktivasyon fonksiyonu sorunu.

        Args:
            model               : PyTorch modeli
            explosion_threshold : Patlama eşiği (None → self.explosion_threshold)

        Returns:
            Dict:
                has_nan       (bool)  : NaN gradyan varlığı
                has_inf       (bool)  : Inf gradyan varlığı
                has_explosion (bool)  : Gradyan patlaması
                max_grad      (float) : Maksimum gradyan büyüklüğü
                total_norm    (float) : Toplam L2 norm
                dead_ratio    (float) : Sıfır gradyanlı parametre oranı [0, 1]
                param_count   (int)   : Toplam eğitilebilir parametre sayısı
                grad_count    (int)   : Gradyanı olan parametre sayısı

        Örnek:
            issues = gm.detect_gradient_issues(model)
            if issues['has_nan']:
                logger.error("NaN gradyan tespit edildi!")
                nan_recovery.trigger()
        """
        _threshold = (
            explosion_threshold
            if explosion_threshold is not None
            else self.explosion_threshold
        )

        has_nan = False
        has_inf = False
        max_grad_val = 0.0
        total_sq_sum = 0.0

        all_trainable_params = list(self._all_trainable_params(model))
        params_with_grad = []
        dead_params = 0  # Gradyanı tamamen sıfır olan parametre sayısı

        for param in all_trainable_params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            params_with_grad.append(param)

            # NaN kontrolü
            if torch.isnan(grad).any():
                has_nan = True

            # Inf kontrolü
            if torch.isinf(grad).any():
                has_inf = True

            # Maksimum gradyan büyüklüğü
            with torch.no_grad():
                abs_max = grad.abs().max().item()
                max_grad_val = max(max_grad_val, abs_max)

            # L2 norm katkısı (toplama için)
            with torch.no_grad():
                total_sq_sum += grad.norm(2).item() ** 2

            # Ölü gradyan tespiti: tüm değerler sıfır mı?
            if not grad.any():
                dead_params += 1

        # Toplam L2 norm
        total_norm = math.sqrt(total_sq_sum)

        # Patlama tespiti
        has_explosion = total_norm > _threshold

        # Ölü gradyan oranı
        grad_count = len(params_with_grad)
        dead_ratio = dead_params / max(grad_count, 1)

        result = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "has_explosion": has_explosion,
            "max_grad": max_grad_val,
            "total_norm": total_norm,
            "dead_ratio": dead_ratio,
            "param_count": len(all_trainable_params),
            "grad_count": grad_count,
        }

        # Kritik sorunlar için uyarı
        if has_nan:
            logger.warning(f"GradientManager: NaN gradyan tespit edildi! norm={total_norm:.4f}")
        if has_inf:
            logger.warning(f"GradientManager: Inf gradyan tespit edildi!")
        if has_explosion:
            logger.warning(
                f"GradientManager: Gradyan patlaması! "
                f"norm={total_norm:.4f} > eşik={_threshold:.4f}"
            )
        if dead_ratio > 0.5:
            logger.warning(
                f"GradientManager: Yüksek ölü gradyan oranı: {dead_ratio:.1%} "
                f"(aktivasyon veya öğrenme oranı sorunu olabilir)"
            )

        return result

    # ------------------------------------------------------------------
    # 5. Gradyan Gürültüsü Enjeksiyonu
    # ------------------------------------------------------------------

    def inject_gradient_noise(
        self,
        model: nn.Module,
        eta: Optional[float] = None,
        step: int = 1,
        noise_type: str = "gaussian",
    ) -> float:
        """
        Gradyanlara Gaussian gürültüsü ekler (Neelakantan et al. 2015).

        Formül:
            std = η / (1 + t)^0.55

            Nerede:
                η  = gürültü ölçeği (noise_eta)
                t  = eğitim adımı numarası (step)
                0.55 = bozunma üssü (kağıtta önerilmiş)

        Motivasyon:
            - Keskin minimumlardan kaçmaya yardımcı olur (yerel optimum tuzağı)
            - Çok derin ağlarda öğrenmeyi iyileştirir
            - Düzenleştirici etki (örtük)
            - Adım büyüdükçe gürültü azalır (annealing)

        Türkçe dil modellemesinde kullanım:
            - Eğitim başında (step küçükken) daha fazla gürültü → keşif
            - Eğitim sonunda (step büyüdükçe) gürültü azalır → yakınsama

        Args:
            model     : PyTorch modeli
            eta       : Gürültü ölçeği η (None → self.noise_eta)
            step      : Mevcut eğitim adımı (≥ 1)
            noise_type: 'gaussian' (şimdilik tek desteklenen)

        Returns:
            Uygulanan gürültü standart sapması

        Referans:
            Neelakantan, A., et al. (2015). Adding Gradient Noise Improves
            Learning for Very Deep Networks. ICLR 2016 Workshop.
        """
        _eta = eta if eta is not None else self.noise_eta

        if step < 1:
            step = 1

        # Neelakantan et al. 2015 gürültü std hesabı
        # std = η / (1 + t)^0.55
        noise_std = _eta / ((1 + step) ** 0.55)

        if noise_std < 1e-10:
            # Gürültü pratikte sıfır → atla
            return noise_std

        params_updated = 0
        for param in self._trainable_params_with_grad(model):
            if noise_type == "gaussian":
                # N(0, std²) gürültüsü
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.detach().add_(noise)
                params_updated += 1

        if params_updated > 0:
            logger.debug(
                f"Gradyan gürültüsü eklendi: std={noise_std:.6f} "
                f"(η={_eta}, step={step}, {params_updated} parametre)"
            )

        return noise_std

    # ------------------------------------------------------------------
    # Yardımcı Metotlar
    # ------------------------------------------------------------------

    def get_per_layer_norms(
        self,
        model: nn.Module,
    ) -> Dict[str, float]:
        """
        Her isimlendirilmiş katman için gradyan norm'larını hesaplar.

        Hangi katmanların sorun çıkardığını tespit etmek için kullanılır.
        TensorBoard veya loglama sistemine aktarılabilir.

        Args:
            model: PyTorch modeli

        Returns:
            {katman_adı: gradyan_normu} sözlüğü
        """
        layer_norms: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            norm = param.grad.detach().norm(2).item()
            layer_norms[name] = norm

        return layer_norms

    def zero_gradients(self, model: nn.Module) -> None:
        """
        Model gradyanlarını sıfırlar.

        set_to_none=True kullanımı: None atamak, sıfır tensör oluşturmaktan
        daha hızlı ve bellek açısından daha verimlidir.

        Args:
            model: PyTorch modeli
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

    def get_summary(
        self,
        model: nn.Module,
    ) -> Dict:
        """
        Mevcut gradyan durumunun tam özet raporunu döndürür.

        Args:
            model: PyTorch modeli

        Returns:
            Norm, sağlık durumu ve katman normlarını içeren sözlük
        """
        norm = self.calculate_gradient_norm(model)
        issues = self.detect_gradient_issues(model)
        return {
            "total_norm": norm,
            **issues,
        }

    def __repr__(self) -> str:
        return (
            f"GradientManager("
            f"max_norm={self.max_norm}, "
            f"agc_lambda={self.agc_lambda}, "
            f"explosion_threshold={self.explosion_threshold}, "
            f"noise_eta={self.noise_eta}"
            f")"
        )
