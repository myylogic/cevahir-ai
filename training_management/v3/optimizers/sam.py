"""
SAM - Sharpness-Aware Minimization Optimizer
=============================================
Referans: Foret et al. 2021, "Sharpness-Aware Minimization for Efficiently
          Improving Generalization" (ICLR 2022)
          https://arxiv.org/abs/2010.01412

Adaptive SAM (ASAM) referansı:
          Kwon et al. 2021, "ASAM: Adaptive Sharpness-Aware Minimization for
          Scale-Invariant Learning of Deep Neural Networks" (ICML 2021)
          https://arxiv.org/abs/2102.11600

Motivasyon:
    Klasik optimizerlar loss fonksiyonunun minimum noktasına (sharp minimum)
    yakınsarlar. Sharp minimum'lar iyi generalization sağlamaz çünkü
    test distribution biraz kaydığında loss dramatik artar.

    SAM bunun yerine 'flat minimum' arar: çevresindeki neighborhood'da
    da düşük loss olan bölgeler. Bu bölgeler daha iyi generalization sağlar.

2-Aşamalı Güncelleme Mekanizması:
    1. first_step:
       - w_adv = w + epsilon  (adversarial perturbation)
       - epsilon = rho * g / ||g||  (normalized gradient yönünde adım)
       - Ağırlıklar geçici olarak w_adv'ye taşınır

    2. second_step:
       - Perturbed noktada gradient hesapla: g(w_adv)
       - Base optimizer ile güncelle: w = w - lr * g(w_adv)
       - Ağırlıkları w'ye (original) geri al

    Adaptive SAM (adaptive=True):
       - epsilon = rho * |w| * g / ||g||  (scale-invariant perturbation)
       - Farklı büyüklükteki ağırlıklar için daha dengeli perturbation

Kullanım:
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=2e-4, weight_decay=0.01)

    # Training loop (her batch için 2 forward-backward pass gerekir):
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        return loss

    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    criterion(model(x), y).backward()
    optimizer.second_step(zero_grad=True)

Dikkat:
    - Her batch için 2 forward pass gerektiğinden computational cost ~2x artar.
    - rho=0.05 çoğu uygulama için iyi başlangıç değeridir.
    - Büyük modellerde rho'yu 0.02-0.05 arasında tutun.
"""

import logging
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Keskin loss valley'lerinden (sharp minima) kaçarak düz minima'ya
    (flat minima) yakınsar. Bu sayede daha iyi generalization elde edilir.

    Args:
        params: Model parametreleri (model.parameters()).
        base_optimizer: Temel optimizer sınıfı (AdamW, SGD vb.).
                        SAM bu optimizer'ı sarar.
        rho (float): Perturbation büyüklüğü. Tipik aralık: 0.01-0.1.
                     Varsayılan: 0.05.
        adaptive (bool): Adaptive SAM (ASAM) kullan. Scale-invariant
                         perturbation için True. Varsayılan: False.
        **kwargs: base_optimizer'a iletilen parametreler (lr, weight_decay vb.).

    Raises:
        ValueError: rho negatifse.

    Referans:
        Foret et al. 2021 - https://arxiv.org/abs/2010.01402
        Kwon et al. 2021 (Adaptive SAM) - https://arxiv.org/abs/2102.11600
    """

    def __init__(
        self,
        params: Union[Iterable, list],
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"rho negatif olamaz: {rho}")

        # SAM varsayılan parametreleri
        defaults: Dict[str, Any] = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        # Temel optimizer'ı başlat (aynı param groups ile)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # param_groups referansını base_optimizer ile senkronize et
        self.param_groups = self.base_optimizer.param_groups

        # SAM defaults'ı her param group'a ekle
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)

        logger.info(
            "SAM başlatıldı: base_optimizer=%s, rho=%.4f, adaptive=%s",
            base_optimizer.__name__,
            rho,
            adaptive,
        )

    # ------------------------------------------------------------------
    # Dahili yardımcı metodlar
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """
        Tüm param gruplarındaki gradient'lerin birleşik L2 normunu hesaplar.

        Adaptive SAM için ağırlıklı norm: ||w * g||
        Normal SAM için: ||g||

        Returns:
            torch.Tensor: Skaler norm değeri.
        """
        # Tüm parametreleri aynı cihaza normalize etmek için referans cihaz
        shared_device = self.param_groups[0]["params"][0].device

        # FIX: accumulate squared norms directly — avoids Python list + torch.stack()
        # allocation; equivalent math: sqrt(sum(||g_i||^2)) = L2 norm of all gradients
        norm_sq = torch.zeros(1, device=shared_device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # FIX: skip torch.ones_like(p) in non-adaptive path — p.grad is sufficient
                g = torch.abs(p) * p.grad if group["adaptive"] else p.grad
                norm_sq += g.norm(p=2).to(shared_device).pow_(2)
        return norm_sq.sqrt().squeeze()

    # ------------------------------------------------------------------
    # SAM güncelleme adımları
    # ------------------------------------------------------------------

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        SAM'ın 1. adımı: Adversarial perturbation uygular.

        Gradient yönünde normalize edilmiş adım atar:
            epsilon = rho * g / ||g||
            w_adv = w + epsilon

        Ağırlıklar geçici olarak perturbed konuma taşınır.
        Bu konumda 2. forward-backward pass yapılmalı, ardından
        second_step() çağrılmalıdır.

        Args:
            zero_grad (bool): Perturbation sonrası gradient'leri sıfırla.
                              Varsayılan: False.

        Not:
            Bu metod çağrılmadan önce loss.backward() ile gradient'ler
            hesaplanmış olmalıdır.
        """
        # Gradient normunu hesapla
        grad_norm = self._grad_norm()

        # Her parametre için perturbation uygula
        for group in self.param_groups:
            # Ölçekleme faktörü: rho / (||g|| + epsilon)
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perturbation: epsilon = scale * (|w|^2 veya 1) * g
                # FIX: skip torch.ones_like(p) in non-adaptive path (identity multiply)
                e_w = (
                    (p.pow(2) * p.grad if group["adaptive"] else p.grad)
                    * scale.to(p)
                )

                # w_adv = w + epsilon (orijinal w'yi SAM state'e kaydet)
                p.add_(e_w)
                # Perturbation vektörünü restore için sakla
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        SAM'ın 2. adımı: Perturbed noktadaki gradient ile güncelleme yapar.

        Öncesinde w_adv konumunda gradient hesaplanmış olmalıdır.
        Bu metod:
            1. w_adv → w (ağırlıkları geri alır)
            2. g(w_adv) ile base_optimizer günceller

        Args:
            zero_grad (bool): Güncelleme sonrası gradient'leri sıfırla.
                              Varsayılan: False.

        Not:
            Bu metod first_step() ve ikinci loss.backward() çağrısından sonra
            çağrılmalıdır.
        """
        # Ağırlıkları orijinal konuma geri al (w_adv → w)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "e_w" not in self.state[p]:
                    logger.warning(
                        "SAM second_step: '%s' parametresi için e_w bulunamadı. "
                        "first_step() çağrıldı mı?",
                        p.shape,
                    )
                    continue
                # w = w_adv - epsilon (geri al)
                p.sub_(self.state[p]["e_w"])

        # Base optimizer ile gerçek güncellemeyi yap
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Standart optimizer interface için fallback step metodu.

        Eğer closure verilmişse, SAM'ın 2-adımlı güncellemesini otomatik yapar.
        Bu yöntem kullanımı more convenient ancak less flexible'dır.

        Args:
            closure (Callable, optional): Loss hesaplayan kapatma fonksiyonu.
                                          Kullanım: lambda: criterion(model(x), y)

        Returns:
            Optional[torch.Tensor]: closure verilmişse loss değeri, yoksa None.

        Not:
            Önerilmez: Manuel first_step/second_step kullanımı daha esnektir.
            Bu metod yalnızca standart Optimizer API uyumluluğu içindir.
        """
        if closure is None:
            # Closure yoksa sadece base optimizer step at
            logger.warning(
                "SAM.step() closure olmadan çağrıldı. "
                "Manuel first_step/second_step kullanımı önerilir."
            )
            return self.base_optimizer.step()

        # İlk forward-backward (perturbation için)
        with torch.enable_grad():
            loss = closure()
        loss.backward()
        self.first_step(zero_grad=True)

        # İkinci forward-backward (actual update için)
        with torch.enable_grad():
            loss = closure()
        loss.backward()
        self.second_step(zero_grad=True)

        return loss

    # ------------------------------------------------------------------
    # State dict yönetimi (checkpoint desteği)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """
        Optimizer durumunu checkpoint için döndürür.

        SAM state ve base_optimizer state'ini birleştirir.

        Returns:
            Dict: SAM ve base_optimizer state'ini içeren sözlük.
        """
        sam_state = super().state_dict()
        base_state = self.base_optimizer.state_dict()
        return {
            "sam_state": sam_state,
            "base_optimizer_state": base_state,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Checkpoint'ten optimizer durumunu yükler.

        Args:
            state_dict (Dict): state_dict() ile kaydedilen sözlük.

        Raises:
            KeyError: Gerekli anahtarlar eksikse.
        """
        required = {"sam_state", "base_optimizer_state"}
        missing = required - state_dict.keys()
        if missing:
            raise KeyError(f"state_dict'te eksik alanlar: {missing}")

        super().load_state_dict(state_dict["sam_state"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])

        logger.info("SAM state_dict yüklendi.")

    def __repr__(self) -> str:
        return (
            f"SAM(base_optimizer={self.base_optimizer.__class__.__name__}, "
            f"rho={self.param_groups[0]['rho']}, "
            f"adaptive={self.param_groups[0]['adaptive']})"
        )
