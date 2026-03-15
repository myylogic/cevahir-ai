"""
Lookahead Optimizer Wrapper
===========================
Referans: Zhang et al. 2019, "Lookahead Optimizer: k steps forward, 1 step back"
          NeurIPS 2019
          https://arxiv.org/abs/1907.08610

Motivasyon:
    Stochastic gradient methodları (SGD, Adam vb.) yüksek variance nedeniyle
    oscillation (salınım) gösterebilir. Lookahead, bu salınımları azaltmak için
    iki ağırlık seti tutar:

    fast_weights:  Her adımda base optimizer ile güncellenir (hızlı, gürültülü)
    slow_weights:  Her k adımda interpolasyon ile güncellenir (yavaş, kararlı)

    slow = slow + alpha * (fast - slow)   [k adımda bir]
    fast = slow                            [slow'a sıfırla]

    Bu mekanizma:
    - Yüksek frekanslı gradient noise'u filtreler
    - Daha kararlı convergence sağlar
    - Herhangi bir base optimizer ile kullanılabilir

Önerilen Değerler:
    k=5, alpha=0.5 (orijinal makale)
    SAM ile birleştirildiğinde: Lookahead(SAM(params, AdamW, rho=0.05), k=5, alpha=0.5)

Kullanım:
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    # Normal training loop:
    for batch in dataloader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()  # Her k adımda bir slow_weights güncellenir
        optimizer.zero_grad()
"""

import copy
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)


class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimizer wrapper.

    Herhangi bir base optimizer'ı sarar ve 'slow weights' mekanizması ekler.
    Her k adımda bir slow weights, fast weights'e doğru interpolasyon yapar.
    Bu sayede oscillation azalır ve daha kararlı convergence elde edilir.

    Args:
        base_optimizer (torch.optim.Optimizer): Sarılacak temel optimizer.
                        (SGD, Adam, AdamW, SAM vb.)
        k (int): Slow weights güncelleme sıklığı (adım sayısı).
                 Küçük k → daha sık güncelleme → daha agresif düzeltme.
                 Büyük k → daha seyrek güncelleme → daha gürültü filtreleme.
                 Önerilen: 5-10. Varsayılan: 5.
        alpha (float): Slow weights interpolasyon oranı. [0, 1] arasında.
                       alpha=0: slow weights hiç güncellenmez.
                       alpha=1: slow weights = fast weights.
                       Önerilen: 0.5. Varsayılan: 0.5.

    Raises:
        ValueError: k < 1 veya alpha dışı aralıkta ise.

    Referans:
        Zhang et al. 2019 - https://arxiv.org/abs/1907.08610
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
    ) -> None:
        if k < 1:
            raise ValueError(f"k en az 1 olmalıdır, alınan: {k}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha [0, 1] arasında olmalıdır, alınan: {alpha}")

        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha

        # Lookahead adım sayacı (her k adımda bir slow weights güncellenir)
        self._lookahead_step: int = 0

        # param_groups'u base_optimizer ile paylaş
        self.param_groups = self.base_optimizer.param_groups
        self.state: Dict = defaultdict(dict)

        # Slow weights'i başlat (fast weights'in kopyası)
        self._init_slow_weights()

        logger.info(
            "Lookahead başlatıldı: base_optimizer=%s, k=%d, alpha=%.2f",
            base_optimizer.__class__.__name__,
            k,
            alpha,
        )

    # ------------------------------------------------------------------
    # Dahili yardımcı metodlar
    # ------------------------------------------------------------------

    def _init_slow_weights(self) -> None:
        """
        Slow weights'i fast weights'in değerleriyle başlatır.

        Her parametre için slow_weights = current_param kopyası oluşturur.
        Sadece float tensor'lar için slow weights tutulur.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    param_state = self.state[p]
                    # Slow weights başlangıçta fast weights ile aynı
                    param_state["slow_weight"] = p.detach().clone()

    # ------------------------------------------------------------------
    # Slow/Fast weights yönetimi (context manager için yardımcılar)
    # ------------------------------------------------------------------

    def _backup_and_load_cache(self) -> None:
        """
        Fast weights'i yedekler ve slow weights'i modele yükler.

        average_parameters() veya inference öncesi kullanılır.
        Slow weights daha kararlı olduğundan inference için tercih edilir.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and "slow_weight" in self.state[p]:
                    param_state = self.state[p]
                    # Fast weights'i yedekle
                    param_state["fast_weight_backup"] = p.detach().clone()
                    # Slow weights'i modele yükle
                    p.data.copy_(param_state["slow_weight"])

    def _clear_and_load_backup(self) -> None:
        """
        Slow weights'i kaldırır ve yedeklenen fast weights'i geri yükler.

        _backup_and_load_cache() çağrısının ardından training'e
        devam etmek için çağrılır.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and "fast_weight_backup" in self.state[p]:
                    param_state = self.state[p]
                    # Fast weights'i geri yükle
                    p.data.copy_(param_state["fast_weight_backup"])
                    # Yedeği temizle
                    del param_state["fast_weight_backup"]

    def _update_slow_weights(self) -> None:
        """
        Slow weights'i fast weights yönünde günceller (interpolasyon).

        slow = slow + alpha * (fast - slow)
             = (1 - alpha) * slow + alpha * fast

        Bu güncellemenin ardından fast weights slow weights'e eşitlenerek
        yeni başlangıç noktası belirlenir.
        """
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    param_state = self.state[p]
                    if "slow_weight" not in param_state:
                        # Slow weight yoksa başlat
                        param_state["slow_weight"] = p.detach().clone()
                        continue

                    slow = param_state["slow_weight"]
                    # Interpolasyon: slow = slow + alpha * (fast - slow)
                    slow.add_(p.detach() - slow, alpha=self.alpha)
                    # Fast weights'i slow'a sıfırla (yeni başlangıç)
                    p.data.copy_(slow)

    # ------------------------------------------------------------------
    # Ana step metodu
    # ------------------------------------------------------------------

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Bir optimizer adımı atar.

        Base optimizer ile fast weights'i günceller.
        Her k adımda bir slow weights interpolasyonu yapar.

        Args:
            closure (Callable, optional): Loss hesaplayan closure fonksiyonu.

        Returns:
            Optional[torch.Tensor]: closure verilmişse loss, yoksa None.
        """
        # Base optimizer ile fast weights güncelle
        loss = self.base_optimizer.step(closure)
        self._lookahead_step += 1

        # Her k adımda bir slow weights güncelle
        if self._lookahead_step % self.k == 0:
            self._update_slow_weights()
            logger.debug(
                "Lookahead slow weights güncellendi (adım=%d, k=%d, alpha=%.2f)",
                self._lookahead_step,
                self.k,
                self.alpha,
            )

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Gradient'leri sıfırlar (base optimizer'a iletir)."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # State dict yönetimi (checkpoint desteği)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """
        Lookahead durumunu checkpoint için döndürür.

        Fast weights, slow weights ve base_optimizer state'ini içerir.

        Returns:
            Dict: Tam Lookahead state sözlüğü.
        """
        # Slow weights'i ayrı serileştir (param index'e göre eşle)
        slow_weights_list: List[Optional[torch.Tensor]] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and "slow_weight" in self.state[p]:
                    slow_weights_list.append(
                        self.state[p]["slow_weight"].cpu().clone()
                    )
                else:
                    slow_weights_list.append(None)

        return {
            "lookahead_step": self._lookahead_step,
            "k": self.k,
            "alpha": self.alpha,
            "slow_weights": slow_weights_list,
            "base_optimizer_state": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Checkpoint'ten Lookahead durumunu yükler.

        Args:
            state_dict (Dict): state_dict() ile kaydedilen sözlük.

        Raises:
            KeyError: Gerekli anahtarlar eksikse.
        """
        required = {"lookahead_step", "k", "alpha", "slow_weights", "base_optimizer_state"}
        missing = required - state_dict.keys()
        if missing:
            raise KeyError(f"state_dict'te eksik alanlar: {missing}")

        # Hiper-parametre uyarıları
        if state_dict["k"] != self.k:
            logger.warning(
                "Lookahead load_state_dict: k uyumsuzluğu "
                "(checkpoint=%d, mevcut=%d). Checkpoint değeri kullanılıyor.",
                state_dict["k"],
                self.k,
            )
        if state_dict["alpha"] != self.alpha:
            logger.warning(
                "Lookahead load_state_dict: alpha uyumsuzluğu "
                "(checkpoint=%.4f, mevcut=%.4f). Checkpoint değeri kullanılıyor.",
                state_dict["alpha"],
                self.alpha,
            )

        self._lookahead_step = state_dict["lookahead_step"]
        self.k = state_dict["k"]
        self.alpha = state_dict["alpha"]

        # Base optimizer state'ini yükle
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])

        # Slow weights'i geri yükle
        slow_weights_iter = iter(state_dict["slow_weights"])
        device = self.param_groups[0]["params"][0].device

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    slow_w = next(slow_weights_iter, None)
                    if slow_w is not None:
                        self.state[p]["slow_weight"] = slow_w.to(
                            device=device, dtype=p.dtype
                        )

        logger.info(
            "Lookahead state_dict yüklendi: step=%d, k=%d, alpha=%.2f",
            self._lookahead_step,
            self.k,
            self.alpha,
        )

    # ------------------------------------------------------------------
    # Kullanışlı özellikler
    # ------------------------------------------------------------------

    @property
    def num_lookahead_steps(self) -> int:
        """Toplam Lookahead adımı sayısını döndürür."""
        return self._lookahead_step

    @property
    def slow_weight_updates(self) -> int:
        """Slow weights'in kaç kez güncellendiğini döndürür."""
        return self._lookahead_step // self.k

    def add_param_group(self, param_group: Dict) -> None:
        """
        Yeni parametre grubu ekler (base_optimizer'a iletir).

        Args:
            param_group (Dict): Eklenecek parametre grubu.
        """
        self.base_optimizer.add_param_group(param_group)
        # Yeni grubun parametrelerine slow weights ekle
        for p in param_group["params"]:
            if p.requires_grad:
                self.state[p]["slow_weight"] = p.detach().clone()

    def __repr__(self) -> str:
        return (
            f"Lookahead(base_optimizer={self.base_optimizer.__class__.__name__}, "
            f"k={self.k}, alpha={self.alpha}, "
            f"total_steps={self._lookahead_step})"
        )
