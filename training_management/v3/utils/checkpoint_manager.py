# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ — Training Management V3
================================================================================
Dosya : training_management/v3/utils/checkpoint_manager.py
Modül : CheckpointManager
Görev : Atomik checkpoint kaydetme/yükleme, rotasyon ve metadata yönetimi.

Tasarım kararları:
  • Atomic save  : .tmp dosyasına yaz, ardından os.replace() ile yeniden adlandır.
                   Güç kesintisinde bozuk checkpoint riski sıfıra düşer.
  • Slotlar      : best.pth, last.pth, periodic/epoch_{N}.pth
  • EMA & SWA    : Ayrı anahtarlar olarak checkpoint içine gömülür.
  • Rotasyon     : periodic/ altında en fazla max_checkpoints tutulur.
  • index.json   : Tüm kayıtlı checkpoint'lerin metadata listesi.
  • Verifier     : Kayıt sonrası CheckpointVerifier çağrılır (inject edilebilir).

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDict — checkpoint dosyasının tam sözlüğü
# ---------------------------------------------------------------------------

class CheckpointData(TypedDict, total=False):
    """Checkpoint dosyasının tam şeması."""
    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    ema_state: Optional[Dict[str, Any]]          # EMA model ağırlıkları
    swa_state: Optional[Dict[str, Any]]          # SWA model ağırlıkları
    metrics: Dict[str, float]
    train_config: Dict[str, Any]
    saved_at: str                                # ISO-8601 UTC zaman damgası


# ---------------------------------------------------------------------------
# CheckpointVerifier protokolü — enjekte edilebilir doğrulama hook'u
# ---------------------------------------------------------------------------

class CheckpointVerifier:
    """
    Kayıt sonrası doğrulama hook'u.
    Alt sınıflar verify() metodunu override ederek özel kontroller ekleyebilir.
    """

    def verify(self, path: str, data: CheckpointData) -> bool:
        """
        Dosyayı yeniden yükleyerek model_state anahtarlarının varlığını doğrular.

        Args:
            path: Kaydedilen checkpoint dosyasının tam yolu.
            data: Kaydedilen veri sözlüğü (referans).

        Returns:
            True → doğrulama başarılı, False → hata var.
        """
        try:
            loaded: dict = torch.load(path, map_location="cpu", weights_only=False)
            # Temel anahtarların varlığını kontrol et
            required = {"epoch", "model_state"}
            missing = required - set(loaded.keys())
            if missing:
                logger.error(
                    "[CheckpointVerifier] Doğrulama başarısız — eksik anahtarlar: %s",
                    missing,
                )
                return False
            # Kaydedilen ve yeniden yüklenen epoch değerlerinin eşleştiğini kontrol et
            if loaded.get("epoch") != data.get("epoch"):
                logger.error(
                    "[CheckpointVerifier] Epoch uyuşmazlığı: beklenen=%s, okunan=%s",
                    data.get("epoch"),
                    loaded.get("epoch"),
                )
                return False
            logger.debug(
                "[CheckpointVerifier] OK — epoch=%s, path=%s",
                data.get("epoch"),
                path,
            )
            return True
        except Exception as exc:
            logger.error("[CheckpointVerifier] Okuma hatası: %s", exc)
            return False


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    V3 için atomik checkpoint yöneticisi.

    Parametre açıklamaları:
        checkpoint_dir     : Tüm checkpoint'lerin saklanacağı kök dizin.
        max_checkpoints    : periodic/ klasöründe tutulacak maksimum checkpoint sayısı.
        save_every_n_epochs: Her N epoch'ta bir periyodik checkpoint kaydedilir.
        verifier           : CheckpointVerifier örneği. None ise doğrulama atlanır.

    Dizin yapısı::

        checkpoint_dir/
            best.pth           ← En iyi metrik checkpoint'i
            last.pth           ← Son epoch checkpoint'i
            index.json         ← Tüm kayıtların metadata listesi
            periodic/
                epoch_0010.pth
                epoch_0020.pth
                ...
    """

    _INDEX_FILE = "index.json"
    _BEST_FILE  = "best.pth"
    _LAST_FILE  = "last.pth"
    _PERIODIC_DIR = "periodic"

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_every_n_epochs: int = 10,
        verifier: Optional[CheckpointVerifier] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max(1, max_checkpoints)
        self.save_every_n_epochs = max(1, save_every_n_epochs)
        self.verifier: Optional[CheckpointVerifier] = verifier

        # Alt dizinleri oluştur
        self._periodic_dir = self.checkpoint_dir / self._PERIODIC_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._periodic_dir.mkdir(parents=True, exist_ok=True)

        # index.json'u belleğe yükle
        self._index: List[Dict[str, Any]] = self._load_index()

        logger.info(
            "[CheckpointManager] Başlatıldı — dir=%s, max=%d, every=%d",
            self.checkpoint_dir,
            self.max_checkpoints,
            self.save_every_n_epochs,
        )

    # ------------------------------------------------------------------
    # Public API — kaydetme
    # ------------------------------------------------------------------

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Checkpoint'i atomik olarak kaydeder.

        EMA nesnesi, `.shadow` veya `.state_dict()` attribute'larına sahip olabilir
        (torch_ema, home-brew EMA gibi). Her iki durumu da işler.

        SWA modeli varsa caller, `ema` yerine swa modelini ayrı geçirir —
        bu metot EMA ve SWA'yı ayrı anahtarlarda saklar.

        Returns:
            Kaydedilen dosyanın tam yolu (str).
        """
        if metrics is None:
            metrics = {}
        if train_config is None:
            train_config = {}

        # --- Veri hazırlama ---
        payload: CheckpointData = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "ema_state": self._extract_ema_state(ema),
            "swa_state": None,          # Ayrıca save_swa() ile güncellenebilir
            "metrics": dict(metrics),
            "train_config": dict(train_config),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        saved_paths: List[str] = []

        # --- last.pth — her epoch ---
        last_path = str(self.checkpoint_dir / self._LAST_FILE)
        self._atomic_save(last_path, payload)
        saved_paths.append(last_path)
        logger.info("[CheckpointManager] last.pth kaydedildi — epoch=%d", epoch)

        # --- best.pth — is_best=True ise ---
        if is_best:
            best_path = str(self.checkpoint_dir / self._BEST_FILE)
            self._atomic_save(best_path, payload)
            saved_paths.append(best_path)
            logger.info(
                "[CheckpointManager] best.pth kaydedildi — epoch=%d, metrics=%s",
                epoch,
                metrics,
            )

        # --- Periyodik checkpoint — her N epoch ---
        primary_path = last_path
        if epoch % self.save_every_n_epochs == 0:
            periodic_name = f"epoch_{epoch:04d}.pth"
            periodic_path = str(self._periodic_dir / periodic_name)
            self._atomic_save(periodic_path, payload)
            saved_paths.append(periodic_path)
            primary_path = periodic_path
            logger.info(
                "[CheckpointManager] Periyodik checkpoint kaydedildi — %s",
                periodic_name,
            )
            # Rotasyon
            self._rotate()

        # --- index.json güncelle ---
        self._update_index(primary_path, epoch, metrics, is_best)

        # --- Doğrulama hook'u ---
        if self.verifier is not None:
            for p in saved_paths:
                ok = self.verifier.verify(p, payload)
                if not ok:
                    logger.warning(
                        "[CheckpointManager] Doğrulama başarısız — %s", p
                    )

        return primary_path

    def save_swa(
        self,
        epoch: int,
        swa_model: torch.nn.Module,
        base_checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        SWA model state'ini mevcut last.pth checkpoint'ine ekler.
        base_checkpoint_path None ise last.pth kullanılır.
        """
        target_path = base_checkpoint_path or str(
            self.checkpoint_dir / self._LAST_FILE
        )
        if not os.path.exists(target_path):
            logger.warning(
                "[CheckpointManager] SWA kaydı için hedef bulunamadı: %s", target_path
            )
            return
        try:
            data: dict = torch.load(target_path, map_location="cpu", weights_only=False)
            data["swa_state"] = swa_model.state_dict()
            self._atomic_save(target_path, data)
            logger.info(
                "[CheckpointManager] SWA state kaydedildi — epoch=%d", epoch
            )
        except Exception as exc:
            logger.error("[CheckpointManager] SWA kayıt hatası: %s", exc)

    # ------------------------------------------------------------------
    # Public API — yükleme
    # ------------------------------------------------------------------

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        best.pth'i yükler ve model/optimizer/scheduler/ema'ya state uygular.

        Returns:
            Yüklenen tam checkpoint sözlüğü.
        """
        path = str(self.checkpoint_dir / self._BEST_FILE)
        return self._load_and_apply(path, model, optimizer, scheduler, ema)

    def load_last(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        last.pth'i yükler ve model/optimizer/scheduler/ema'ya state uygular.
        """
        path = str(self.checkpoint_dir / self._LAST_FILE)
        return self._load_and_apply(path, model, optimizer, scheduler, ema)

    def load_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Belirli bir epoch'un periyodik checkpoint'ini yükler.

        Aranan dosya: periodic/epoch_{epoch:04d}.pth
        """
        periodic_name = f"epoch_{epoch:04d}.pth"
        path = str(self._periodic_dir / periodic_name)
        return self._load_and_apply(path, model, optimizer)

    # ------------------------------------------------------------------
    # Public API — sorgulama
    # ------------------------------------------------------------------

    def get_best_metric(self, metric: str = "val_loss") -> float:
        """
        index.json'daki en iyi checkpoint'in ilgili metriğini döner.
        Metrik bulunamazsa float('inf') döner.
        """
        best_entries = [e for e in self._index if e.get("is_best", False)]
        if not best_entries:
            # best flagli yoksa tüm listeden val_loss'u minimize et
            values = [
                e.get("metrics", {}).get(metric, float("inf"))
                for e in self._index
            ]
            return min(values, default=float("inf"))
        # En son is_best entry'sinin metriği
        last_best = best_entries[-1]
        return last_best.get("metrics", {}).get(metric, float("inf"))

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        index.json'daki tüm checkpoint kayıtlarını döner.
        Sonuç, epoch'a göre sıralanmış şekilde gelir.
        """
        return sorted(self._index, key=lambda e: e.get("epoch", 0))

    # ------------------------------------------------------------------
    # Private — atomik kayıt
    # ------------------------------------------------------------------

    def _atomic_save(self, path: str, payload: Dict) -> None:
        """
        Payload'u önce geçici bir .tmp dosyasına yazar, ardından atomik
        os.replace() çağrısıyla hedef yola taşır.

        Bu sayede güç kesintisi veya disk dolu durumunda hedef dosya bozulmaz.
        """
        tmp_path = path + ".tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            # tmp dosyasını temizle
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            logger.error("[CheckpointManager] Atomik kayıt başarısız: %s — %s", path, exc)
            raise

    # ------------------------------------------------------------------
    # Private — index yönetimi
    # ------------------------------------------------------------------

    def _update_index(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """
        index.json'a yeni bir kayıt ekler ve dosyayı diske yazar.
        """
        entry: Dict[str, Any] = {
            "path": path,
            "epoch": epoch,
            "metrics": dict(metrics),
            "is_best": is_best,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._index.append(entry)
        self._save_index()

    def _save_index(self) -> None:
        index_path = str(self.checkpoint_dir / self._INDEX_FILE)
        tmp_path = index_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, index_path)
        except Exception as exc:
            logger.error("[CheckpointManager] index.json yazılamadı: %s", exc)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _load_index(self) -> List[Dict[str, Any]]:
        index_path = self.checkpoint_dir / self._INDEX_FILE
        if not index_path.exists():
            return []
        try:
            with open(str(index_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except Exception as exc:
            logger.warning("[CheckpointManager] index.json okunamadı: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private — rotasyon
    # ------------------------------------------------------------------

    def _rotate(self) -> None:
        """
        periodic/ dizinindeki checkpoint'leri epoch sırasına göre sıralar ve
        max_checkpoints limitini aşanları siler.
        """
        try:
            files = sorted(self._periodic_dir.glob("epoch_*.pth"))
            while len(files) > self.max_checkpoints:
                oldest = files.pop(0)
                try:
                    oldest.unlink()
                    logger.debug(
                        "[CheckpointManager] Eski checkpoint silindi: %s", oldest.name
                    )
                    # index'ten de kaldır
                    self._index = [
                        e for e in self._index
                        if Path(e.get("path", "")).name != oldest.name
                    ]
                except OSError as exc:
                    logger.warning(
                        "[CheckpointManager] Silme hatası: %s — %s", oldest, exc
                    )
            # Silme sonrası index güncelle
            if len(self._periodic_dir.glob("epoch_*.pth")) <= self.max_checkpoints:
                self._save_index()
        except Exception as exc:
            logger.error("[CheckpointManager] Rotasyon hatası: %s", exc)

    # ------------------------------------------------------------------
    # Private — yükleme
    # ------------------------------------------------------------------

    def _load_and_apply(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Verilen yoldan checkpoint yükler ve bileşenlere uygular.

        Raises:
            FileNotFoundError: Checkpoint dosyası bulunamazsa.
            RuntimeError: Yükleme sırasında beklenmeyen hata oluşursa.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[CheckpointManager] Checkpoint bulunamadı: {path}"
            )

        logger.info("[CheckpointManager] Yükleniyor: %s", path)
        try:
            data: CheckpointData = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"[CheckpointManager] Checkpoint okunamadı: {path} — {exc}"
            ) from exc

        # --- Model ---
        model_state = data.get("model_state")
        if model_state is None:
            raise RuntimeError(
                f"[CheckpointManager] 'model_state' anahtarı eksik: {path}"
            )
        # strict=False: mimari değişikliklerinde esnek yükleme
        incompatible = model.load_state_dict(model_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "[CheckpointManager] State dict uyumsuzluğu — "
                "eksik=%s, beklenmeyen=%s",
                incompatible.missing_keys[:5],
                incompatible.unexpected_keys[:5],
            )

        # --- Optimizer ---
        if optimizer is not None and data.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(data["optimizer_state"])
            except Exception as exc:
                logger.warning(
                    "[CheckpointManager] Optimizer state yüklenemedi: %s", exc
                )

        # --- Scheduler ---
        if scheduler is not None and data.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(data["scheduler_state"])
            except Exception as exc:
                logger.warning(
                    "[CheckpointManager] Scheduler state yüklenemedi: %s", exc
                )

        # --- EMA ---
        if ema is not None and data.get("ema_state") is not None:
            self._apply_ema_state(ema, data["ema_state"])

        epoch = data.get("epoch", 0)
        metrics = data.get("metrics", {})
        logger.info(
            "[CheckpointManager] Yüklendi — epoch=%d, metrics=%s", epoch, metrics
        )
        return dict(data)

    # ------------------------------------------------------------------
    # Private — EMA yardımcıları
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ema_state(ema: Optional[Any]) -> Optional[Dict[str, Any]]:
        """
        EMA nesnesinden state dict çıkarır.
        Çeşitli EMA implementasyonlarıyla (torch_ema, özel sınıflar) uyumludur.
        """
        if ema is None:
            return None
        # torch_ema veya benzeri: .state_dict()
        if hasattr(ema, "state_dict") and callable(ema.state_dict):
            try:
                return ema.state_dict()
            except Exception:
                pass
        # Özel EMA: .shadow_params veya .shadow sözlüğü
        if hasattr(ema, "shadow_params"):
            return {"shadow_params": [p.clone() for p in ema.shadow_params]}
        if hasattr(ema, "shadow"):
            return {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in ema.shadow.items()
            }
        logger.warning(
            "[CheckpointManager] EMA state çıkarılamadı (bilinmeyen tip: %s)",
            type(ema).__name__,
        )
        return None

    @staticmethod
    def _apply_ema_state(ema: Any, state: Dict[str, Any]) -> None:
        """
        EMA state'ini nesneye uygular.
        """
        if hasattr(ema, "load_state_dict") and callable(ema.load_state_dict):
            try:
                ema.load_state_dict(state)
                return
            except Exception as exc:
                logger.warning(
                    "[CheckpointManager] EMA load_state_dict başarısız: %s", exc
                )
        if hasattr(ema, "shadow_params") and "shadow_params" in state:
            for p, saved in zip(ema.shadow_params, state["shadow_params"]):
                p.data.copy_(saved.data)
            return
        if hasattr(ema, "shadow") and isinstance(state, dict):
            for k, v in state.items():
                if k in ema.shadow:
                    if torch.is_tensor(v):
                        ema.shadow[k].copy_(v)
            return
        logger.warning(
            "[CheckpointManager] EMA state uygulanamadı (bilinmeyen tip: %s)",
            type(ema).__name__,
        )
