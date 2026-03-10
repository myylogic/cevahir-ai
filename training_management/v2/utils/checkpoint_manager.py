# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: checkpoint_manager.py
Modül: training_management/v2/utils
Görev: Checkpoint Manager - Model durumlarını güvenli ve yönetilebilir şekilde
       kaydetmek/yüklemek için CheckpointManager. Atomik kayıt (tmp → os.replace),
       last/best alias dosyaları, top-K rotasyon, index.json ile meta yönetimi
       ve güvenli map_location işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (checkpoint yönetimi)
- Design Patterns: Manager Pattern (checkpoint yönetimi)
- Endüstri Standartları: Checkpoint management best practices

KULLANIM:
- Model checkpoint kaydetmek için
- Model checkpoint yüklemek için
- Checkpoint rotasyonu için
- Eğitime kaldığı yerden devam etmek için

BAĞIMLILIKLAR:
- torch: Model state dict işlemleri
- json: Meta yönetimi
- os, shutil: Dosya işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import os
import io
import json
import time
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple, List, Literal

import torch

try:
    from training_management.v2.utils.training_logger import TrainingLogger
except Exception:
    TrainingLogger = None  # type: ignore

# V2: config.parameters artık yok, direkt default değerleri kullanıyoruz
CHECKPOINT_MODEL = os.path.abspath("./checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _now_ts() -> float:
    return time.time()


class CheckpointManager:
    """
    Güvenli checkpoint yönetimi.

    Args:
        checkpoint_model_dir: Kayıt klasörü
        max_checkpoints: En fazla saklanacak checkpoint sayısı
        device: Yükleme aygıtı ('cpu'/'cuda')
        sort_key: Rotasyonda sıralama anahtarı ('ctime' veya 'metric')
        metric_mode: 'min' (düşük daha iyi) veya 'max' (yüksek daha iyi)
        filename_pattern: Kaydetme adı formatı. '{epoch}' ve opsiyonel '{tag}' kullanılabilir.
        logger: TrainingLogger örneği (opsiyonel); yoksa standart logging kullanılır.
    """

    INDEX_FILE = "index.json"
    LAST_LINK = "last.pth"
    BEST_LINK = "best.pth"

    def __init__(
        self,
        checkpoint_model_dir: str = CHECKPOINT_MODEL,
        max_checkpoints: int = 5,
        device: str = DEVICE,
        sort_key: Literal["ctime", "metric"] = "ctime",
        metric_mode: Literal["min", "max"] = "min",
        filename_pattern: str = "checkpoint_epoch_{epoch:04d}{tag}.pth",
        logger: Optional[Any] = None,
    ) -> None:
        self.dir = os.path.abspath(checkpoint_model_dir)
        self.max = int(max_checkpoints)
        self.device = device
        self.sort_key = sort_key
        self.metric_mode = metric_mode
        self.filename_pattern = filename_pattern

        os.makedirs(self.dir, exist_ok=True)

        # ✅ SOLID: Logger dependency injection (TrainingManager'dan geçirilir)
        if logger is not None:
            self.logger = logger
        else:
            # Fallback: Basit console logger (dosya logging yok)
            self.logger = logging.getLogger("CheckpointManager")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)

        # index.json yükle/oluştur
        self.index_path = os.path.join(self.dir, self.INDEX_FILE)
        self.index = self._load_index()

    # ------------------------------------------------------------------ public API

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        *,
        training_history: Optional[Dict[str, Any]] = None,
        metric: Optional[float] = None,
        is_best: Optional[bool] = None,
        tag: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None,
        with_optimizer: bool = True,
    ) -> str:
        """
        Modern kayıt fonksiyonu (önerilen). Atomik kayıt + index güncelleme.

        Returns:
            str: Kaydedilen dosya yolu.
        """
        tag_str = f"_{tag}" if tag else ""
        filename = (
            os.path.basename(filepath)
            if filepath
            else self.filename_pattern.format(epoch=epoch, tag=tag_str)
        )
        path = filepath or os.path.join(self.dir, filename)

        # [DEBUG] Checkpoint kaydetmeden önce model instance kontrolü
        if model is not None:
            model_state_dict_before = model.state_dict()
            model_keys_before = list(model_state_dict_before.keys())
            model_type = type(model).__name__
            is_simple_model = (
                len(model_keys_before) == 3 and 
                all(k in model_keys_before for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            self._log_info("=" * 60)
            self._log_info(f"[CHECKPOINT DEBUG] CheckpointManager.save() - Model kontrolü:")
            self._log_info(f"  Model Type: {model_type}")
            self._log_info(f"  State Dict Keys: {len(model_keys_before)}")
            self._log_info(f"  İlk 10 Key: {model_keys_before[:10]}")
            self._log_info(f"  SimpleModel mi? {is_simple_model}")
            if is_simple_model:
                self._log_info("  ⚠️ KRİTİK UYARI: SimpleModel state_dict kaydediliyor!")
            else:
                self._log_info("  ✅ CevahirNeuralNetwork state_dict kaydediliyor")
            self._log_info("=" * 60)

        # payload
        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "training_history": training_history or {},
            "metric": metric,
            "saved_at": _now_ts(),
        }
        if with_optimizer and optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra_state:
            payload["extra_state"] = dict(extra_state)

        self._atomic_save(payload, path)
        self._update_index_after_save(path, epoch=epoch, metric=metric)

        # last alias
        self._update_alias(self.LAST_LINK, path)

        # best alias (metric varsa ve daha iyiyse)
        better = False
        if is_best is not None:
            better = bool(is_best)
        elif metric is not None:
            better = self._is_strictly_better(metric, self.index.get("best_metric"))

        if better:
            self._update_alias(self.BEST_LINK, path)
            self.index["best_path"] = os.path.basename(path)
            self.index["best_metric"] = metric
            self._save_index()

        # rotasyon
        self._rotate()

        self._log_info(f"Checkpoint kaydedildi → {path}")
        return path

    # Geriye dönük uyum: eski adlar
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        training_history: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None,
    ) -> str:
        """DEPRECATED: save() kullanın. Geriye dönük uyum için tutuldu."""
        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            training_history=training_history,
            filepath=filepath,
            with_optimizer=True,
        )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        filename: Optional[str] = None,
        which: Literal["last", "best", "path"] = "path",
        map_location: Optional[str] = None,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Checkpoint yükle.

        Args:
            filename: which="path" ise zorunlu (tam yol).
            which: "last" (last.pth), "best" (best.pth) veya "path".
            load_optimizer: True ise optimizer state yüklenir.
            strict: model.load_state_dict(strict)

        Returns:
            (epoch, history)
        """
        path = self._resolve_ckpt_path(filename, which)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint bulunamadı: which={which}, filename={filename}")

        mloc = torch.device(map_location or self.device)
        ckpt = torch.load(path, map_location=mloc)

        # doğrulama
        self._validate_payload(ckpt, required_keys=["model_state_dict", "epoch"])

        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if load_optimizer and optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                self._log_warning(f"Optimizer state yüklenemedi (devam ediliyor): {e}")

        epoch = int(ckpt.get("epoch", 0))
        history = ckpt.get("training_history", {})
        self._log_info(f"Checkpoint yüklendi → {path} (epoch={epoch})")
        return epoch, history

    # Geriye dönük uyum: eski ad
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str,
    ) -> Dict[str, Any]:
        """DEPRECATED: load() kullanın. Geriye dönük uyum için tutuldu."""
        epoch, hist = self.load(model, optimizer, filename=filename, which="path")
        return {"epoch": epoch, "training_history": hist}

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        which: Literal["last", "best"] = "last",
        map_location: Optional[str] = None,
        load_optimizer: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Eğitimde kaldığın yerden devam. Dönüş: (start_epoch, history)
        Not: start_epoch = loaded_epoch + 1
        """
        epoch, hist = self.load(
            model=model,
            optimizer=optimizer,
            which=which,
            map_location=map_location,
            load_optimizer=load_optimizer,
        )
        start_epoch = (epoch or 0) + 1
        return start_epoch, hist

    def list_checkpoints(self) -> List[str]:
        """Klasördeki pth dosyalarını isim sırasına göre döndürür."""
        try:
            files = sorted(glob.glob(os.path.join(self.dir, "*.pth")))
            return files
        except Exception as e:
            self._log_error(f"Checkpoint listesi alınamadı: {e}")
            return []

    def get_last_checkpoint(self) -> Optional[str]:
        path = os.path.join(self.dir, self.LAST_LINK)
        return path if os.path.exists(path) else None

    def get_best_checkpoint(self) -> Optional[str]:
        path = os.path.join(self.dir, self.BEST_LINK)
        return path if os.path.exists(path) else None

    # --------------------------------------------------------------- internal I/O

    def _ensure_dir(self) -> None:
        os.makedirs(self.dir, exist_ok=True)

    def _atomic_save(self, obj: Dict[str, Any], dst_path: str) -> None:
        """tmp dosyaya yaz → fsync → os.replace ile atomik kaydet."""
        self._ensure_dir()
        tmp_path = dst_path + ".tmp"
        try:
            # torch.save doğrudan tmp_path'e
            with io.BytesIO() as buffer:
                torch.save(obj, buffer)
                data = buffer.getvalue()
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dst_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _update_alias(self, alias_name: str, target_path: str) -> None:
        """Windows uyumlu: symlink yerine kopya/replace kullan."""
        alias_path = os.path.join(self.dir, alias_name)
        try:
            # Var ise atomik replace
            shutil.copy2(target_path, alias_path)
        except Exception as e:
            self._log_warning(f"Alias güncellenemedi ({alias_name}): {e}")

    # ------------------------------------------------------------------- index db

    def _load_index(self) -> Dict[str, Any]:
        path = os.path.join(self.dir, self.INDEX_FILE)
        if not os.path.exists(path):
            idx = {"checkpoints": [], "best_path": None, "best_metric": None}
            self._save_index(idx)
            return idx
        try:
            with open(path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            # basit doğrulama
            idx.setdefault("checkpoints", [])
            idx.setdefault("best_path", None)
            idx.setdefault("best_metric", None)
            return idx
        except Exception:
            # bozuk ise yeniden oluştur
            idx = {"checkpoints": [], "best_path": None, "best_metric": None}
            self._save_index(idx)
            return idx

    def _save_index(self, idx: Optional[Dict[str, Any]] = None) -> None:
        obj = idx if idx is not None else self.index
        tmp = os.path.join(self.dir, self.INDEX_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.index_path)

    def _update_index_after_save(self, path: str, *, epoch: int, metric: Optional[float]) -> None:
        name = os.path.basename(path)
        entry = {
            "path": name,
            "epoch": int(epoch),
            "ctime": os.path.getctime(path),
            "metric": metric,
        }
        self.index["checkpoints"].append(entry)
        self._save_index()

    # ------------------------------------------------------------------- rotation

    def _rotate(self) -> None:
        """max_checkpoints kuralını uygula."""
        try:
            items = list(self.index.get("checkpoints", []))
            if len(items) <= self.max:
                return

            # Silinmeye adayları belirle
            to_delete = len(items) - self.max
            if self.sort_key == "metric":
                # metric'e göre kötüleri at (None en kötü)
                def score(x):
                    m = x.get("metric")
                    if m is None:
                        return float("inf") if self.metric_mode == "min" else float("-inf")
                    return m

                reverse = (self.metric_mode == "min")
                # kötü → iyi sıralayıp baştan to_delete seç
                sorted_items = sorted(items, key=score, reverse=reverse)
                victims = sorted_items[:to_delete]
            else:
                # ctime'a göre en eskiyi sil
                victims = sorted(items, key=lambda x: x.get("ctime", 0))[:to_delete]

            # Sil ve index’ten düş
            for v in victims:
                p = os.path.join(self.dir, v["path"])
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        self._log_info(f"Eski checkpoint silindi → {p}")
                except Exception as e:
                    self._log_warning(f"Checkpoint silinemedi ({p}): {e}")
                items.remove(v)

            self.index["checkpoints"] = items
            self._save_index()

        except Exception as e:
            self._log_warning(f"Rotasyon hatası: {e}")

    # ------------------------------------------------------------------ utilities

    def _resolve_ckpt_path(
        self, filename: Optional[str], which: Literal["last", "best", "path"]
    ) -> Optional[str]:
        if which == "last":
            p = self.get_last_checkpoint()
            return p
        if which == "best":
            p = self.get_best_checkpoint()
            if p:
                return p
            # index’te best_path varsa onu kullan
            b = self.index.get("best_path")
            if b:
                bp = os.path.join(self.dir, b)
                return bp if os.path.exists(bp) else None
            return None
        if which == "path":
            return filename
        return None

    def _is_strictly_better(self, metric: float, best_metric: Optional[float]) -> bool:
        if best_metric is None:
            return True
        if self.metric_mode == "min":
            return metric < best_metric
        return metric > best_metric

    @staticmethod
    def _validate_payload(obj: Dict[str, Any], required_keys: List[str]) -> None:
        for k in required_keys:
            if k not in obj:
                raise KeyError(f"Checkpoint payload eksik anahtar: {k}")

    # --------------------------------------------------------------------- logging

    def _log_info(self, msg: str) -> None:
        if hasattr(self.logger, "log_info"):
            self.logger.log_info(msg)
        else:
            self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        if hasattr(self.logger, "log_warning"):
            self.logger.log_warning(msg)
        else:
            self.logger.warning(msg)

    def _log_error(self, msg: str) -> None:
        if hasattr(self.logger, "log_error"):
            self.logger.log_error(msg)
        else:
            self.logger.error(msg)
