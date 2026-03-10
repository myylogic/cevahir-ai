# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: model_saver.py
Modül: model_management
Görev: Model Saver - Model ve eğitim bilgilerini güvenli bir şekilde kaydetmek için
       kullanılan modül. Model state dict kaydetme, optimizer/scheduler state
       kaydetme, config kaydetme, atomik kayıt (tmp → os.replace) ve checkpoint
       yönetimi işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (model kaydetme)
- Design Patterns: Saver Pattern (model kaydetme)
- Endüstri Standartları: Model saving best practices

KULLANIM:
- Model checkpoint kaydetmek için
- Model state dict kaydetmek için
- Optimizer/scheduler state kaydetmek için
- Config kaydetme için

BAĞIMLILIKLAR:
- torch: PyTorch işlemleri
- json: Config kaydetme
- tempfile: Atomik kayıt için

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Modül-özel logger (root logger'ı yeniden konfig etmeyelim)
saver_logger = logging.getLogger("model_saver")
if not saver_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    saver_logger.addHandler(handler)
    saver_logger.setLevel(logging.INFO)


# ----------------------------- Yardımcılar ----------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Geçici dosyaya yazıp atomik olarak hedefe taşır."""
    dir_ = os.path.dirname(path) or "."
    _ensure_dir(dir_)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # Windows/Unix uyumlu atomik replace
        os.replace(tmp_path, path)
    finally:
        # Beklenmedik hata durumunda tmp kalmışsa temizle
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _torch_save_atomic(obj: Any, path: str) -> None:
    """torch.save'i atomik şekilde uygular."""
    # torch.save doğrudan dosyaya yazar; biz önce bytelara serileştiriyoruz
    # Not: Bu yaklaşım büyük checkpointlerde RAM kullanır; çok büyük dosyalar için
    # doğrudan torch.save(path) tercih edilebilir. İhtiyaca göre flag eklenebilir.
    import io
    
    #  CUDA ASSERT FIX: CUDA context bozulmuşsa CPU'ya taşıyarak kaydet
    try:
        buf = io.BytesIO()
        torch.save(obj, buf)
        _atomic_write_bytes(path, buf.getvalue())
    except (RuntimeError, torch.cuda.CudaError) as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            # CUDA hatası: obj'deki tensor'ları CPU'ya taşı
            saver_logger.warning(f"CUDA hatası tespit edildi, CPU'ya taşıyarak kaydediliyor: {e}")
            try:
                # Obj bir dict ise, içindeki tensor'ları CPU'ya taşı
                if isinstance(obj, dict):
                    cpu_obj = {}
                    for key, value in obj.items():
                        if isinstance(value, torch.Tensor):
                            cpu_obj[key] = value.cpu()
                        elif isinstance(value, dict):
                            # Nested dict (state_dict gibi)
                            cpu_obj[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                        else:
                            cpu_obj[key] = value
                    obj = cpu_obj
                elif isinstance(obj, torch.nn.Module):
                    # Model ise state_dict'i CPU'ya taşı
                    obj = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in obj.state_dict().items()}
                
                # CPU'da tekrar dene
                buf = io.BytesIO()
                torch.save(obj, buf)
                _atomic_write_bytes(path, buf.getvalue())
                saver_logger.info(f"Model CPU'ya taşındıktan sonra başarıyla kaydedildi: {path}")
            except Exception as e2:
                saver_logger.error(f"CPU'ya taşıma sonrası kayıt başarısız: {e2}", exc_info=True)
                raise RuntimeError("Model kaydedilemedi (CUDA hatası ve CPU fallback başarısız).") from e2
        else:
            raise


def _generate_filename(epoch: Optional[int], template: str) -> str:
    if "{epoch" in template:
        if epoch is None:
            raise ValueError("Dosya adı şablonu epoch içeriyor ama epoch verilmedi.")
        return template.format(epoch=epoch)
    return template


def _write_latest_marker(save_dir: str, filename: str) -> None:
    """
    Symlink yerine 'latest.txt' marker dosyası oluşturur.
    (Windows'ta symlink yetki gerektirebilir.)
    """
    try:
        latest_path = os.path.join(save_dir, "latest.txt")
        _atomic_write_bytes(latest_path, filename.encode("utf-8"))
    except Exception as e:
        saver_logger.warning(f"'latest' işaretçisi yazılamadı: {e}")


def _prune_old_checkpoints(save_dir: str, pattern_prefix: str, keep_last_n: int) -> None:
    """Belirli bir önekle başlayan eski checkpointleri budar (sadece aynı klasörde)."""
    try:
        files = sorted(
            [f for f in os.listdir(save_dir) if f.startswith(pattern_prefix) and f.endswith(".pth")]
        )
        if keep_last_n > 0 and len(files) > keep_last_n:
            to_remove = files[:-keep_last_n]
            for fn in to_remove:
                try:
                    os.remove(os.path.join(save_dir, fn))
                except Exception as e:
                    saver_logger.warning(f"Eski checkpoint silinemedi: {fn} ({e})")
    except Exception as e:
        saver_logger.warning(f"Checkpoint budama sırasında sorun: {e}")


# ----------------------------- Ana Sınıf ----------------------------- #

class ModelSaver:
    """
    ModelSaver:
    Model durumunu, optimizer, scheduler ve ek bilgileri kaydetmek için kullanılan bir sınıf.
    """

    # --- Yeni, güçlü API --- #
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        *,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        epoch: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_dir: str = "saved_models",
        filename: Optional[str] = None,
        filename_template: str = "checkpoint_ep{epoch:04d}.pth",
        create_latest_marker: bool = True,
        keep_last_n: int = 0,
        prefix_for_prune: str = "checkpoint_",
    ) -> str:
        """
        Tam checkpoint kaydeder: model/optimizer/scheduler state_dict + epoch/config/metadata.

        Returns:
            str: Kaydedilen dosyanın tam yolu.
        """
        try:
            _ensure_dir(save_dir)

            # Dosya adı üret
            if filename:
                fname = filename
            else:
                # epoch verilmemişse şablonda epoch yerini kullanmayan bir ad türet
                if epoch is None and "{epoch" in filename_template:
                    fname = "checkpoint.pth"
                else:
                    fname = _generate_filename(epoch, filename_template)
            path = os.path.join(save_dir, fname)

            # Checkpoint objesi
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "config": config,
                "metadata": metadata,
            }

            # Atomik kaydet
            _torch_save_atomic(ckpt, path)
            saver_logger.info(f"Checkpoint kaydedildi: {path}")

            # latest işaretçisi
            if create_latest_marker:
                _write_latest_marker(save_dir, fname)

            # Budama
            if keep_last_n and keep_last_n > 0:
                _prune_old_checkpoints(save_dir, prefix_for_prune, keep_last_n)

            return path

        except Exception as e:
            saver_logger.error(f"Checkpoint kaydı başarısız: {e}", exc_info=True)
            raise RuntimeError("Model checkpoint kaydedilemedi.") from e

    @staticmethod
    def save_weights_only(
        model: nn.Module,
        *,
        save_dir: str = "saved_models",
        filename: str = "weights.pth",
    ) -> str:
        """Sadece model ağırlıklarını (state_dict) kaydeder."""
        try:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, filename)
            _torch_save_atomic(model.state_dict(), path)
            saver_logger.info(f"Ağırlıklar kaydedildi: {path}")
            return path
        except Exception as e:
            saver_logger.error(f"Ağırlık kaydı başarısız: {e}", exc_info=True)
            raise RuntimeError("Ağırlıklar kaydedilemedi.") from e

    @staticmethod
    def save_full_model(
        model: nn.Module,
        *,
        save_dir: str = "saved_models",
        filename: str = "full_model.pth",
    ) -> str:
        """
        Modeli tek dosyada kaydeder (pickle). Not: Genellikle state_dict tercih edilir.
        """
        try:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, filename)
            _torch_save_atomic(model, path)
            saver_logger.info(f"Tam model dosyası kaydedildi: {path}")
            return path
        except Exception as e:
            saver_logger.error(f"Tam model kaydı başarısız: {e}", exc_info=True)
            raise RuntimeError("Tam model kaydedilemedi.") from e

    @staticmethod
    def save_additional_info(
        info: Dict[str, Any],
        *,
        save_dir: str = "saved_models",
        filename: str = "additional_info.json",
    ) -> str:
        """Ek bilgileri JSON formatında atomik olarak kaydeder."""
        try:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, filename)
            data = json.dumps(info, ensure_ascii=False, indent=4).encode("utf-8")
            _atomic_write_bytes(path, data)
            saver_logger.info(f"Ek bilgiler kaydedildi: {path}")
            return path
        except Exception as e:
            saver_logger.error(f"Ek bilgiler kaydı başarısız: {e}", exc_info=True)
            raise RuntimeError("Ek bilgiler kaydedilemedi.") from e

    # --- Geriye dönük uyumlu API (wrap) --- #
    @staticmethod
    def save_model(
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        save_dir: str = "models/",
        model_name: str = "model.pth",
    ) -> None:
        """
        Eski imza ile uyumlu kalalım: tek bir .pth içine hepsini yazar.
        (Yeni API: save_checkpoint önerilir.)
        """
        try:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, model_name)
            ckpt = {
                "state_dict": model.state_dict(),  # Geriye dönük uyumluluk için
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,  # Geriye dönük uyumluluk için
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state": scheduler.state_dict() if scheduler else None,  # Geriye dönük uyumluluk için
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": additional_info.get("epoch") if additional_info else None,
                "config": additional_info.get("config") if additional_info else None,
                "additional_info": additional_info,  # Test'lerde additional_info olarak bekleniyor
                "metadata": additional_info,  # Geriye dönük uyumluluk için
            }
            _torch_save_atomic(ckpt, path)
            saver_logger.info(f"Model ve ilgili durumlar kaydedildi: {path}")
        except Exception as e:
            saver_logger.error(f"Model kaydı başarısız: {e}", exc_info=True)
            raise RuntimeError("Model kaydedilemedi.") from e
