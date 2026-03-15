# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: model_loader.py
Modül: model_management
Görev: Model Loader - Kaydedilen modelleri ve ilgili bilgileri yükler. Model state
       dict yükleme, optimizer state yükleme, scheduler state yükleme, config yükleme
       ve checkpoint yükleme işlemlerini yapar. Güvenli map_location ve device
       yönetimi sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (model yükleme)
- Design Patterns: Loader Pattern (model yükleme)
- Endüstri Standartları: Model loading best practices

KULLANIM:
- Model checkpoint yüklemek için
- Model state dict yüklemek için
- Optimizer/scheduler state yüklemek için
- Config yükleme için

BAĞIMLILIKLAR:
- torch: PyTorch işlemleri
- json: Config yükleme

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim

from model_management.exceptions import (
    CheckpointNotFoundError,
    CheckpointCorruptError,
    CheckpointVersionError,
    VocabSizeMismatchError,
)

# Modül-özel logger (root logger'ı yeniden konfig etmeyelim)
loader_logger = logging.getLogger("model_loader")
if not loader_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    loader_logger.addHandler(handler)
    loader_logger.setLevel(logging.INFO)


# ----------------------------- Yardımcılar ----------------------------- #

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Kullanıcı device vermediyse cuda/mps/cpu sırasıyla otomatik seç."""
    if isinstance(device, torch.device):
        return device
    d = (device or "").strip().lower()
    if d in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if d == "cpu" or not d:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    loader_logger.warning(f"Tanınmayan device='{device}', CPU seçildi.")
    return torch.device("cpu")


def _with_aliases(config: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    """'learning_rate' -> 'lr' gibi alias eşlemeleri uygular; hedef anahtar yoksa kopyalar."""
    out = dict(config or {})
    for src, dst in aliases.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    return out


def _filter_kwargs_for_ctor(model_class: Type[nn.Module], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Model __init__ imzasına göre config'ten güvenli argüman süzme."""
    try:
        sig = inspect.signature(model_class.__init__)
        allowed = {p.name for p in sig.parameters.values()
                   if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        allowed.discard("self")
        return {k: v for k, v in cfg.items() if k in allowed}
    except (TypeError, ValueError):
        loader_logger.warning("Model imzası okunamadı; ctor'a config argümanı geçilmeyecek.")
        return {}


def _verify_sha256(path: str, *, raise_on_fail: bool = True) -> bool:
    """
    Disk üzerindeki .sha256 sidecar dosyasıyla checkpoint bütünlüğünü doğrular.

    Döndürür:
        True  → hash eşleşti veya sidecar dosyası yok (doğrulama atlandı).
        False → hash eşleşmedi (raise_on_fail=False ise).

    Fırlatır:
        CheckpointCorruptError → raise_on_fail=True ve hash uyumsuz.
    """
    sha_path = path + ".sha256"
    if not os.path.exists(sha_path):
        loader_logger.debug(f"SHA-256 sidecar yok, bütünlük doğrulaması atlandı: {path}")
        return True   # Sidecar yoksa kontrol edilemez — pass

    try:
        with open(sha_path, "rb") as f:
            expected_hash = f.read().decode("utf-8").strip()
    except Exception as exc:
        loader_logger.warning(f"SHA-256 sidecar okunamadı: {exc}")
        return True   # Okunamazsa → geç

    # Dosyayı streaming hesapla
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    actual_hash = sha.hexdigest()

    if actual_hash != expected_hash:
        msg = f"SHA-256 uyumsuzluğu: beklenen={expected_hash[:16]}..., hesaplanan={actual_hash[:16]}..."
        if raise_on_fail:
            raise CheckpointCorruptError(path=path, reason=msg)
        loader_logger.error(f"[Loader] ❌ {msg}")
        return False

    loader_logger.debug(f"[Loader] ✅ SHA-256 doğrulandı: {actual_hash[:16]}...")
    return True


def _check_version_compatibility(ckpt: Dict[str, Any], path: str) -> None:
    """
    Checkpoint format versiyonunu kontrol eder.
    Kritik uyumsuzluk varsa CheckpointVersionError fırlatır.
    """
    meta = ckpt.get("metadata") or {}
    if not isinstance(meta, dict):
        return

    saved_format = meta.get("checkpoint_format")
    if saved_format is None:
        return   # Eski format (versiyon bilgisi yok) → geç

    # Format 1'den 2'ye: geriye dönük uyumlu (sadece meta eklendi)
    # Format 3+'dan yükleme: uyumsuz (gelecek kırıcı değişiklik için hazırlık)
    _CURRENT_FORMAT = 2
    _MIN_COMPATIBLE = 1

    if int(saved_format) > _CURRENT_FORMAT:
        raise CheckpointVersionError(
            path=path,
            saved_version=saved_format,
            expected_version=f"<={_CURRENT_FORMAT}",
        )


def _torch_load(path: str, map_location: torch.device, *, weights_only: Optional[bool]) -> Any:
    """
    PyTorch 2.x'teki weights_only parametresini destekleyerek güvenli yükleme.
    SHA-256 bütünlük doğrulaması entegre edilmiştir.
    """
    if not os.path.exists(path):
        raise CheckpointNotFoundError(path)

    # SHA-256 bütünlük kontrolü (sidecar varsa)
    _verify_sha256(path, raise_on_fail=True)

    try:
        # PyTorch 2.3+: weights_only var
        if "weights_only" in torch.load.__code__.co_varnames:  # type: ignore[attr-defined]
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        # Eski sürüm uyumu
        return torch.load(path, map_location=map_location)
    except TypeError:
        # Parametre desteklenmiyorsa düş
        return torch.load(path, map_location=map_location)


def _extract_state_dicts(ckpt: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Checkpoint tipini belirle ve state_dict'leri çıkar.
    Döndürür: (model_sd, optimizer_sd, scheduler_sd, meta)
    """
    # Doğrudan OrderedDict (model state_dict'i)
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        keys = set(ckpt.keys())
        # Tam checkpoint
        if "model_state_dict" in keys:
            model_sd = ckpt["model_state_dict"]
            opt_sd = ckpt.get("optimizer_state_dict")
            sch_sd = ckpt.get("scheduler_state_dict")
            meta = {
                "epoch": ckpt.get("epoch"),
                "config": ckpt.get("config"),
            }
            return model_sd, opt_sd, sch_sd, meta
        # Bazı framework'lerde 'state_dict' anahtarı olur
        if "state_dict" in keys and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], None, None, {}
        # Düz state_dict gibi davran
        return ckpt, None, None, {}
    # Farklı tipte ise kullanıcıya bırak
    raise ValueError("Beklenmeyen checkpoint biçimi: model state_dict anahtarları bulunamadı.")


# ----------------------------- Ana Sınıf ----------------------------- #

class ModelLoader:
    """
    ModelLoader:
    Kaydedilen modeli ve ilgili ek bilgileri yüklemek için yardımcı bir sınıf.
    """

    # ---- Model ---- #
    @staticmethod
    def load_model(
        model_class: Type[nn.Module],
        model_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        weights_only: Optional[bool] = None,
    ) -> nn.Module:
        """
        Kaydedilmiş model dosyasını yükler (düz state_dict veya tam checkpoint destekler).

        Args:
            model_class: Yüklenecek model sınıfı.
            model_path: .pth yolu (state_dict veya tam checkpoint olabilir).
            device: 'cpu'/'cuda'/'mps' ya da torch.device.
            config: Model ctor'u için opsiyonel konfig (imzaya göre süzülür).
            extra_model_kwargs: Ctor'a enjekte edilecek ek kwargs.
            strict: state_dict yüklemesini strict yap.
            weights_only: torch 2.x 'weights_only' bayrağı (None → PyTorch'a bırak).

        Returns:
            nn.Module: Yüklenmiş model.
        """
        dev = _resolve_device(device)
        loader_logger.info(f"Model yükleniyor: {model_path} (device={dev}, strict={strict})")

        # 1) Checkpoint'i oku
        ckpt = _torch_load(model_path, map_location=dev, weights_only=weights_only)

        # 1b) Versiyon uyumluluğunu kontrol et
        if isinstance(ckpt, dict):
            _check_version_compatibility(ckpt, model_path)

            # Checkpoint meta verisini logla
            meta_info = (ckpt.get("metadata") or {})
            if isinstance(meta_info, dict):
                cevahir_ver = meta_info.get("cevahir_version", "?")
                saved_at = meta_info.get("saved_at", "?")
                total_params = meta_info.get("total_params")
                loader_logger.info(
                    f"[Loader] Checkpoint meta: "
                    f"cevahir={cevahir_ver}, saved_at={saved_at}"
                    + (f", params={total_params/1e6:.1f}M" if total_params else "")
                )

        # 2) state_dict'leri ayıkla
        model_sd, _, _, _ = _extract_state_dicts(ckpt)

        # 3) Model örneği oluştur
        ctor_cfg = _with_aliases(config or {}, {"learning_rate": "lr", "n_heads": "num_heads"})
        if extra_model_kwargs:
            ctor_cfg.update(extra_model_kwargs)
        ctor_kwargs = _filter_kwargs_for_ctor(model_class, ctor_cfg)

        # 'vocab_size' gibi kritik değer config'te varsa garanti et
        if "vocab_size" not in ctor_kwargs and config and "vocab_size" in config:
            ctor_kwargs["vocab_size"] = config["vocab_size"]

        try:
            model = model_class(**ctor_kwargs).to(dev)
        except TypeError as e:
            loader_logger.error(f"Model ctor argümanları hatalı: {ctor_kwargs}")
            raise

        # 4) Vocab boyutu kontrolü (erken hata → daha net mesaj)
        if "embedding.weight" in model_sd:
            ckpt_vocab = model_sd["embedding.weight"].shape[0]
            model_vocab = model.embedding.weight.shape[0] if hasattr(model, "embedding") else None
            if model_vocab is not None and ckpt_vocab != model_vocab:
                raise VocabSizeMismatchError(
                    model_vocab=model_vocab,
                    checkpoint_vocab=ckpt_vocab,
                )

        # 5) Ağırlıkları yükle
        try:
            missing, unexpected = model.load_state_dict(model_sd, strict=strict)
        except RuntimeError as load_err:
            # Boyut uyumsuzluğunu daha net açıkla
            raise CheckpointCorruptError(
                path=model_path,
                reason=f"state_dict yüklenemedi: {load_err}",
            ) from load_err

        if missing:
            loader_logger.warning(f"[Loader] Eksik anahtarlar ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            loader_logger.warning(f"[Loader] Beklenmeyen anahtarlar ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        loader_logger.info("Model başarıyla yüklendi ve cihaza taşındı.")
        return model

    # ---- Opt/Scheduler ---- #
    @staticmethod
    def load_optimizer(
        optimizer: optim.Optimizer,
        optimizer_path_or_state: Union[str, Dict[str, Any]],
        *,
        device: Optional[Union[str, torch.device]] = None,
        weights_only: Optional[bool] = None,
    ) -> optim.Optimizer:
        """
        Optimizer durumunu dosyadan veya state_dict nesnesinden yükler.
        """
        try:
            if isinstance(optimizer_path_or_state, str):
                dev = _resolve_device(device)
                loader_logger.info(f"Optimizer state yükleniyor: {optimizer_path_or_state}")
                state = _torch_load(optimizer_path_or_state, map_location=dev, weights_only=weights_only)
            else:
                state = optimizer_path_or_state

            optimizer.load_state_dict(state)
            loader_logger.info("Optimizer başarıyla yüklendi.")
            return optimizer

        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Optimizer dosyası bulunamadı: {fnf_error}", exc_info=True)
            raise RuntimeError("Optimizer yüklenemedi.") from fnf_error
        except Exception as e:
            loader_logger.error(f"Optimizer yükleme sırasında hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Optimizer yüklenemedi.") from e

    @staticmethod
    def load_scheduler(
        scheduler: optim.lr_scheduler.LRScheduler,
        scheduler_path_or_state: Union[str, Dict[str, Any]],
        *,
        device: Optional[Union[str, torch.device]] = None,
        weights_only: Optional[bool] = None,
    ) -> optim.lr_scheduler.LRScheduler:
        """
        Scheduler durumunu dosyadan veya state_dict nesnesinden yükler.
        """
        try:
            if isinstance(scheduler_path_or_state, str):
                dev = _resolve_device(device)
                loader_logger.info(f"Scheduler state yükleniyor: {scheduler_path_or_state}")
                state = _torch_load(scheduler_path_or_state, map_location=dev, weights_only=weights_only)
            else:
                state = scheduler_path_or_state

            scheduler.load_state_dict(state)
            loader_logger.info("Scheduler başarıyla yüklendi.")
            return scheduler

        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Scheduler dosyası bulunamadı: {fnf_error}", exc_info=True)
            raise RuntimeError("Scheduler yüklenemedi.") from fnf_error
        except Exception as e:
            loader_logger.error(f"Scheduler yükleme sırasında hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Scheduler yüklenemedi.") from e

    # ---- Ek bilgiler ---- #
    @staticmethod
    def load_additional_info(info_path: str) -> Dict[str, Any]:
        """
        Ek bilgileri JSON formatında yükler (ör. eğitim geçmişi, metrikler).
        """
        try:
            loader_logger.info(f"Ek bilgiler yükleniyor: {info_path}")
            if not os.path.exists(info_path):
                raise FileNotFoundError(f"Ek bilgi dosyası bulunamadı: {info_path}")

            with open(info_path, "r", encoding="utf-8") as json_file:
                additional_info = json.load(json_file)

            loader_logger.info("Ek bilgiler başarıyla yüklendi.")
            return additional_info

        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Ek bilgi dosyası bulunamadı: {fnf_error}", exc_info=True)
            raise RuntimeError("Ek bilgiler yüklenemedi.") from fnf_error
        except json.JSONDecodeError as json_error:
            loader_logger.error(f"JSON formatında hata: {json_error}", exc_info=True)
            raise RuntimeError("Ek bilgiler JSON formatında değil.") from json_error
        except Exception as e:
            loader_logger.error(f"Ek bilgiler yüklenirken hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Ek bilgiler yüklenemedi.") from e

    # ---- Tek çağrıda hepsi ---- #
    @staticmethod
    def load_checkpoint_raw(
        path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        weights_only: Optional[bool] = None,
    ) -> Any:
        """Ham checkpoint'i döndür (gelişmiş kurguya dışarıdan ihtiyaç olabilir)."""
        dev = _resolve_device(device)
        return _torch_load(path, map_location=dev, weights_only=weights_only)

    @staticmethod
    def load_all(
        model_class: Type[nn.Module],
        ckpt_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        weights_only: Optional[bool] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Tek çağrıda model + optimizer_state_dict + scheduler_state_dict + meta (epoch/config) döndürür.
        """
        dev = _resolve_device(device)
        loader_logger.info(f"Tüm checkpoint yükleniyor: {ckpt_path}")

        ckpt = _torch_load(ckpt_path, map_location=dev, weights_only=weights_only)
        model_sd, opt_sd, sch_sd, meta = _extract_state_dicts(ckpt)

        # Modeli kur ve yükle
        ctor_cfg = _with_aliases(config or {}, {"learning_rate": "lr", "n_heads": "num_heads"})
        if extra_model_kwargs:
            ctor_cfg.update(extra_model_kwargs)
        ctor_kwargs = _filter_kwargs_for_ctor(model_class, ctor_cfg)
        if "vocab_size" not in ctor_kwargs and config and "vocab_size" in config:
            ctor_kwargs["vocab_size"] = config["vocab_size"]

        model = model_class(**ctor_kwargs).to(dev)
        missing, unexpected = model.load_state_dict(model_sd, strict=strict)
        if missing or unexpected:
            loader_logger.warning(f"state_dict uyuşmazlıkları: missing={missing or []}, unexpected={unexpected or []}")

        loader_logger.info("Checkpoint yükleme tamamlandı.")
        return model, opt_sd, sch_sd, meta
