"""
checkpoint_verifier.py
=======================
Cevahir Türkçe Dil Modeli - Checkpoint Doğrulama Sistemi

Kaydedilen checkpoint'lerin bütünlüğünü ve kullanılabilirliğini doğrular.
Bozuk checkpoint'lerin production'a geçmesini önler.

Doğrulama Adımları:
    1. Checkpoint'i yeniden yükle (dosya okunabilir mi?)
    2. State dict key'lerini modelle karşılaştır
    3. Tensor shape'lerini kontrol et
    4. Dummy input ile model forward pass testi yap
    5. Başarısız → uyarı log'u + False döndür

Atomic save sonrası çağrılması önerilir.
"""

from __future__ import annotations

import datetime
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointVerifier:
    """
    Checkpoint kayıt sonrası doğrulama sistemi.

    Atomic save sonrası checkpoint'in geçerli, eksiksiz ve
    çalışabilir olduğunu doğrular.

    Doğrulama Adımları:
        1. Checkpoint dosyasını yükle
        2. State dict key'lerini model ile karşılaştır
        3. Tensor shape'lerini doğrula
        4. Dummy forward pass ile modelin çalıştığını teyit et
        5. Hata varsa ayrıntılı rapor oluştur

    Args:
        device: Doğrulama sırasında kullanılacak cihaz ('cpu' veya 'cuda').
                Forward pass testleri bu cihazda çalışır.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._last_report: Dict[str, Any] = {}

        logger.info("CheckpointVerifier başlatıldı | device=%s", device)

    # ------------------------------------------------------------------
    # Herkese Açık API
    # ------------------------------------------------------------------

    def verify(
        self,
        checkpoint_path: str,
        model: nn.Module,
        vocab_size: Optional[int] = None,
    ) -> bool:
        """
        Verilen checkpoint dosyasını kapsamlı şekilde doğrular.

        Adım Adım:
            1. Dosyanın varlığı ve okunabilirliği kontrol edilir.
            2. Checkpoint yüklenir (torch.load).
            3. State dict key'leri modelle karşılaştırılır.
            4. Tensor shape'leri doğrulanır.
            5. Dummy forward pass çalıştırılır (vocab_size verilmişse).

        Args:
            checkpoint_path: Doğrulanacak .pt / .pth dosyasının yolu.
            model:           Referans PyTorch modeli.
            vocab_size:      Forward pass testi için vocab boyutu.
                             None ise forward pass atlanır.

        Returns:
            True  → checkpoint geçerli ve kullanılabilir.
            False → checkpoint bozuk veya eksik.
        """
        verified_at = datetime.datetime.now().isoformat()
        error_details: List[str] = []
        passed = False

        try:
            # Adım 1: Dosya varlığı kontrolü
            if not os.path.exists(checkpoint_path):
                msg = f"Checkpoint dosyası bulunamadı: {checkpoint_path}"
                logger.error(msg)
                error_details.append(msg)
                self._save_report(checkpoint_path, False, error_details, verified_at)
                return False

            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            logger.info(
                "Checkpoint doğrulaması başlıyor | path=%s | size=%.2f MB",
                checkpoint_path,
                file_size_mb,
            )

            # Adım 2 & 3: Yükleme + key kontrolü
            keys_ok, key_errors, state_dict = self._load_and_check_keys(
                checkpoint_path, model
            )
            if not keys_ok:
                error_details.extend(key_errors)
                logger.error("Key doğrulaması başarısız: %s", key_errors)
                self._save_report(checkpoint_path, False, error_details, verified_at)
                return False

            # Adım 4: Shape kontrolü
            if state_dict is not None:
                shapes_ok, shape_errors = self._check_shapes(state_dict, model)
                if not shapes_ok:
                    error_details.extend(shape_errors)
                    logger.error("Shape doğrulaması başarısız: %s", shape_errors)
                    self._save_report(checkpoint_path, False, error_details, verified_at)
                    return False

            # Adım 5: Forward pass (isteğe bağlı)
            if vocab_size is not None:
                # State dict'i modele yükle (geçici)
                original_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                try:
                    if state_dict is not None:
                        model.load_state_dict(state_dict, strict=True)
                    forward_ok, forward_error = self._forward_pass_test(
                        model, vocab_size
                    )
                finally:
                    # Orijinal state'i geri yükle
                    model.load_state_dict(original_state, strict=True)

                if not forward_ok:
                    error_details.append(forward_error or "Forward pass başarısız.")
                    logger.error("Forward pass testi başarısız: %s", forward_error)
                    self._save_report(checkpoint_path, False, error_details, verified_at)
                    return False

            passed = True
            logger.info(
                "Checkpoint doğrulaması BASARILI | path=%s", checkpoint_path
            )

        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"Beklenmedik doğrulama hatası: {exc}"
            logger.error("%s\n%s", msg, tb)
            error_details.append(msg)
            error_details.append(tb)
            passed = False

        finally:
            self._save_report(checkpoint_path, passed, error_details, verified_at)

        return passed

    def get_verification_report(self) -> Dict[str, Any]:
        """
        En son doğrulama işleminin raporunu döndürür.

        Returns:
            Dict içeriği:
                path          : Doğrulanan checkpoint yolu
                passed        : Doğrulama başarılı mı?
                error_details : Hata mesajları listesi
                verified_at   : Doğrulama zaman damgası (ISO 8601)
                file_size_mb  : Dosya boyutu (MB), bilinmiyorsa None
        """
        return dict(self._last_report)

    # ------------------------------------------------------------------
    # Dahili Doğrulama Adımları
    # ------------------------------------------------------------------

    def _load_and_check_keys(
        self, path: str, model: nn.Module
    ) -> tuple[bool, List[str], Optional[Dict[str, torch.Tensor]]]:
        """
        Checkpoint'i yükler ve state dict key'lerini model ile karşılaştırır.

        Kontrol Edilen Durumlar:
            - Eksik key (modelde var ama checkpoint'te yok)
            - Fazladan key (checkpoint'te var ama modelde yok)
            - Boş state dict

        Args:
            path:  Checkpoint dosyası yolu.
            model: Referans model.

        Returns:
            (başarılı, hata_listesi, state_dict) tuple'ı.
        """
        errors: List[str] = []
        state_dict: Optional[Dict[str, torch.Tensor]] = None

        try:
            # map_location ile CPU'ya yükle (GPU bağımsız)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            errors.append(f"Checkpoint yüklenemedi: {exc}")
            return False, errors, None

        # Checkpoint formatını belirle
        if isinstance(checkpoint, dict):
            # Checkpoint hem model state'i hem de metadata içerebilir
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Doğrudan state dict olabilir
                state_dict = checkpoint
        else:
            errors.append(
                f"Geçersiz checkpoint formatı: {type(checkpoint).__name__}. "
                "Dict bekleniyor."
            )
            return False, errors, None

        if not state_dict:
            errors.append("Checkpoint state dict boş.")
            return False, errors, None

        # Model ve checkpoint key setlerini karşılaştır
        model_keys: Set[str] = set(model.state_dict().keys())
        ckpt_keys: Set[str] = set(state_dict.keys())

        missing_in_ckpt = model_keys - ckpt_keys
        extra_in_ckpt = ckpt_keys - model_keys

        if missing_in_ckpt:
            errors.append(
                f"Checkpoint'te eksik key'ler ({len(missing_in_ckpt)} adet): "
                f"{sorted(missing_in_ckpt)[:10]}"
                f"{'...' if len(missing_in_ckpt) > 10 else ''}"
            )

        if extra_in_ckpt:
            # Fazla key uyarı seviyesinde loglansın (hata sayılmaz)
            logger.warning(
                "Checkpoint'te fazla key'ler (%d adet): %s",
                len(extra_in_ckpt),
                sorted(extra_in_ckpt)[:10],
            )

        if errors:
            return False, errors, state_dict

        logger.debug(
            "Key doğrulaması başarılı | checkpoint_keys=%d | model_keys=%d",
            len(ckpt_keys),
            len(model_keys),
        )
        return True, [], state_dict

    def _check_shapes(
        self, state_dict: Dict[str, torch.Tensor], model: nn.Module
    ) -> tuple[bool, List[str]]:
        """
        Checkpoint tensörlerinin model parametreleriyle aynı shape'e sahip
        olup olmadığını kontrol eder.

        Args:
            state_dict: Yüklenen checkpoint state dict'i.
            model:      Referans model.

        Returns:
            (başarılı, hata_listesi) tuple'ı.
        """
        errors: List[str] = []
        model_state = model.state_dict()
        mismatched: List[str] = []

        for key, ckpt_tensor in state_dict.items():
            if key not in model_state:
                continue  # Fazla key zaten _load_and_check_keys'te loglandı

            model_tensor = model_state[key]

            if not isinstance(ckpt_tensor, torch.Tensor):
                errors.append(
                    f"Key '{key}': Tensor olmayan değer: {type(ckpt_tensor).__name__}"
                )
                continue

            if ckpt_tensor.shape != model_tensor.shape:
                mismatch_msg = (
                    f"Key '{key}': "
                    f"checkpoint shape={tuple(ckpt_tensor.shape)} != "
                    f"model shape={tuple(model_tensor.shape)}"
                )
                mismatched.append(mismatch_msg)
                logger.error("Shape uyuşmazlığı: %s", mismatch_msg)

        if mismatched:
            errors.append(
                f"Shape uyuşmazlıkları ({len(mismatched)} adet): "
                + " | ".join(mismatched[:5])
                + ("..." if len(mismatched) > 5 else "")
            )
            return False, errors

        logger.debug(
            "Shape doğrulaması başarılı | kontrol edilen parametre sayısı=%d",
            len(state_dict),
        )
        return True, []

    def _forward_pass_test(
        self, model: nn.Module, vocab_size: int
    ) -> tuple[bool, Optional[str]]:
        """
        Modeli doğrulama cihazına taşır ve dummy bir input ile forward pass
        yapar. Modelin hata vermeden çalışıp çalışmadığını test eder.

        Args:
            model:      Test edilecek model (state dict yüklenmiş olmalı).
            vocab_size: Input token ID'lerinin üst sınırı (exclusive).

        Returns:
            (başarılı, hata_mesajı) tuple'ı.
        """
        original_training = model.training
        try:
            model.eval()
            model.to(self.device)

            # Dummy input: küçük bir sekans (batch_size=1, seq_len=8)
            dummy_input = torch.randint(
                low=0,
                high=max(1, vocab_size),
                size=(1, 8),
                device=self.device,
                dtype=torch.long,
            )

            with torch.no_grad():
                output = model(dummy_input)

            # Output'un geçerli bir tensör veya tuple olup olmadığını kontrol et
            if output is None:
                return False, "Model forward pass None döndürdü."

            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    return False, "Model çıktısında NaN değerleri var."
                if torch.isinf(output).any():
                    return False, "Model çıktısında Inf değerleri var."

            logger.debug(
                "Forward pass testi başarılı | input_shape=(1, 8) | device=%s",
                self.device,
            )
            return True, None

        except Exception as exc:
            tb = traceback.format_exc()
            error_msg = f"Forward pass hatası: {exc}"
            logger.error("%s\n%s", error_msg, tb)
            return False, error_msg

        finally:
            # Model'i orijinal moduna geri al
            if original_training:
                model.train()
            else:
                model.eval()

    # ------------------------------------------------------------------
    # Dahili Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _save_report(
        self,
        path: str,
        passed: bool,
        error_details: List[str],
        verified_at: str,
    ) -> None:
        """Son doğrulama raporunu dahili alana kaydeder."""
        file_size_mb: Optional[float] = None
        if os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
            except OSError:
                pass

        self._last_report = {
            "path": path,
            "passed": passed,
            "error_details": error_details,
            "verified_at": verified_at,
            "file_size_mb": file_size_mb,
        }

    def __repr__(self) -> str:
        last_path = self._last_report.get("path", "N/A")
        last_passed = self._last_report.get("passed", None)
        return (
            f"CheckpointVerifier("
            f"device='{self.device}', "
            f"last_path='{last_path}', "
            f"last_passed={last_passed})"
        )
