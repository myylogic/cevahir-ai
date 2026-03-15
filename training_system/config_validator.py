# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ — Training System V3
================================================================================
Dosya : training_system/config_validator.py
Modül : ConfigValidator
Görev : train.py TRAIN_CONFIG'ini eğitim başlamadan önce doğrular.

Kontroller:
  • Zorunlu alanların varlığı
  • Tür doğrulaması (int, float, bool, str)
  • Değer aralığı doğrulaması (lr > 0, dropout aralığı, vb.)
  • Tutarlılık kontrolleri (SWA start, scheduler uyumu, vb.)
  • Önerilen (best-practice) alanlar için uyarılar

Kullanım::

    from training_system.config_validator import ConfigValidator

    ConfigValidator.validate_and_raise(TRAIN_CONFIG)
    # veya:
    result = ConfigValidator().validate(TRAIN_CONFIG)
    result.print_report()

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Config doğrulama sonucu.

    Attributes:
        passed     : Kritik hata yoksa True.
        errors     : Eğitimi engelleyen kritik hatalar.
        warnings   : Dikkat edilmesi gereken uyarılar.
        suggestions: En iyi uygulamalar için öneriler.
    """
    passed:      bool
    errors:      List[str] = field(default_factory=list)
    warnings:    List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def print_report(self) -> None:
        """Renkli terminal raporu yazdırır."""
        _sep  = "═" * 70
        _line = "─" * 70

        status_icon = "✅ BAŞARILI" if self.passed else "❌ HATA VAR"
        print(f"\n{_sep}")
        print(f"  CONFIG DOĞRULAMA RAPORU — {status_icon}")
        print(_sep)

        if self.errors:
            print(f"\n  ❌ HATALAR ({len(self.errors)} adet):")
            print(_line)
            for i, err in enumerate(self.errors, 1):
                print(f"    {i}. {err}")

        if self.warnings:
            print(f"\n  ⚠️  UYARILAR ({len(self.warnings)} adet):")
            print(_line)
            for i, w in enumerate(self.warnings, 1):
                print(f"    {i}. {w}")

        if self.suggestions:
            print(f"\n  💡 ÖNERİLER ({len(self.suggestions)} adet):")
            print(_line)
            for i, s in enumerate(self.suggestions, 1):
                print(f"    {i}. {s}")

        if not self.errors and not self.warnings and not self.suggestions:
            print("\n  Tüm kontroller başarıyla geçti, sorun bulunamadı.")

        print(f"\n{_sep}\n")


# ---------------------------------------------------------------------------
# Tür kontrolü yardımcıları
# ---------------------------------------------------------------------------

_TYPE_CONSTRAINTS: Dict[str, Tuple[type, ...]] = {
    # Mimarı
    "embed_dim":          (int,),
    "seq_proj_dim":       (int,),
    "num_heads":          (int,),
    "num_layers":         (int,),
    "ffn_dim":            (int, type(None)),
    # Eğitim
    "epochs":             (int,),
    "batch_size":         (int,),
    "learning_rate":      (float, int),
    "dropout":            (float, int),
    "weight_decay":       (float, int),
    "grad_accum_steps":   (int,),
    "max_grad_norm":      (float, int),
    "gradient_clip":      (float, int),
    "warmup_epochs":      (int,),
    "warmup_steps":       (int,),
    "warmup_start_factor":(float, int),
    "early_stopping_patience": (int,),
    # Kayıp
    "label_smoothing":    (float, int),
    "entropy_coeff":      (float, int),
    "focal_gamma":        (float, int),
    "min_response_tokens":(int,),
    "aux_loss_weight":    (float, int),
    "eos_token_weight":   (float, int),
    "ignore_index":       (int,),
    # Scheduled sampling
    "min_teacher_forcing":(float, int),
    "ss_start_epoch":     (int,),
    "ss_end_epoch":       (int,),
    # SAM / Lookahead
    "sam_rho":            (float, int),
    "lookahead_k":        (int,),
    "lookahead_alpha":    (float, int),
    "llrd_decay":         (float, int),
    # EMA / SWA
    "ema_decay":          (float, int),
    "swa_start":          (int,),
    "swa_lr":             (float, int),
    # Curriculum
    "curriculum_epochs":  (int,),
    "curriculum_min_len": (int,),
    # Güvenlik
    "nan_tolerance":      (int,),
    "spike_n_sigma":      (float, int),
    "spike_window":       (int,),
    "agc_lambda":         (float, int),
    "gradient_noise_eta": (float, int),
    # İzleme
    "inference_probe_interval": (int,),
    "grad_health_log_every":    (int,),
    "token_dist_window":        (int,),
    # Booleans
    "use_focal_loss":      (bool,),
    "scheduled_sampling":  (bool,),
    "use_llrd":            (bool,),
    "use_sam":             (bool,),
    "use_lookahead":       (bool,),
    "use_ema":             (bool,),
    "use_swa":             (bool,),
    "use_curriculum":      (bool,),
    "gradient_noise":      (bool,),
    "track_token_distribution": (bool,),
    "use_amp":             (bool,),
    "tie_weights":         (bool,),
    "use_rmsnorm":         (bool,),
    "use_swiglu":          (bool,),
    "use_flash_attention": (bool,),
    "use_gradient_checkpointing": (bool,),
    "pre_norm":            (bool,),
    "causal_mask":         (bool,),
    "use_moe":             (bool,),
    # Scheduler
    "T_0":                 (int,),
    "T_mult":              (int, float),
    "lr_decay_factor":     (float, int),
    "lr_decay_patience":   (int,),
    "lr_threshold":        (float, int),
    "lr_min":              (float, int),
    # MoE
    "num_experts":         (int,),
    "moe_top_k":           (int,),
    # Strings
    "data_dir":            (str,),
    "model_save_path":     (str,),
    "vocab_path":          (str,),
    "merges_path":         (str,),
    "device":              (str,),
    "scheduler_type":      (str,),
    "optimizer_type":      (str,),
    "grad_clip_type":      (str,),
    "attention_type":      (str,),
    "pe_mode":             (str,),
}

_RANGE_CONSTRAINTS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # (min_val_exclusive_or_inclusive, max_val) — None = sınır yok
    "embed_dim":          (1, None),
    "seq_proj_dim":       (1, None),
    "num_heads":          (1, None),
    "num_layers":         (1, None),
    "epochs":             (1, None),
    "batch_size":         (1, None),
    "learning_rate":      (0.0, None),       # > 0
    "dropout":            (0.0, 1.0),        # 0 < x < 1
    "weight_decay":       (0.0, None),
    "grad_accum_steps":   (1, None),
    "label_smoothing":    (0.0, 1.0),
    "entropy_coeff":      (0.0, None),
    "focal_gamma":        (0.0, None),
    "min_response_tokens":(0, None),
    "aux_loss_weight":    (0.0, None),
    "eos_token_weight":   (0.0, None),
    "min_teacher_forcing":(0.0, 1.0),
    "sam_rho":            (0.0, None),
    "lookahead_k":        (1, None),
    "lookahead_alpha":    (0.0, 1.0),
    "llrd_decay":         (0.0, 1.0),
    "ema_decay":          (0.0, 1.0),
    "swa_start":          (0, None),
    "swa_lr":             (0.0, None),
    "curriculum_epochs":  (0, None),
    "nan_tolerance":      (0, None),
    "spike_n_sigma":      (0.0, None),
    "spike_window":       (1, None),
    "agc_lambda":         (0.0, None),
    "inference_probe_interval": (1, None),
    "grad_health_log_every":    (1, None),
    "T_0":                (1, None),
    "num_experts":        (1, None),
    "moe_top_k":          (1, None),
    "warmup_start_factor":(0.0, 1.0),
    "max_grad_norm":      (0.0, None),
    "gradient_clip":      (0.0, None),
}


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """
    TRAIN_CONFIG doğrulayıcısı.

    Tüm kontroller stateless'tır; aynı validator örneği tekrar tekrar kullanılabilir.
    """

    REQUIRED_FIELDS: List[str] = [
        "embed_dim",
        "seq_proj_dim",
        "num_heads",
        "num_layers",
        "epochs",
        "batch_size",
        "learning_rate",
        "data_dir",
        "model_save_path",
        "vocab_path",
        "merges_path",
    ]

    # Öneri eşikleri
    _SUGGESTED_FIELDS: List[str] = [
        "weight_decay",
        "dropout",
        "label_smoothing",
        "warmup_epochs",
        "early_stopping_patience",
        "checkpoint_dir",
        "training_history_path",
    ]

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Config'i kapsamlı biçimde doğrular.

        Args:
            config: TRAIN_CONFIG sözlüğü.

        Returns:
            ValidationResult — hata, uyarı ve önerileri içerir.
        """
        errors:      List[str] = []
        warnings:    List[str] = []
        suggestions: List[str] = []

        # 1. Zorunlu alan varlık kontrolü
        errors.extend(self._check_required(config))

        # 2. Tür kontrolleri
        errors.extend(self._check_types(config))

        # 3. Değer aralığı kontrolleri
        errors.extend(self._check_ranges(config))

        # 4. Tutarlılık kontrolleri
        cons_errors, cons_warnings = self._check_consistency(config)
        errors.extend(cons_errors)
        warnings.extend(cons_warnings)

        # 5. Best-practice önerileri
        suggestions.extend(self._check_best_practices(config))

        passed = len(errors) == 0
        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # 1. Zorunlu alan kontrolü
    # ------------------------------------------------------------------

    def _check_required(self, config: Dict[str, Any]) -> List[str]:
        """Zorunlu alanların config'te bulunup bulunmadığını kontrol eder."""
        errors: List[str] = []
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in config:
                errors.append(
                    f"Zorunlu alan eksik: '{field_name}'"
                )
            elif config[field_name] is None:
                errors.append(
                    f"Zorunlu alan None olamaz: '{field_name}'"
                )
        return errors

    # ------------------------------------------------------------------
    # 2. Tür kontrolleri
    # ------------------------------------------------------------------

    def _check_types(self, config: Dict[str, Any]) -> List[str]:
        """Belirtilen alanların beklenen tipte olduğunu kontrol eder."""
        errors: List[str] = []
        for key, expected_types in _TYPE_CONSTRAINTS.items():
            if key not in config:
                continue  # Opsiyonel alan — varlık zorunlu değil
            val = config[key]
            if val is None and type(None) in expected_types:
                continue
            if not isinstance(val, expected_types):
                # bool, int'in alt sınıfıdır — bool'u int olarak kabul etme
                if bool in expected_types and isinstance(val, bool):
                    continue
                if int in expected_types and isinstance(val, bool):
                    errors.append(
                        f"Tür hatası: '{key}' bool olmamalı, int bekleniyor."
                        f" Mevcut: {type(val).__name__}"
                    )
                    continue
                type_names = " | ".join(t.__name__ for t in expected_types if t is not type(None))
                errors.append(
                    f"Tür hatası: '{key}' → beklenen {type_names},"
                    f" bulunan {type(val).__name__} (değer: {val!r})"
                )
        return errors

    # ------------------------------------------------------------------
    # 3. Değer aralığı kontrolleri
    # ------------------------------------------------------------------

    def _check_ranges(self, config: Dict[str, Any]) -> List[str]:
        """Sayısal alanların geçerli aralıkta olduğunu kontrol eder."""
        errors: List[str] = []
        for key, (min_val, max_val) in _RANGE_CONSTRAINTS.items():
            if key not in config or config[key] is None:
                continue
            val = config[key]
            if not isinstance(val, (int, float)):
                continue
            if min_val is not None and val <= min_val:
                # Özel durum: 0.0 min değer, tam olarak sıfır geçmez
                errors.append(
                    f"Aralık hatası: '{key}' = {val} → {key} > {min_val} olmalı"
                )
            if max_val is not None and val > max_val:
                errors.append(
                    f"Aralık hatası: '{key}' = {val} → {key} <= {max_val} olmalı"
                )

        # learning_rate için özel — sıfır dahil kontrol
        lr = config.get("learning_rate")
        if lr is not None and isinstance(lr, (int, float)) and lr <= 0:
            errors.append(
                f"Değer hatası: 'learning_rate' = {lr} → pozitif olmalı"
            )

        return errors

    # ------------------------------------------------------------------
    # 4. Tutarlılık kontrolleri
    # ------------------------------------------------------------------

    def _check_consistency(
        self, config: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Config anahtarları arasındaki tutarlılığı kontrol eder.

        Returns:
            (errors, warnings) tuple'ı.
        """
        errors:   List[str] = []
        warnings: List[str] = []
        epochs = config.get("epochs", 1)

        # --- SWA: swa_start >= epochs ---
        if config.get("use_swa", False):
            swa_start = config.get("swa_start", 0)
            if isinstance(swa_start, int) and isinstance(epochs, int):
                if swa_start >= epochs:
                    errors.append(
                        f"Tutarsızlık: use_swa=True ancak swa_start={swa_start} >= epochs={epochs}."
                        f" SWA hiçbir zaman devreye girmeyecek."
                    )

        # --- LLRD: optimizer_type != adamw ---
        if config.get("use_llrd", False):
            opt_type = str(config.get("optimizer_type", "adamw")).lower()
            if opt_type not in ("adamw", "adam_w"):
                warnings.append(
                    f"Uyarı: use_llrd=True ancak optimizer_type='{opt_type}'."
                    f" LLRD AdamW ile tasarlanmıştır."
                )

        # --- Scheduled Sampling: ss_end_epoch > epochs ---
        if config.get("scheduled_sampling", False):
            ss_end = config.get("ss_end_epoch", 0)
            if isinstance(ss_end, int) and isinstance(epochs, int):
                if ss_end > epochs:
                    errors.append(
                        f"Tutarsızlık: scheduled_sampling=True ancak ss_end_epoch={ss_end} > epochs={epochs}."
                        f" Scheduled sampling tamamlanamayacak."
                    )
            ss_start = config.get("ss_start_epoch", 0)
            if isinstance(ss_start, int) and isinstance(ss_end, int):
                if ss_start >= ss_end:
                    errors.append(
                        f"Tutarsızlık: ss_start_epoch={ss_start} >= ss_end_epoch={ss_end}."
                        f" Scheduled sampling aralığı geçersiz."
                    )

        # --- Tie weights: embed_dim != seq_proj_dim ---
        if config.get("tie_weights", False):
            embed   = config.get("embed_dim")
            seq_proj = config.get("seq_proj_dim")
            if (embed is not None and seq_proj is not None
                    and isinstance(embed, int) and isinstance(seq_proj, int)
                    and embed != seq_proj):
                errors.append(
                    f"Tutarsızlık: tie_weights=True ancak embed_dim={embed} != seq_proj_dim={seq_proj}."
                    f" Ağırlık paylaşımı için bu iki değer eşit olmalı."
                )

        # --- grad_accum_steps > batch_size ---
        grad_accum = config.get("grad_accum_steps", 1)
        batch_size = config.get("batch_size", 1)
        if (isinstance(grad_accum, int) and isinstance(batch_size, int)
                and grad_accum > batch_size):
            warnings.append(
                f"Uyarı: grad_accum_steps={grad_accum} > batch_size={batch_size}."
                f" Bu genellikle istenmez; grad_accum_steps küçük batch boyutunda"
                f" kullanılır, büyük batch'te etkin batch zaten yeterince büyük."
            )

        # --- use_amp=True + device="cpu" ---
        if config.get("use_amp", False):
            device = str(config.get("device", "cuda")).lower()
            if "cpu" in device:
                warnings.append(
                    "Uyarı: use_amp=True ancak device='cpu'."
                    " CPU'da mixed precision desteklenmez (PyTorch 2.x'te kısmi destek var)."
                    " GPU olmadan AMP devre dışı kalacak."
                )

        # --- label_smoothing=0.0 (entropy collapse riski) ---
        ls = config.get("label_smoothing", 0.0)
        if isinstance(ls, (int, float)) and float(ls) == 0.0:
            # Bu hata değil, uyarı
            warnings.append(
                "Uyarı: label_smoothing=0.0 — entropy collapse riski var."
                " V3 için label_smoothing=0.1 önerilir (Szegedy et al. 2016)."
            )

        # --- MoE: moe_top_k >= num_experts ---
        if config.get("use_moe", False):
            num_exp = config.get("num_experts", 1)
            top_k   = config.get("moe_top_k", 1)
            if (isinstance(num_exp, int) and isinstance(top_k, int)
                    and top_k >= num_exp):
                errors.append(
                    f"Tutarsızlık: moe_top_k={top_k} >= num_experts={num_exp}."
                    f" Top-k, expert sayısından küçük olmalı."
                )

        # --- num_heads embed_dim'i bölemiyor ---
        embed = config.get("embed_dim")
        heads = config.get("num_heads")
        if (embed is not None and heads is not None
                and isinstance(embed, int) and isinstance(heads, int)
                and heads > 0 and embed % heads != 0):
            errors.append(
                f"Tutarsızlık: embed_dim={embed} num_heads={heads} tarafından bölünemiyor."
                f" head_dim = {embed}/{heads} tam sayı olmalı."
            )

        # --- Cosine restarts scheduler: T_0 kontrolü ---
        if str(config.get("scheduler_type", "")).lower() in (
            "cosine_restarts", "cosine_annealing_warm_restarts"
        ):
            T_0 = config.get("T_0", 1)
            if isinstance(T_0, int) and T_0 <= 0:
                errors.append(
                    f"Tutarsızlık: scheduler_type=cosine_restarts ancak T_0={T_0} <= 0."
                )

        # --- SAM + Lookahead kombinasyonu uyarısı ---
        if config.get("use_sam", False) and config.get("use_lookahead", False):
            warnings.append(
                "Uyarı: use_sam=True ve use_lookahead=True aynı anda aktif."
                " SAM+Lookahead kombinasyonu deneyseldir ve 4x yavaşlamaya yol açabilir."
            )

        # --- Gradient noise + SAM ---
        if config.get("gradient_noise", False) and config.get("use_sam", False):
            warnings.append(
                "Uyarı: gradient_noise=True ve use_sam=True aynı anda aktif."
                " SAM zaten pertürbasyon ekliyor; ekstra gürültü gereksiz olabilir."
            )

        # --- Curriculum end epoch ---
        if config.get("use_curriculum", False):
            curr_epochs = config.get("curriculum_epochs", 0)
            if isinstance(curr_epochs, int) and isinstance(epochs, int):
                if curr_epochs >= epochs:
                    warnings.append(
                        f"Uyarı: curriculum_epochs={curr_epochs} >= epochs={epochs}."
                        f" Curriculum tüm eğitimi kapsıyor, normal veri asla kullanılmayacak."
                    )

        return errors, warnings

    # ------------------------------------------------------------------
    # 5. Best-practice önerileri
    # ------------------------------------------------------------------

    def _check_best_practices(self, config: Dict[str, Any]) -> List[str]:
        """Olmayan ama önerilen alanlar için öneriler üretir."""
        suggestions: List[str] = []

        for field_name in self._SUGGESTED_FIELDS:
            if field_name not in config:
                suggestions.append(
                    f"'{field_name}' alanı config'te yok. Eklenmesi önerilir."
                )

        # EMA önerisi
        if not config.get("use_ema", False):
            suggestions.append(
                "use_ema=False — EMA ağırlıkları genellikle daha iyi genelleme sağlar."
                " use_ema=True, ema_decay=0.9999 ile denenebilir."
            )

        # Inference probe önerisi
        if "inference_probe_interval" not in config:
            suggestions.append(
                "'inference_probe_interval' ayarlanmamış."
                " Üretim kalitesini izlemek için inference_probe_interval=5 önerilir."
            )

        # Flash attention önerisi
        if not config.get("use_flash_attention", False):
            suggestions.append(
                "use_flash_attention=False — GPU varsa flash attention 2-3x hız sağlar."
                " use_flash_attention=True denenebilir."
            )

        # RoPE önerisi
        pe_mode = str(config.get("pe_mode", "sinusoidal")).lower()
        if pe_mode == "sinusoidal":
            suggestions.append(
                "pe_mode='sinusoidal' — Modern modeller RoPE kullanır."
                " pe_mode='rope' denenebilir (daha iyi uzun dizilerde genelleme)."
            )

        # Gradient checkpointing büyük model için
        num_layers = config.get("num_layers", 0)
        if isinstance(num_layers, int) and num_layers >= 12:
            if not config.get("use_gradient_checkpointing", False):
                suggestions.append(
                    f"num_layers={num_layers} için use_gradient_checkpointing=True önerilir"
                    " (bellek tasarrufu)."
                )

        return suggestions

    # ------------------------------------------------------------------
    # Class method — hata varsa raise et
    # ------------------------------------------------------------------

    @classmethod
    def validate_and_raise(cls, config: Dict[str, Any]) -> ValidationResult:
        """
        Config'i doğrular; hata varsa ValueError fırlatır, uyarıları loglar.

        Args:
            config: TRAIN_CONFIG sözlüğü.

        Returns:
            ValidationResult (sadece hata yoksa döner).

        Raises:
            ValueError: Kritik hata bulunursa.
        """
        validator = cls()
        result = validator.validate(config)

        # Uyarı ve önerileri logla
        for w in result.warnings:
            logger.warning("[ConfigValidator] %s", w)
        for s in result.suggestions:
            logger.info("[ConfigValidator] ÖNERI: %s", s)

        if not result.passed:
            result.print_report()
            error_summary = "\n".join(f"  - {e}" for e in result.errors)
            raise ValueError(
                f"[ConfigValidator] Config doğrulama başarısız "
                f"({len(result.errors)} hata):\n{error_summary}"
            )

        logger.info(
            "[ConfigValidator] Config geçerli — %d uyarı, %d öneri.",
            len(result.warnings),
            len(result.suggestions),
        )
        return result
