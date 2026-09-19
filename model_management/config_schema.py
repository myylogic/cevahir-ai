# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config_schema.py
Modül: model_management
Görev: Model yönetimi için tip-güvenli, doğrulanabilir yapılandırma şemaları.
       Düz Dict[str, Any] kullanımının yerine geçer; typo hatalarını derleme
       zamanında (IDE) ve çalışma zamanında (validate()) yakalar.

       ModelArchConfig  → Mimari parametreler (embed_dim, num_heads, ...)
       TrainingConfig   → Eğitim parametreleri (lr, batch_size, ...)
       CheckpointConfig → Checkpoint I/O ayarları
       DistributedConfig→ DDP / FSDP ayarları
       QuantConfig      → Quantization ayarları

KULLANIM:
    arch = ModelArchConfig(embed_dim=512, num_heads=8, vocab_size=60000)
    arch.validate()   # ValueError fırlatır; sorun varsa açıklar
    cfg_dict = arch.to_dict()
    arch2 = ModelArchConfig.from_dict(cfg_dict)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional
from copy import deepcopy
import warnings

CONFIG_VERSION = 1
ARCHITECTURE_VERSION = "cevahir-capabilities-1"
LEGACY_PROFILES = {"model_manager": {"num_layers": 12}}


# ══════════════════════════════════════════════════════════════════════════════
# Temel Yardımcılar
# ══════════════════════════════════════════════════════════════════════════════

class _SchemaBase:
    """Ortak to_dict / from_dict / validate arayüzü."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, strict: bool = False) -> "_SchemaBase":
        """
        Dict'ten örnek oluşturur; bilinmeyen anahtarlar sessizce atlanır
        (geriye dönük uyumluluk için).
        """
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(d) - known
        if unknown and strict:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)  # type: ignore[call-arg]

    def validate(self) -> None:
        """Geçersiz değerler için ValueError fırlatır."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 1. Model Mimari Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelArchConfig(_SchemaBase):
    """
    CevahirNeuralNetwork mimarisini tanımlayan parametreler.
    Tüm V-2/V-3/V-4/V-5 özellikleri tek çatı altında.
    """

    # ── Temel Boyutlar ────────────────────────────────────────────────────────
    embed_dim: int = 512
    """Gömme boyutu. Tipik: 256 (hızlı test) | 512 (standart) | 1024 (büyük)."""

    num_heads: int = 8
    """Dikkat başlığı sayısı. embed_dim % num_heads == 0 zorunlu."""

    num_layers: int = 8
    """Transformer katman sayısı."""

    ffn_dim: Optional[int] = None
    """FFN ara boyutu. None → 4 × embed_dim (endüstri standardı)."""

    vocab_size: int = 60000
    """Kelime hazinesi boyutu. TokenizerCore'dan alınır."""

    max_seq_length: int = 2048
    """Maksimum dizi uzunluğu."""

    dropout: float = 0.1
    """Dropout oranı [0, 1)."""

    # ── Normalizasyon & Aktivasyon ────────────────────────────────────────────
    pre_norm: bool = True
    """Pre-norm (GPT-2/3 tarzı) vs post-norm (BERT tarzı)."""

    use_rmsnorm: bool = True
    """RMSNorm: RMS tabanlı normalizasyon; hız ve stabilite ölçüm gerektirir."""

    use_swiglu: bool = True
    """SwiGLU aktivasyonu (LLaMA / PaLM standardı)."""

    # ── Attention Mekanizması ─────────────────────────────────────────────────
    causal_mask: bool = True
    """Autoregressive (GPT) eğitimi için causal masking."""

    use_flash_attention: bool = False
    """Opsiyonel harici Flash Attention; kullanılabilirlik donanım ve bağımlılığa bağlıdır."""

    num_kv_heads: Optional[int] = None
    """
    GQA (Grouped Query Attention) KV head sayısı.
    None → standart MHA (num_kv_heads = num_heads).
    num_heads=8 ve num_kv_heads=2 için KV tensörleri dörtte bir boyuttadır.
    """

    sliding_window: Optional[int] = None
    """
    Sliding Window Attention pencere boyutu.
    None → full attention. 512 / 2048 / 4096.
    """

    # ── Positional Encoding ───────────────────────────────────────────────────
    pe_mode: str = "rope"
    """Konum kodlaması: 'rope' | 'sinusoidal' | 'learned'."""

    rope_scaling_type: str = "none"
    """YaRN context uzatma: 'none' | 'yarn' | 'linear'."""

    rope_scaling_factor: float = 1.0
    """Uzatma faktörü. 2.0 = 2x, 4.0 = 4x (YaRN için)."""

    # ── KV Cache ─────────────────────────────────────────────────────────────
    use_kv_cache: bool = True
    """KV cache (inference hızlandırma; eğitimde kapalı)."""

    max_cache_len: int = 2048
    """Maksimum cache uzunluğu."""

    # ── Ağırlık Paylaşımı & Gradient Checkpointing ───────────────────────────
    tie_weights: bool = True
    """Input embedding ↔ output projection ağırlık paylaşımı."""

    use_gradient_checkpointing: bool = True
    """
    Gradient Checkpointing: aktivasyonları yeniden hesaplar.
    Bellek ve süre etkisi model, girdi ve donanıma göre ölçülmelidir.
    """

    # ── Mixture of Experts ───────────────────────────────────────────────────
    use_moe: bool = False
    """MoE FFN bloğu etkin mi? ."""

    num_experts: int = 8
    """MoE expert sayısı. use_moe=True iken etkin."""

    moe_top_k: int = 2
    """Her token için seçilecek expert sayısı."""

    # ── Quantization ─────────────────────────────────────────────────────────
    quantization_type: str = "none"
    """
    Quantization türü:
    'none'  → standart float16/float32
    'int8'  → bitsandbytes LLM.int8() (inference + eğitim)
    'int4'  → GPTQ/AWQ tarzı 4-bit (inference)
    """

    # ── Sequence Projection ──────────────────────────────────────────────────
    seq_proj_dim: Optional[int] = None
    """
    Output projeksiyon boyutu. None → embed_dim ile aynı.
    Eski uyumluluk alanı; farklıysa çekirdek weight tying seçimini kapatır.
    Katmanlar embed_dim kullanır; bağımsız projeksiyon oluşturulmaz.
    """

    # These capabilities already exist in the core; persist them in the schema.
    use_pytorch_sdpa: bool = True
    use_qk_norm: bool = False
    parallel_residual: bool = False
    logit_soft_cap: float = 30.0
    attn_logit_cap: float = 0.0
    drop_path_rate: float = 0.0
    use_advanced_checkpointing: bool = False
    checkpointing_strategy: str = "selective"
    pe_dropout: float = 0.0
    rope_original_max_len: int = 2048
    kv_eviction_strategy: str = "sliding_window"
    kv_num_sink_tokens: int = 4
    moe_jitter_noise: float = 0.01
    moe_load_balance_alpha: float = 0.01

    def validate(self) -> None:
        errors: List[str] = []

        if self.embed_dim <= 0:
            errors.append(f"embed_dim pozitif olmalı, gelen: {self.embed_dim}")
        if self.num_heads <= 0:
            errors.append(f"num_heads pozitif olmalı, gelen: {self.num_heads}")
        if self.num_heads > 0 and self.embed_dim % self.num_heads != 0:
            errors.append(
                f"embed_dim ({self.embed_dim}) % num_heads ({self.num_heads}) != 0; "
                f"head_dim = {self.embed_dim}/{self.num_heads} tam sayı olmalı"
            )
        if self.num_layers <= 0:
            errors.append(f"num_layers pozitif olmalı, gelen: {self.num_layers}")
        if self.vocab_size <= 0:
            errors.append(f"vocab_size pozitif olmalı, gelen: {self.vocab_size}")
        if not (0.0 <= self.dropout < 1.0):
            errors.append(f"dropout [0, 1) aralığında olmalı, gelen: {self.dropout}")
        if self.rope_scaling_factor < 1.0:
            errors.append(f"rope_scaling_factor >= 1.0 olmalı, gelen: {self.rope_scaling_factor}")
        if self.num_kv_heads is not None:
            if self.num_kv_heads <= 0:
                errors.append(f"num_kv_heads pozitif olmalı, gelen: {self.num_kv_heads}")
            if self.num_kv_heads > 0 and self.num_heads % self.num_kv_heads != 0:
                errors.append(
                    f"num_heads ({self.num_heads}) % num_kv_heads ({self.num_kv_heads}) != 0"
                )
        # seq_proj_dim is retained for legacy config/checkpoint roundtrips.
        # The current core projects directly from embed_dim; it does not use this
        # historical field to size either tied or untied output weights.
        if self.use_moe and (self.num_experts <= 0 or self.moe_top_k <= 0 or self.moe_top_k > self.num_experts):
            errors.append(
                f"moe_top_k ({self.moe_top_k}) > num_experts ({self.num_experts})"
            )
        if self.quantization_type not in ("none", "int8", "fp16", "bf16", "int8_dynamic"):
            errors.append(f"quantization_type geçersiz: {self.quantization_type!r}")
        if self.checkpointing_strategy not in ("selective", "layer_wise", "adaptive"):
            errors.append("Invalid checkpointing strategy")
        if self.pe_mode not in ("rope", "sinusoidal", "learned"):
            errors.append(f"pe_mode geçersiz: {self.pe_mode!r}")
        if self.rope_scaling_type not in ("none", "yarn", "linear"):
            errors.append(f"rope_scaling_type geçersiz: {self.rope_scaling_type!r}")
        if self.max_seq_length <= 0 or self.max_cache_len <= 0:
            errors.append("Sequence and cache lengths must be positive")
        if self.ffn_dim is not None and self.ffn_dim <= 0:
            errors.append("ffn_dim must be positive")
        if self.seq_proj_dim is not None and self.seq_proj_dim <= 0:
            errors.append("seq_proj_dim must be positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            errors.append("sliding_window must be positive or None")
        if not 0 <= self.drop_path_rate < 1 or not 0 <= self.pe_dropout < 1:
            errors.append("Drop probabilities must be in [0, 1)")
        if self.attn_logit_cap < 0 or self.logit_soft_cap < 0:
            errors.append("Logit caps must be non-negative")
        if self.kv_eviction_strategy not in ("none", "sliding_window"):
            errors.append("Invalid KV eviction strategy")
        if not 0 <= self.kv_num_sink_tokens < self.max_cache_len:
            errors.append("kv_num_sink_tokens must be in [0, max_cache_len)")
        if self.num_heads > 0 and self.pe_mode == "rope" and (self.embed_dim // self.num_heads) % 2:
            errors.append("RoPE requires an even head dimension")
        if self.moe_jitter_noise < 0 or self.moe_load_balance_alpha < 0:
            errors.append("MoE jitter and auxiliary loss coefficient must be non-negative")

        if errors:
            raise ValueError(
                f"ModelArchConfig doğrulama başarısız ({len(errors)} hata):\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    @property
    def head_dim(self) -> int:
        """Her dikkat başlığının boyutu."""
        return self.embed_dim // self.num_heads

    @property
    def effective_ffn_dim(self) -> int:
        """Decoder ile aynı etkin FFN genişliği; açık boyut korunur."""
        from src.neural_network_module.architecture_contracts import resolve_ffn_dim
        return resolve_ffn_dim(self.embed_dim, self.ffn_dim, gated=self.use_swiglu)

    @property
    def parameter_count_estimate(self) -> int:
        """
        Tahmini parametre sayısı (MoE hariç basit hesap).
        Gerçek sayı için ModelProfiler.count_parameters() kullanın.
        """
        V = self.vocab_size
        D = self.embed_dim
        L = self.num_layers
        F = self.effective_ffn_dim

        embedding = V * D
        kv_dim = (self.num_kv_heads or self.num_heads) * self.head_dim
        attention = L * (2 * D * D + 2 * D * kv_dim)
        ffn_mult = self.num_experts if self.use_moe else 1
        ffn = L * ((3 if self.use_swiglu else 2) * D * F) * ffn_mult
        router = L * D * self.num_experts if self.use_moe else 0
        norms = L * 2 * D               # RMSNorm/LayerNorm
        tied = self.tie_weights and self.seq_proj_dim in (None, D)
        output = 0 if tied else V * D
        return embedding + attention + ffn + norms + output + router

    def __repr__(self) -> str:
        return (
            f"ModelArchConfig("
            f"embed={self.embed_dim}, heads={self.num_heads}, "
            f"layers={self.num_layers}, vocab={self.vocab_size}, "
            f"moe={self.use_moe}, quant={self.quantization_type!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Eğitim Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig(_SchemaBase):
    """Eğitim sürecini kontrol eden parametreler."""

    learning_rate: float = 2e-4
    batch_size: int = 64
    grad_accum_steps: int = 4
    epochs: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    warmup_steps: int = 1500
    warmup_epochs: int = 1
    optimizer: str = "adamw"
    """'adamw' | 'adamw8bit' | 'adam' | 'radam' | 'sgd'"""

    scheduler_type: str = "reduce_on_plateau"
    lr_decay_factor: float = 0.75
    lr_decay_patience: int = 15
    lr_min: float = 1e-6
    seed: int = 42
    device: str = "cpu"
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    use_ema: bool = False
    ema_decay: float = 0.999
    precision: str = "auto"

    def validate(self) -> None:
        errors: List[str] = []
        if self.learning_rate <= 0:
            errors.append(f"learning_rate pozitif olmalı: {self.learning_rate}")
        if self.batch_size <= 0:
            errors.append(f"batch_size pozitif olmalı: {self.batch_size}")
        if self.grad_accum_steps <= 0:
            errors.append(f"grad_accum_steps pozitif olmalı: {self.grad_accum_steps}")
        if self.precision not in ("auto", "fp32", "fp16", "bf16"):
            errors.append(f"Unsupported precision: {self.precision}")
        if not (0.0 <= self.dropout < 1.0):
            errors.append(f"dropout [0,1) olmalı: {self.dropout}")
        if self.optimizer not in ("adamw", "adamw8bit", "adamw_8bit", "adam", "radam", "sgd", "rmsprop"):
            errors.append(f"optimizer geçersiz: {self.optimizer!r}")
        if errors:
            raise ValueError(
                f"TrainingConfig doğrulama başarısız:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    @property
    def effective_batch_size(self) -> int:
        """Gerçek batch boyutu = batch_size × grad_accum_steps."""
        return self.batch_size * self.grad_accum_steps


# ══════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointConfig(_SchemaBase):
    """Checkpoint kaydetme ve yükleme ayarları."""

    save_dir: str = "saved_models/checkpoints"
    model_save_path: str = "saved_models/cevahir_model.pth"
    filename_template: str = "checkpoint_ep{epoch:04d}.pth"
    keep_last_n: int = 5
    """Kaç eski checkpoint korunsun (0 = hepsini sakla)."""

    save_every_n_epochs: int = 10
    save_best_only: bool = False
    """Sadece en iyi val_loss checkpoint'i kaydet."""

    enable_sha256: bool = True
    """SHA-256 bütünlük doğrulaması."""

    enable_versioning: bool = True
    """Checkpoint meta verisine Cevahir sürüm bilgisi ekle."""

    compression: bool = False
    """gzip sıkıştırma (büyük checkpointler için yer tasarrufu)."""

    def validate(self) -> None:
        if self.keep_last_n < 0:
            raise ValueError(f"keep_last_n >= 0 olmalı: {self.keep_last_n}")
        if self.save_every_n_epochs <= 0:
            raise ValueError(f"save_every_n_epochs pozitif olmalı: {self.save_every_n_epochs}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dağıtık Eğitim Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedConfig(_SchemaBase):
    """DDP / FSDP dağıtık eğitim ayarları."""

    enabled: bool = False
    backend: str = "nccl"
    """'nccl' (GPU) | 'gloo' (CPU/mixed) | 'mpi'"""

    strategy: str = "ddp"
    """'ddp' | 'fsdp' | 'none'"""

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # FSDP özgü
    fsdp_sharding_strategy: str = "full_shard"
    """'full_shard' | 'shard_grad_op' | 'no_shard'"""

    fsdp_mixed_precision: bool = True

    def validate(self) -> None:
        if self.backend not in ("nccl", "gloo", "mpi"):
            raise ValueError(f"Geçersiz distributed backend: {self.backend!r}")
        if self.strategy not in ("ddp", "fsdp", "none"):
            raise ValueError(f"Geçersiz distributed strategy: {self.strategy!r}")
        if self.world_size < 1:
            raise ValueError(f"world_size >= 1 olmalı: {self.world_size}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Quantization Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantConfig(_SchemaBase):
    """Model quantization ayarları."""

    quant_type: str = "none"
    """'none' | 'int8' | 'int4'"""

    load_in_8bit: bool = False
    """bitsandbytes INT8 yükleme."""

    load_in_4bit: bool = False
    """bitsandbytes INT4 / GPTQ yükleme."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """4-bit hesaplama dtype'ı: 'float16' | 'bfloat16' | 'float32'"""

    bnb_4bit_quant_type: str = "nf4"
    """Quantization türü: 'nf4' (önerilen) | 'fp4'"""

    bnb_4bit_use_double_quant: bool = True
    """Double quantization → ek ~0.4 bit/parametre tasarrufu."""

    def validate(self) -> None:
        if self.quant_type not in ("none", "int8", "fp16", "bf16", "int8_dynamic"):
            raise ValueError(f"Geçersiz quant_type: {self.quant_type!r}")
        if self.load_in_8bit or self.load_in_4bit:
            raise ValueError("bitsandbytes loading flags are not integrated with the Cevahir core; use explicit apply_quantization().")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Birleşik Konfigürasyon (Kolaylık sınıfı)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CevahirConfig(_SchemaBase):
    """
    Tüm konfigürasyon bölümlerini tek çatı altında toplayan üst sınıf.

    Kullanım:
        cfg = CevahirConfig.from_flat_dict(TRAIN_CONFIG)
        cfg.arch.validate()
        cfg.training.validate()
    """
    arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    config_version: int = CONFIG_VERSION
    architecture_version: str = ARCHITECTURE_VERSION
    extras: Dict[str, Any] = field(default_factory=dict)

    def validate_all(self) -> None:
        """Tüm alt konfigürasyonları doğrular. İlk hata anında durur."""
        self.arch.validate()
        self.training.validate()
        self.checkpoint.validate()
        self.distributed.validate()
        self.quant.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "architecture_version": self.architecture_version,
            "arch": self.arch.to_dict(),
            "training": self.training.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "distributed": self.distributed.to_dict(),
            "quant": self.quant.to_dict(),
            "extras": deepcopy(self.extras),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, strict: bool = False) -> "CevahirConfig":
        _check_config_version(d)
        if "arch" not in d:
            return cls.from_flat_dict(d)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown and strict:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        return cls(
            arch=ModelArchConfig.from_dict(d.get("arch", {}), strict=strict),
            training=TrainingConfig.from_dict(d.get("training", {}), strict=strict),
            checkpoint=CheckpointConfig.from_dict(d.get("checkpoint", {}), strict=strict),
            distributed=DistributedConfig.from_dict(d.get("distributed", {}), strict=strict),
            quant=QuantConfig.from_dict(d.get("quant", {}), strict=strict),
            extras={**deepcopy(d.get("extras", {})), **{k: deepcopy(d[k]) for k in unknown}},
        )

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "CevahirConfig":
        """
        train.py'deki düz TRAIN_CONFIG sözlüğünden CevahirConfig oluşturur.
        Bilinmeyen anahtarlar atlanır.
        """
        d = normalize_model_config(d)
        schema_types = (ModelArchConfig, TrainingConfig, CheckpointConfig, DistributedConfig, QuantConfig)
        known = {f.name for t in schema_types for f in fields(t)} | {"config_version", "architecture_version"}
        distributed = DistributedConfig.from_dict(d)
        if "distributed_strategy" in d:
            distributed.strategy = d["distributed_strategy"]
            distributed.enabled = distributed.strategy != "none"
        return cls(
            arch=ModelArchConfig.from_dict(d),
            training=TrainingConfig.from_dict(d),
            checkpoint=CheckpointConfig.from_dict(d),
            distributed=distributed,
            quant=QuantConfig.from_dict({**d, "quant_type": d["quantization_type"]}),
            extras={k: deepcopy(v) for k, v in d.items() if k not in known},
        )

    def validate(self) -> None:
        self.validate_all()

    def __repr__(self) -> str:
        return (
            f"CevahirConfig(\n"
            f"  arch={self.arch!r},\n"
            f"  training={self.training!r},\n"
            f"  distributed={self.distributed!r},\n"
            f"  quant={self.quant!r}\n"
            f")"
        )


def _check_config_version(config: Dict[str, Any]) -> int:
    version = config.get("config_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version not in (0, CONFIG_VERSION):
        raise ValueError(f"Unsupported config_version {version!r}; supported versions: 0, {CONFIG_VERSION}")
    return version


def normalize_model_config(config: Optional[Dict[str, Any]] = None, *, strict: bool = False, legacy_profile: Optional[str] = None) -> Dict[str, Any]:
    """Migrate legacy flat/nested settings without mutating them or dropping extras.

    Explicit legacy dimensions win over defaults. Conflicting aliases raise rather
    than silently building a checkpoint-incompatible network. Version 0 is the
    unversioned legacy mapping; version 1 names the capability schema, not weights.
    """
    incoming = deepcopy(dict(config or {}))
    _check_config_version(incoming)
    data = deepcopy(incoming.get("extras", {}))
    if not isinstance(data, dict):
        raise ValueError("extras must be a mapping")
    for section in ("arch", "model", "training", "checkpoint"):
        if section in incoming:
            if not isinstance(incoming[section], dict):
                raise ValueError(f"{section} must be a mapping")
            for key, value in incoming[section].items():
                if key in data and data[key] != value:
                    raise ValueError(f"Conflicting configuration value: {key}")
                data[key] = value
    for key, value in incoming.items():
        if key not in ("arch", "model", "training", "checkpoint", "extras", "distributed", "quant", "compile"):
            if key in data and data[key] != value:
                raise ValueError(f"Conflicting flat/nested configuration value: {key}")
            data[key] = value
    aliases = {"d_model": "embed_dim", "n_heads": "num_heads", "n_layers": "num_layers", "ff_dim": "ffn_dim", "drop_rate": "dropout", "pe_max_len": "max_seq_length"}
    for old, new in aliases.items():
        if old in data:
            if new in data and data[new] != data[old]:
                raise ValueError(f"Conflicting aliases {old} and {new}")
            data[new] = data.pop(old)
    compile_cfg = incoming.get("compile", {})
    if compile_cfg:
        for key in ("enabled", "mode", "dynamic", "fullgraph"):
            if key in compile_cfg:
                target = "torch_compile" if key == "enabled" else f"torch_compile_{key}"
                if target in data and data[target] != compile_cfg[key]:
                    raise ValueError(f"Conflicting compile setting {key}")
                data[target] = compile_cfg[key]
    dist = incoming.get("distributed", {})
    if dist:
        strategy = dist.get("strategy", "none") if dist.get("enabled", False) else "none"
        if "distributed_strategy" in data and data["distributed_strategy"] != strategy:
            raise ValueError("Conflicting distributed strategy")
        data["distributed_strategy"] = strategy
        for key, value in dist.items():
            if key not in ("enabled", "strategy"):
                data["distributed_backend" if key == "backend" else key] = value
    quant = incoming.get("quant", {})
    if quant:
        for key, value in quant.items():
            target = "quantization_type" if key == "quant_type" else key
            if target in data and data[target] != value:
                raise ValueError(f"Conflicting quantization setting {key}")
            data[target] = value
    known = {f.name for t in (ModelArchConfig, TrainingConfig, CheckpointConfig, DistributedConfig, QuantConfig) for f in fields(t)}
    if "quant_type" in data:
        if "quantization_type" in data and data["quantization_type"] != data["quant_type"]:
            raise ValueError("Conflicting quantization type aliases")
        data["quantization_type"] = data.pop("quant_type")
    if data.get("load_in_8bit") or data.get("load_in_4bit"):
        raise ValueError("bitsandbytes loading flags are not integrated with the Cevahir core")
    known |= {"config_version", "architecture_version", "torch_compile", "torch_compile_mode", "torch_compile_dynamic", "torch_compile_fullgraph", "distributed_strategy", "distributed_backend", "log_level", "use_tensorboard", "attention_type", "normalization_type"}
    if strict and set(data) - known:
        raise ValueError(f"Unknown configuration fields: {sorted(set(data) - known)}")
    result = ModelArchConfig().to_dict()
    if legacy_profile is not None and incoming.get("config_version", 0) == 0:
        if legacy_profile not in LEGACY_PROFILES:
            raise ValueError(f"Unknown legacy profile: {legacy_profile}")
        result.update(LEGACY_PROFILES[legacy_profile])
    result.update(data)
    result["seq_proj_dim"] = result["embed_dim"] if result.get("seq_proj_dim") is None else result["seq_proj_dim"]
    result["pe_max_len"] = result["max_seq_length"]
    result.setdefault("learning_rate", TrainingConfig().learning_rate)
    result.setdefault("precision", TrainingConfig().precision)
    result["config_version"] = CONFIG_VERSION
    result["architecture_version"] = ARCHITECTURE_VERSION
    ModelArchConfig.from_dict(result).validate()
    return result


def tiny_model_config(**overrides: Any) -> Dict[str, Any]:
    """CPU-sized preset for contract tests; no language-quality claim."""
    return normalize_model_config({
        "vocab_size": 128, "embed_dim": 32, "num_heads": 4, "num_layers": 2,
        "ffn_dim": 64, "dropout": 0.0, "max_seq_length": 64, "max_cache_len": 64,
        "use_gradient_checkpointing": False, "device": "cpu", "log_level": 50,
        **overrides,
    })
