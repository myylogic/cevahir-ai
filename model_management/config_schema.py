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

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Temel Yardımcılar
# ══════════════════════════════════════════════════════════════════════════════

class _SchemaBase:
    """Ortak to_dict / from_dict / validate arayüzü."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_SchemaBase":
        """
        Dict'ten örnek oluşturur; bilinmeyen anahtarlar sessizce atlanır
        (geriye dönük uyumluluk için).
        """
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
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
    """RMSNorm: LayerNorm'a göre ~%10 hızlı, stabilite benzer."""

    use_swiglu: bool = True
    """SwiGLU aktivasyonu (LLaMA / PaLM standardı)."""

    # ── Attention Mekanizması ─────────────────────────────────────────────────
    causal_mask: bool = True
    """Autoregressive (GPT) eğitimi için causal masking."""

    use_flash_attention: bool = True
    """Flash Attention 2.0 — bellek O(N) ve 2-3x hız."""

    num_kv_heads: Optional[int] = None
    """
    GQA (Grouped Query Attention) KV head sayısı.
    None → standart MHA (num_kv_heads = num_heads).
    2    → %75 KV cache azalması (LLaMA-2/3 standardı).
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
    ~%30-40 bellek tasarrufu, ~%20-30 yavaşlama.
    """

    # ── Mixture of Experts ───────────────────────────────────────────────────
    use_moe: bool = False
    """MoE FFN bloğu etkin mi? (GPT-4 / Mixtral standardı)."""

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
    tie_weights=True için seq_proj_dim == embed_dim zorunlu.
    """

    def validate(self) -> None:
        errors: List[str] = []

        if self.embed_dim <= 0:
            errors.append(f"embed_dim pozitif olmalı, gelen: {self.embed_dim}")
        if self.num_heads <= 0:
            errors.append(f"num_heads pozitif olmalı, gelen: {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
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
            if self.num_heads % self.num_kv_heads != 0:
                errors.append(
                    f"num_heads ({self.num_heads}) % num_kv_heads ({self.num_kv_heads}) != 0"
                )
        if self.tie_weights:
            seq_proj = self.seq_proj_dim if self.seq_proj_dim is not None else self.embed_dim
            if seq_proj != self.embed_dim:
                errors.append(
                    f"tie_weights=True gerektirir seq_proj_dim == embed_dim "
                    f"({seq_proj} != {self.embed_dim})"
                )
        if self.use_moe and self.moe_top_k > self.num_experts:
            errors.append(
                f"moe_top_k ({self.moe_top_k}) > num_experts ({self.num_experts})"
            )
        if self.quantization_type not in ("none", "int8", "int4"):
            errors.append(f"quantization_type geçersiz: {self.quantization_type!r}")
        if self.pe_mode not in ("rope", "sinusoidal", "learned"):
            errors.append(f"pe_mode geçersiz: {self.pe_mode!r}")
        if self.rope_scaling_type not in ("none", "yarn", "linear"):
            errors.append(f"rope_scaling_type geçersiz: {self.rope_scaling_type!r}")

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
        """Gerçek FFN ara boyutu (None ise 4×embed_dim)."""
        return self.ffn_dim if self.ffn_dim is not None else 4 * self.embed_dim

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
        attention = L * (4 * D * D)     # Q, K, V, O projeksiyon
        ffn_mult = self.num_experts if self.use_moe else 1
        ffn = L * (2 * D * F) * ffn_mult  # FFN (SwiGLU için 2 matris)
        norms = L * 2 * D               # RMSNorm/LayerNorm
        output = 0 if self.tie_weights else V * D
        return embedding + attention + ffn + norms + output

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
    optimizer: str = "adamw8bit"
    """'adamw' | 'adamw8bit' | 'adam' | 'radam' | 'sgd'"""

    scheduler_type: str = "reduce_on_plateau"
    lr_decay_factor: float = 0.75
    lr_decay_patience: int = 15
    lr_min: float = 1e-6
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    def validate(self) -> None:
        errors: List[str] = []
        if self.learning_rate <= 0:
            errors.append(f"learning_rate pozitif olmalı: {self.learning_rate}")
        if self.batch_size <= 0:
            errors.append(f"batch_size pozitif olmalı: {self.batch_size}")
        if self.grad_accum_steps <= 0:
            errors.append(f"grad_accum_steps pozitif olmalı: {self.grad_accum_steps}")
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
        if self.quant_type not in ("none", "int8", "int4"):
            raise ValueError(f"Geçersiz quant_type: {self.quant_type!r}")
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit ve load_in_4bit aynı anda True olamaz.")


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

    def validate_all(self) -> None:
        """Tüm alt konfigürasyonları doğrular. İlk hata anında durur."""
        self.arch.validate()
        self.training.validate()
        self.checkpoint.validate()
        self.distributed.validate()
        self.quant.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch.to_dict(),
            "training": self.training.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "distributed": self.distributed.to_dict(),
            "quant": self.quant.to_dict(),
        }

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "CevahirConfig":
        """
        train.py'deki düz TRAIN_CONFIG sözlüğünden CevahirConfig oluşturur.
        Bilinmeyen anahtarlar atlanır.
        """
        return cls(
            arch=ModelArchConfig.from_dict(d),    # type: ignore[return-value]
            training=TrainingConfig.from_dict(d),  # type: ignore[return-value]
            checkpoint=CheckpointConfig.from_dict(d),  # type: ignore[return-value]
            distributed=DistributedConfig.from_dict(d),  # type: ignore[return-value]
            quant=QuantConfig.from_dict(d),        # type: ignore[return-value]
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
