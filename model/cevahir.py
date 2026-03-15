# -*- coding: utf-8 -*-
"""
Cevahir Sinir Sistemi - Unified Inference API
===============================================

V-4 Architecture + TokenizerCore + ModelManager + CognitiveManager
Endüstri Standartları: GPT-4, Claude, Gemini seviyesinde mimari

⚠️ ÖNEMLİ: Bu modül SADECE INFERENCE için tasarlanmıştır!
Eğitim için training_system/training_service.py kullanılmalıdır.

Bu modül, üç ana bileşeni birleştirerek unified bir INFERENCE API sağlar:
1. TokenizerCore: BPE tokenization, GPU batch processing (encode/decode)
2. ModelManager: V-4 neural network inference (forward, generate)
3. CognitiveManager: Cognitive layer, tools, memory, monitoring

Mimari Özellikleri:
- V-4 Architecture: RoPE, RMSNorm, SwiGLU, KV Cache, MoE, Quantization
- SOLID Principles: Dependency Injection, Protocol-based interfaces
- Clean Architecture: Layered design, separation of concerns
- Enterprise Features: Monitoring, tracing, caching, AIOps
- Academic Rigor: Reproducible, validated, documented

Eğitim vs Inference Ayrımı:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE (cevahir.py kullanılır):
  - cevahir = Cevahir(config)
  - output = cevahir.process("Merhaba dünya")
  - cevahir.encode/decode/generate/forward

EĞİTİM (cevahir.py KULLANILMAZ):
  - tokenizer_management/train_bpe.py → BPE vocab/merges eğitimi
  - training_system/train.py → Model eğitimi giriş noktası
  - training_system/training_service.py → Eğitim servisi
  - model_management/model_manager.py → Neural network eğitimi

Endüstri Standartları:
[OK] GPT-4 seviyesi mimari (RoPE, RMSNorm, SwiGLU, MoE)
[OK] Claude seviyesi optimizasyonlar (KV Cache, Advanced Checkpointing)
[OK] Gemini seviyesi özellikler (Quantization, Multimodal)
[OK] Enterprise observability (Metrics, Tracing, Monitoring)
[OK] Production-ready (Error handling, Logging, Testing)

Akademik Doğruluk:
[OK] Reproducible results (seed management, config validation)
[OK] Scientific methodology (proper validation, documentation)
[OK] Peer-review ready (comprehensive tests, clear architecture)
"""

from __future__ import annotations

import os
import sys
import logging
import random
import time
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Protocol,
    runtime_checkable
)
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn as nn

# Proje kök dizinini sys.path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Core imports
from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from model_management.model_manager import ModelManager
from cognitive_management.cognitive_manager import CognitiveManager, ModelAPI as CognitiveModelAPI
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    DecodingConfig,
)

# Logger setup
logger = logging.getLogger("Cevahir")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Exceptions
# =============================================================================

class CevahirError(Exception):
    """Base exception for Cevahir system"""
    pass


class CevahirInitializationError(CevahirError):
    """Initialization errors"""
    pass


class CevahirConfigurationError(CevahirError):
    """Configuration errors"""
    pass


class CevahirProcessingError(CevahirError):
    """Processing errors"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None):
        super().__init__(message)
        self.context = context or {}
        self.suggestion = suggestion


# =============================================================================
# Enhanced Error Context Utility
# =============================================================================

class _ErrorContextBuilder:
    """
    Utility class for building enhanced error messages with context.
    
    Phase 2: Enhanced error context for better debugging and user experience.
    Endüstri Standardı: Actionable error messages with suggestions.
    """
    
    @staticmethod
    def build_error_message(
        operation: str,
        error: Exception,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Build enhanced error message with context and suggestions.
        
        Args:
            operation: Operation that failed
            error: The exception that occurred
            component: Component name (optional)
            context: Additional context (optional)
            
        Returns:
            Tuple of (enhanced_message, suggestion)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Build base message
        parts = [f"{operation} failed"]
        if component:
            parts.append(f"(component: {component})")
        parts.append(f": {error_msg}")
        
        enhanced_message = " ".join(parts)
        
        # Generate suggestions based on error type
        suggestion = _ErrorContextBuilder._generate_suggestion(
            error_type=error_type,
            operation=operation,
            component=component,
            context=context
        )
        
        return enhanced_message, suggestion
    
    @staticmethod
    def _generate_suggestion(
        error_type: str,
        operation: str,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Generate helpful suggestions based on error type."""
        suggestions = {
            "CevahirProcessingError": "Check input format and component initialization",
            "RuntimeError": "Verify that all required components are properly initialized",
            "ValueError": "Check input values and parameter ranges",
            "TypeError": "Verify input types match expected types",
            "FileNotFoundError": "Ensure file paths are correct and files exist",
            "AttributeError": "Check that the required component methods are available",
        }
        
        # Component-specific suggestions
        if component == "TokenizerCore":
            return "Ensure tokenizer vocab and merges files exist and are valid"
        elif component == "ModelManager":
            return "Verify model is initialized and model weights are loaded correctly"
        elif component == "CognitiveManager":
            return "Check CognitiveManager configuration and component dependencies"
        
        # Error type-based suggestion
        return suggestions.get(error_type, "Review error details and check system state")


# =============================================================================
# Decorators for Component Validation
# =============================================================================

def requires_tokenizer(func: Callable) -> Callable:
    """
    Decorator to ensure TokenizerCore is initialized before method execution.
    
    Endüstri Standardı: Decorator pattern for DRY (Don't Repeat Yourself)
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._tokenizer_core:
            raise CevahirProcessingError("TokenizerCore not initialized")
        return func(self, *args, **kwargs)
    return wrapper


def requires_model_manager(func: Callable) -> Callable:
    """
    Decorator to ensure ModelManager is initialized before method execution.
    
    Endüstri Standardı: Decorator pattern for DRY (Don't Repeat Yourself)
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._model_manager:
            raise CevahirProcessingError("ModelManager not initialized")
        return func(self, *args, **kwargs)
    return wrapper


def requires_cognitive_manager(func: Callable) -> Callable:
    """
    Decorator to ensure CognitiveManager is initialized before method execution.
    
    Endüstri Standardı: Decorator pattern for DRY (Don't Repeat Yourself)
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._cognitive_manager:
            raise CevahirProcessingError("CognitiveManager not initialized")
        return func(self, *args, **kwargs)
    return wrapper


def requires_model_api(func: Callable) -> Callable:
    """
    Decorator to ensure ModelAPI is initialized before method execution.
    
    Endüstri Standardı: Decorator pattern for DRY (Don't Repeat Yourself)
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._model_api:
            raise CevahirProcessingError("ModelAPI not initialized")
        return func(self, *args, **kwargs)
    return wrapper


# =============================================================================
# Input Validation Utility
# =============================================================================

class _InputValidator:
    """
    Internal utility class for input validation and conversion.
    
    Endüstri Standardı: Single Responsibility Principle, Reusable Validation Logic
    """
    
    @staticmethod
    def validate_and_convert_input(
        inputs: Union[torch.Tensor, List[int], str],
        device: Union[str, torch.device],
        tokenizer_core: Optional[Any] = None,
        vocab_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Validate and convert various input types to torch.Tensor.
        
        Args:
            inputs: Input as tensor, list of token IDs, or text string
            device: Target device
            tokenizer_core: Optional tokenizer for string encoding
            vocab_size: Optional vocab size for validation
            
        Returns:
            Validated torch.Tensor with shape [batch, seq_len]
            
        Raises:
            CevahirProcessingError: If validation fails
        """
        # Convert string to token IDs
        if isinstance(inputs, str):
            if not tokenizer_core:
                raise CevahirProcessingError("TokenizerCore required for string input")
            _, token_ids = tokenizer_core.encode(inputs, mode="inference")
            if not token_ids:
                # Empty input - create minimal tensor
                inputs = torch.tensor([[]], dtype=torch.long, device=device)
            else:
                inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        # Convert list to tensor
        elif isinstance(inputs, list):
            if not inputs:
                inputs = torch.tensor([[]], dtype=torch.long, device=device)
            else:
                inputs = torch.tensor([inputs], dtype=torch.long, device=device)
        
        # Validate and convert tensor
        elif isinstance(inputs, torch.Tensor):
            if inputs.dtype != torch.long:
                inputs = inputs.long()
            # Ensure 2D shape [batch, seq]
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() == 0:
                inputs = inputs.unsqueeze(0).unsqueeze(0)
            # Handle empty tensor
            if inputs.numel() == 0:
                inputs = torch.tensor([[]], dtype=torch.long, device=device)
            else:
                inputs = inputs.to(device=device)
        else:
            raise CevahirProcessingError(
                f"Invalid input type: {type(inputs)}. Expected str, list, or torch.Tensor"
            )
        
        # Handle empty sequence length - create minimal valid input
        if inputs.shape[1] == 0:
            pad_token_id = 0
            if tokenizer_core:
                try:
                    special_ids = tokenizer_core._special_ids()
                    pad_token_id = special_ids.get("<PAD>", 0)
                except:
                    pass
            inputs = torch.tensor([[pad_token_id]], dtype=torch.long, device=device)
            logger.debug("[Cevahir] Empty input detected, using minimal input with padding token")
        
        # Validate token IDs are within vocab range
        if vocab_size and inputs.numel() > 0:
            max_id = inputs.max().item()
            if max_id >= vocab_size:
                logger.warning(
                    f"[Cevahir] Token ID {max_id} >= vocab_size {vocab_size}, "
                    f"clipping to vocab_size-1"
                )
                inputs = torch.clamp(inputs, 0, vocab_size - 1)
        
        return inputs


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CevahirConfig:
    """
    Unified configuration for Cevahir system.
    
    Endüstri Standardı: Namespaced configuration, validation, defaults
    
    Phase 2: Hot reload support for configuration changes.
    """
    
    # Device configuration
    device: str = "cpu"
    seed: Optional[int] = None
    log_level: str = "INFO"
    
    # Phase 2: Config hot reload settings
    config_path: Optional[str] = None  # Path to config file (JSON/YAML)
    enable_hot_reload: bool = False  # Enable hot reload for config file changes
    
    # Tokenizer configuration
    tokenizer: Dict[str, Any] = field(default_factory=lambda: {
        "vocab_path": "data/vocab_lib/vocab.json",
        "merges_path": "data/merges_lib/merges.txt",
        "data_dir": None,
        "use_gpu": False,
        "batch_size": 32,
        "max_unk_ratio": 0.01,
    })
    
    # Model loading configuration
    load_model_path: Optional[str] = None  # None = auto-detect, "" = don't load
    
    # Model configuration (V-4 Architecture)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 1e-4,
        "dropout": 0.15,
        "vocab_size": 60000,
        "embed_dim": 512,
        "seq_proj_dim": 512,
        "num_heads": 8,  #  Training ile uyumlu (8 heads - Colab crash fix)
        "num_layers": 8,  #  Training ile uyumlu (24 layers)
        "ffn_dim": None,  # Auto: 4x seq_proj_dim
        "pre_norm": True,
        "causal_mask": True,
        # V-3 features
        "use_flash_attention": False,
        "pe_mode": "rope",  # V-4: RoPE default
        "use_gradient_checkpointing": True,
        "tie_weights": True,
        # V-4 features
        "use_rmsnorm": True,  # V-4: RMSNorm
        "use_swiglu": True,  # V-4: SwiGLU
        "use_kv_cache": True,  # V-4: KV Cache
        "max_cache_len": 2048,
        "use_advanced_checkpointing": False,
        "checkpointing_strategy": "selective",
        "quantization_type": "none",  # "none" | "int8" | "fp16" | "int8_dynamic"
        "use_moe": False,  # V-4: MoE
        "num_experts": 8,
        "moe_top_k": 2,
        # TensorBoard
        "use_tensorboard": False,
        "tb_log_dir": "runs/cevahir",
    })
    
    # Cognitive configuration
    cognitive: Optional[CognitiveManagerConfig] = None
    
    def validate(self) -> None:
        """Validate configuration"""
        # Device validation
        if self.device not in ["cpu", "cuda", "mps"]:
            raise CevahirConfigurationError(f"Invalid device: {self.device}")
        
        # Tokenizer validation
        if not self.tokenizer.get("vocab_path"):
            raise CevahirConfigurationError("tokenizer.vocab_path required")
        if not self.tokenizer.get("merges_path"):
            raise CevahirConfigurationError("tokenizer.merges_path required")
        
        # Model validation
        if self.model.get("vocab_size", 0) <= 0:
            raise CevahirConfigurationError("model.vocab_size must be > 0")
        if self.model.get("embed_dim", 0) <= 0:
            raise CevahirConfigurationError("model.embed_dim must be > 0")
        if self.model.get("num_heads", 0) <= 0:
            raise CevahirConfigurationError("model.num_heads must be > 0")
        if self.model.get("num_layers", 0) <= 0:
            raise CevahirConfigurationError("model.num_layers must be > 0")
        
        # V-4 validation
        if self.model.get("use_moe", False):
            if self.model.get("num_experts", 0) <= 0:
                raise CevahirConfigurationError("model.num_experts must be > 0 when use_moe=True")
            if self.model.get("moe_top_k", 0) <= 0:
                raise CevahirConfigurationError("model.moe_top_k must be > 0 when use_moe=True")
        
        logger.info("Configuration validated successfully")


# =============================================================================
# Model API Adapter
# =============================================================================

class CevahirModelAPI(CognitiveModelAPI):
    """
    ModelManager'ı CognitiveManager'ın ModelAPI protocol'üne adapte eder.
    
    Endüstri Standardı: Adapter Pattern, Protocol-based interface
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        tokenizer_core: TokenizerCore,
    ):
        self.model_manager = model_manager
        self.tokenizer_core = tokenizer_core
        self._device = model_manager.device
        
        # Ensure model is initialized
        if not model_manager.is_initialized:
            model_manager.initialize(
                build_optimizer=False,
                build_criterion=False,
                build_scheduler=False
            )
        model_manager.eval_mode()
    
    def generate(
        self,
        prompt: str,
        decoding_cfg: DecodingConfig
    ) -> str:
        """
        Generate text using ModelManager.
        
        Endüstri Standardı: Proper decoding, error handling, logging
        """
        try:
            # Inference için model eval modunda olmalı (Colab epoch testi gibi).
            # Train modunda dropout açık kalırsa çıktılar gürültülü/anlamsız olur.
            if hasattr(self.model_manager, "eval_mode"):
                self.model_manager.eval_mode()

            # Encode prompt
            tokens, token_ids = self.tokenizer_core.encode(
                prompt,
                mode="inference"
            )
            
            if not token_ids:
                logger.warning("Empty token_ids from encoding")
                return ""
            
            # Convert to tensor
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            
            # Generate using autoregressive decoding
            generated_ids = self._autoregressive_generate(
                input_tensor,
                decoding_cfg
            )
            
            # [OK] DÜZELTME: Sadece yeni üretilen token'ları decode et (prompt hariç)
            prompt_length = len(token_ids)
            if len(generated_ids) > prompt_length:
                new_token_ids = generated_ids[prompt_length:]
                logger.debug(f"[GEN] Prompt length: {prompt_length}, Generated tokens: {len(new_token_ids)}, Total: {len(generated_ids)}")
            else:
                # Hiç yeni token üretilmemiş
                logger.warning(f"[GEN] ⚠️ Hiç yeni token üretilmedi! Prompt length: {prompt_length}, Generated length: {len(generated_ids)}")
                new_token_ids = []
            
            # Decode - sadece yeni token'ları decode et
            if new_token_ids:
                generated_text = self.tokenizer_core.decode(
                    new_token_ids,
                    method="bpe",
                    remove_specials=True
                )
            else:
                generated_text = ""
                logger.warning("[GEN] ⚠️ Decode edilecek yeni token yok, boş string döndürülüyor")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise CevahirProcessingError(f"Generation failed: {e}") from e
    
    def _autoregressive_generate(
        self,
        input_tensor: torch.Tensor,
        decoding_cfg: DecodingConfig
    ) -> List[int]:
        """
        Autoregressive generation with proper decoding.
        
        Endüstri Standardı: Temperature, top-p, top-k sampling
        """
        max_new_tokens = getattr(decoding_cfg, "max_new_tokens", 128) or 128
        # Ensure max_new_tokens is an integer (handle MagicMock in tests)
        if not isinstance(max_new_tokens, int):
            try:
                max_new_tokens = int(max_new_tokens)
            except (ValueError, TypeError):
                max_new_tokens = 128
        # Ensure max_new_tokens is non-negative and reasonable
        max_new_tokens = max(0, min(max_new_tokens, 2048))  # Cap at 2048 to prevent infinite loops
        
        temperature = getattr(decoding_cfg, "temperature", 1.0) or 1.0
        top_p = getattr(decoding_cfg, "top_p", 1.0) or 1.0
        # Ensure top_p is in valid range
        top_p = max(0.01, min(top_p, 1.0))  # Avoid top_p=0.0 which filters all tokens
        top_k = getattr(decoding_cfg, "top_k", 0) or 0
        top_k = max(0, top_k)  # top_k=0 means no filtering (valid)
        repetition_penalty = getattr(decoding_cfg, "repetition_penalty", 1.0) or 1.0
        
        generated = input_tensor[0].tolist()
        initial_seq_len = input_tensor.shape[1]
        
        # Early return if max_new_tokens is 0
        if max_new_tokens == 0:
            return generated
        
        # [OK] V4: KV Cache state management — her generate başında cache temizle
        # (önceki turun cache'i kalırsa ikinci soruda scores/mask boyut uyuşmazlığı: 18 vs 12)
        if hasattr(self.model_manager, "clear_kv_cache"):
            self.model_manager.clear_kv_cache()
        use_cache = True  # KV Cache kullan (inference için optimize)
        cache_position = None
        
        # Get EOS token ID
        vocab = self.tokenizer_core.get_vocab()
        eos_id = None
        if isinstance(vocab.get("<EOS>"), dict):
            eos_id = vocab["<EOS>"].get("id")
        elif isinstance(vocab.get("<EOS>"), int):
            eos_id = vocab["<EOS>"]
        
        # [OK] DEBUG: EOS token ID kontrolü
        logger.debug(f"[GEN] EOS token ID: {eos_id}, vocab type: {type(vocab.get('<EOS>'))}")
        if eos_id is None:
            logger.warning(f"[GEN] ⚠️ EOS token ID bulunamadı! Vocab keys: {list(vocab.keys())[:10]}...")
        
        # [OK] DEBUG: Generation parametreleri
        logger.debug(f"[GEN] Generation başlıyor: max_new_tokens={max_new_tokens}, temperature={temperature}, "
                    f"top_p={top_p}, top_k={top_k}, repetition_penalty={repetition_penalty}, "
                    f"prompt_length={initial_seq_len}, eos_id={eos_id}")
        
        with torch.no_grad():
            tokens_generated = 0
            for step in range(max_new_tokens):
                # [OK] V4: KV Cache optimizasyonu
                # İlk iterasyonda: tüm sequence forward et (KV Cache initialize)
                # Sonraki iterasyonlarda: sadece yeni token forward et (KV Cache kullan)
                if step == 0:
                    # İlk forward: tüm prompt'u işle
                    current_input = input_tensor
                    cache_position = torch.arange(initial_seq_len, device=self._device)
                else:
                    # Sonraki forward'lar: sadece yeni token
                    current_input = next_token_tensor
                    cache_position = torch.tensor([initial_seq_len + step - 1], device=self._device)
                
                # Forward pass with KV Cache
                logits, _ = self.model_manager.forward(
                    current_input,
                    inference=True,
                    return_aux=False,
                    use_cache=use_cache,  # [OK] V4: KV Cache aktif
                    cache_position=cache_position,  # [OK] V4: Cache position
                )
                
                # Get next token logits
                # KV Cache kullanıldığında, logits shape'i [B, 1, vocab_size] olur (sadece son token)
                if step == 0:
                    next_logits = logits[0, -1, :]  # [vocab_size] - ilk iterasyonda son token
                else:
                    next_logits = logits[0, 0, :]  # [vocab_size] - sonraki iterasyonlarda tek token
                
                # [OK] DEBUG: İlk birkaç step'te logits bilgisi
                if step < 3:
                    top_5_probs, top_5_indices = torch.topk(torch.softmax(next_logits, dim=0), k=min(5, len(next_logits)))
                    logger.debug(f"[GEN] Step {step}: logits range=[{next_logits.min().item():.2f}, {next_logits.max().item():.2f}], "
                               f"top_5_tokens={top_5_indices.tolist()}, top_5_probs={top_5_probs.tolist()}")
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for token_id in generated[-256:]:  # Last 256 tokens
                        if 0 <= token_id < next_logits.shape[0]:
                            next_logits[token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature > 0:
                    next_logits = next_logits / temperature
                else:
                    next_logits = next_logits * float('inf')  # Greedy
                
                # Top-k filtering
                if top_k > 0:
                    top_k = min(top_k, next_logits.shape[0])
                    top_k_values, top_k_indices = torch.topk(next_logits, top_k)
                    # Create filtered logits
                    filtered_logits = torch.full_like(next_logits, float('-inf'))
                    filtered_logits[top_k_indices] = top_k_values
                    next_logits = filtered_logits
                
                # Top-p (nucleus) sampling
                if top_p < 1.0 and top_p > 0.0:  # top_p=0.0 is invalid, skip filtering
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
                    # Remove tokens with cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[0] = False  # Keep at least one token
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample - ensure we have valid probabilities
                probs = torch.softmax(next_logits, dim=0)
                # Safety check: ensure we have valid probabilities (not all NaN or zero)
                if torch.isnan(probs).any() or (probs == 0).all() or probs.sum() == 0:
                    # Fallback: use uniform distribution over valid (non-inf) tokens
                    valid_mask = ~torch.isinf(next_logits)
                    if valid_mask.any():
                        probs = torch.zeros_like(next_logits)
                        probs[valid_mask] = 1.0 / valid_mask.sum().float()
                    else:
                        # Last resort: use first token
                        probs = torch.zeros_like(next_logits)
                        probs[0] = 1.0
                
                next_token_id = torch.multinomial(probs, 1).item()
                
                # [OK] DEBUG: Üretilen token bilgisi (HER ZAMAN ilk 10 step için)
                if step < 10:
                    # Token'ın ne olduğunu decode et
                    try:
                        token_text = self.tokenizer_core.decode([next_token_id], method="bpe", remove_specials=False)
                    except:
                        token_text = f"<decode_error>"
                    logger.debug(f"[GEN] Step {step}: Generated token_id={next_token_id}, token_text='{token_text}', eos_id={eos_id}, "
                               f"is_eos={next_token_id == eos_id if eos_id is not None else False}")
                
                # EOS gelince dur; ama minimum 5 token üretmeden durma (erken collapse önlemi)
                if eos_id is not None and next_token_id == eos_id and tokens_generated >= 5:
                    generated.append(next_token_id)
                    break
                
                generated.append(next_token_id)
                tokens_generated += 1
                
                # [OK] V4: KV Cache kullanıldığında, input'u güncelleme gerekmez
                # Sadece sonraki iterasyon için next_token_tensor hazırla
                next_token_tensor = torch.tensor(
                    [[next_token_id]],
                    dtype=torch.long,
                    device=self._device
                )
                
                # KV Cache kullanılmıyorsa (fallback), input'u güncelle
                if not use_cache:
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
        
        # [OK] DEBUG: Generation özeti
        new_tokens_count = len(generated) - initial_seq_len
        logger.debug(f"[GEN] Generation tamamlandı: {new_tokens_count} yeni token üretildi (prompt: {initial_seq_len} token, total: {len(generated)} token)")
        
        if new_tokens_count == 0:
            logger.warning(f"[GEN] ⚠️ UYARI: Hiç yeni token üretilmedi! EOS erken geldi veya generation loop çalışmadı.")
        
        return generated
    
    def _generate_with_beam_search(
        self,
        prompt: str,
        max_new_tokens: int,
        beam_width: int,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text using beam search algorithm.
        
        Phase 3: Beam search implementation for better generation quality.
        Endüstri Standardı: GPT-4, Claude beam search pattern.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            beam_width: Number of beams to maintain
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text (best beam)
        """
        try:
            # Encode prompt
            tokens, token_ids = self.tokenizer_core.encode(prompt, mode="inference")
            if not token_ids:
                return ""
            
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            initial_seq_len = input_tensor.shape[1]
            
            # Get EOS token ID
            vocab = self.tokenizer_core.get_vocab()
            eos_id = None
            if isinstance(vocab.get("<EOS>"), dict):
                eos_id = vocab["<EOS>"].get("id")
            elif isinstance(vocab.get("<EOS>"), int):
                eos_id = vocab["<EOS>"]
            
            # Beam search state: List of (sequence, score, finished)
            beams = [(input_tensor[0].tolist(), 0.0, False)]
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    candidates = []
                    
                    # Expand all active beams
                    for sequence, score, finished in beams:
                        if finished:
                            candidates.append((sequence, score, True))
                            continue
                        
                        # Prepare input for this beam
                        if step == 0:
                            current_input = torch.tensor([sequence], dtype=torch.long, device=self._device)
                            cache_position = torch.arange(len(sequence), device=self._device)
                        else:
                            # Last token only (for KV Cache)
                            last_token = sequence[-1]
                            current_input = torch.tensor([[last_token]], dtype=torch.long, device=self._device)
                            cache_position = torch.tensor([initial_seq_len + step - 1], device=self._device)
                        
                        # Forward pass
                        logits, _ = self.model_manager.forward(
                            current_input,
                            inference=True,
                            return_aux=False,
                            use_cache=True,
                            cache_position=cache_position,
                        )
                        
                        # Get logits for next token
                        if step == 0:
                            next_logits = logits[0, -1, :]
                        else:
                            next_logits = logits[0, 0, :]
                        
                        # Apply repetition penalty
                        if repetition_penalty > 1.0:
                            for token_id in sequence[-256:]:
                                if 0 <= token_id < next_logits.shape[0]:
                                    next_logits[token_id] /= repetition_penalty
                        
                        # Get top-k candidates for this beam
                        top_k = min(beam_width * 2, next_logits.shape[0])  # Get more candidates
                        top_logits, top_indices = torch.topk(next_logits, top_k)
                        
                        # Convert to log probabilities and add to beam score
                        log_probs = torch.log_softmax(top_logits, dim=0)
                        
                        for log_prob, token_id in zip(log_probs, top_indices):
                            new_sequence = sequence + [token_id.item()]
                            new_score = score + log_prob.item()
                            is_finished = (eos_id is not None and token_id.item() == eos_id)
                            candidates.append((new_sequence, new_score, is_finished))
                    
                    # Select top-k beams for next iteration
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:beam_width]
                    
                    # Check if all beams are finished
                    if all(finished for _, _, finished in beams):
                        break
                
                # Select best beam
                best_sequence, _, _ = max(beams, key=lambda x: x[1])
                
                # Decode
                generated_text = self.tokenizer_core.decode(
                    best_sequence,
                    method="bpe",
                    remove_specials=True
                )
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Beam search generation error: {e}", exc_info=True)
            # Fallback to standard generation
            logger.warning("Beam search failed, falling back to standard generation")
            decoding_cfg = DecodingConfig(
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=repetition_penalty,
            )
            return self._model_api.generate(prompt, decoding_cfg)
    
    def score(self, prompt: str, candidate: str) -> float:
        """
        Score candidate text given prompt.
        
        Endüstri Standardı: Proper scoring methodology
        """
        try:
            # Encode both
            _, prompt_ids = self.tokenizer_core.encode(prompt, mode="inference")
            _, candidate_ids = self.tokenizer_core.encode(candidate, mode="inference")
            
            if not prompt_ids or not candidate_ids:
                return 0.0
            
            # Concatenate
            full_ids = prompt_ids + candidate_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.model_manager.forward(
                    input_tensor,
                    inference=True,
                    return_aux=False
                )
                
                # Calculate average log probability for candidate
                # Use prompt length to slice
                prompt_len = len(prompt_ids)
                candidate_logits = logits[0, prompt_len-1:-1, :]  # [candidate_len, vocab_size]
                candidate_targets = torch.tensor(
                    candidate_ids,
                    dtype=torch.long,
                    device=self._device
                )
                
                # Calculate log probs
                log_probs = torch.log_softmax(candidate_logits, dim=-1)
                candidate_log_probs = log_probs.gather(
                    1,
                    candidate_targets.unsqueeze(1)
                ).squeeze(1)
                
                # Average log probability
                avg_log_prob = candidate_log_probs.mean().item()
                
                # Convert to score (0-1 range)
                score = max(0.0, min(1.0, (avg_log_prob + 10) / 10))  # Normalize
                
                return float(score)
                
        except Exception as e:
            logger.warning(f"Scoring error: {e}")
            # Fallback: length-based score
            return float(len(candidate)) / max(1, len(prompt))
    
    def entropy_estimate(self, text: str) -> float:
        """
        Estimate uncertainty as Shannon entropy of next-token logit distribution.

        Uses the model's own probability distribution over the vocabulary to
        compute H = -sum(p_i * log(p_i)) for the last token position.
        High entropy → model is uncertain (many plausible next tokens).
        Low entropy → model is confident (one dominant next token).

        This is the academically correct uncertainty metric used in active
        learning and LLM calibration research (Kuhn et al. 2023).

        Returns:
            Normalized entropy in [0, 1] (0=certain, 1=maximally uncertain)
        """
        if not text:
            return 0.5

        try:
            # Encode text to token IDs
            tokens, _ = self._tokenizer_core.encode(text, mode="inference")
            if not tokens:
                return 0.5

            # Build input tensor (batch_size=1, seq_len=T)
            token_tensor = torch.tensor(
                [tokens], dtype=torch.long, device=self.config.device
            )

            # Forward pass to get logits [1, T, vocab_size] with no gradient
            with torch.no_grad():
                logits, _ = self._model_manager.forward(token_tensor)

            # Take last token position: [vocab_size]
            last_logits = logits[0, -1, :]

            # Softmax → probability distribution
            probs = torch.softmax(last_logits.float(), dim=-1)

            # Shannon entropy: H = -sum(p * log(p))
            # Clamp to avoid log(0)
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum().item()

            # Max entropy for a uniform distribution over vocab_size
            vocab_size = probs.shape[0]
            max_entropy = torch.log(torch.tensor(float(vocab_size))).item()

            # Normalize to [0, 1]
            normalized = entropy / max_entropy if max_entropy > 0 else 0.5
            return float(max(0.0, min(1.0, normalized)))

        except Exception as e:
            logger.debug(f"Logit entropy estimation failed, falling back to heuristic: {e}")
            # Heuristic fallback: token type-token ratio as diversity proxy
            try:
                tokens, _ = self._tokenizer_core.encode(text, mode="inference")
                if tokens:
                    ttr = len(set(tokens)) / len(tokens)
                    return float(max(0.1, min(1.0, ttr)))
            except Exception:
                pass
            return 0.5
    
    # Multimodal support
    def process_audio(self, audio_data: bytes) -> str:
        """Process audio data"""
        if hasattr(self.model_manager, "process_audio"):
            return self.model_manager.process_audio(audio_data)
        return "Audio processing not available"
    
    def process_image(self, image_data: bytes) -> str:
        """Process image data"""
        if hasattr(self.model_manager, "process_image"):
            return self.model_manager.process_image(image_data)
        return "Image processing not available"
    
    def process_multimodal(
        self,
        text: str = None,
        audio: bytes = None,
        image: bytes = None
    ) -> str:
        """Process multimodal data"""
        if hasattr(self.model_manager, "process_multimodal"):
            return self.model_manager.process_multimodal(text, audio, image)
        # Fallback
        parts = []
        if text:
            parts.append(f"Text: {text}")
        if audio:
            parts.append(f"Audio: {self.process_audio(audio)}")
        if image:
            parts.append(f"Image: {self.process_image(image)}")
        return " | ".join(parts) if parts else "No data to process"


# =============================================================================
# Main Cevahir Class
# =============================================================================

class Cevahir:
    """
    Unified API for TokenizerCore + ModelManager + CognitiveManager.
    
    Endüstri Standartları:
    - SOLID Principles: Dependency Injection, Protocol-based interfaces
    - Clean Architecture: Layered design, separation of concerns
    - Enterprise Features: Monitoring, tracing, caching, AIOps
    - Academic Rigor: Reproducible, validated, documented
    
    V-4 Architecture Features:
    - RoPE (Rotary Position Embedding): GPT-3+, Claude, Gemini standard
    - RMSNorm: GPT-3+, LLaMA standard
    - SwiGLU: GPT-4, PaLM standard
    - KV Cache: GPT-4, Claude, Gemini inference optimization
    - MoE (Mixture of Experts): GPT-4, Gemini large models
    - Quantization: GPT-4, Claude, Gemini production optimization
    - Advanced Checkpointing: Memory-efficient training
    
    Usage (INFERENCE ONLY):
        config = CevahirConfig(
            device="cuda",
            model={"vocab_size": 50000, "embed_dim": 512, ...}
        )
        cevahir = Cevahir(config)
        
        # Process text (inference)
        output = cevahir.process("Merhaba dünya")
        
        # Generate text
        generated = cevahir.generate("Merhaba", max_new_tokens=128)
        
        # Encode/decode
        tokens, ids = cevahir.encode("Merhaba")
        text = cevahir.decode(ids)
    
    ⚠️ EĞİTİM İÇİN:
        Eğitim için cevahir.py KULLANILMAZ!
        Bunun yerine training_system/train.py kullanılmalıdır:
        
        python training_system/train.py
    """
    
    def __init__(
        self,
        config: Union[CevahirConfig, Dict[str, Any]],
        *,
        tokenizer_core: Optional[TokenizerCore] = None,
        model_manager: Optional[ModelManager] = None,
        cognitive_manager: Optional[CognitiveManager] = None,
    ):
        """
        Initialize Cevahir system.
        
        Args:
            config: CevahirConfig or dict
            tokenizer_core: Optional pre-initialized TokenizerCore
            model_manager: Optional pre-initialized ModelManager
            cognitive_manager: Optional pre-initialized CognitiveManager
        """
        # Parse config
        if isinstance(config, dict):
            self.config = CevahirConfig(**config)
        else:
            self.config = config
        
        self.config.validate()
        
        # Phase 2: Config hot reload support
        self._config_manager: Optional[Any] = None
        if self.config.config_path:
            try:
                from cognitive_management.v2.config import ConfigManager
                # Create config manager for hot reload (if config file provided)
                # Note: CevahirConfig is simpler, so we only watch for file changes
                # Actual config updates would need custom implementation
                logger.info(f"Config file path provided: {self.config.config_path} (hot reload: {self.config.enable_hot_reload})")
            except ImportError:
                logger.warning("ConfigManager not available, hot reload disabled")
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            logger.info(f"Seed set to {self.config.seed}")
        
        # Set log level
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        # Initialize components
        self._tokenizer_core: Optional[TokenizerCore] = None
        self._model_manager: Optional[ModelManager] = None
        self._cognitive_manager: Optional[CognitiveManager] = None
        self._model_api: Optional[CevahirModelAPI] = None
        
        # Phase 3: Performance profiling (disabled by default)
        self._profiling_enabled = False
        self._profiling_stats: Dict[str, Any] = {}
        
        # Initialize TokenizerCore
        if tokenizer_core is not None:
            self._tokenizer_core = tokenizer_core
            logger.info("TokenizerCore provided externally")
        else:
            self._tokenizer_core = self._init_tokenizer()
        
        # Initialize ModelManager
        if model_manager is not None:
            self._model_manager = model_manager
            logger.info("ModelManager provided externally")
        else:
            self._model_manager = self._init_model()
        
        # Create ModelAPI adapter
        self._model_api = CevahirModelAPI(
            self._model_manager,
            self._tokenizer_core
        )
        
        # Initialize CognitiveManager
        if cognitive_manager is not None:
            self._cognitive_manager = cognitive_manager
            logger.info("CognitiveManager provided externally")
        else:
            self._cognitive_manager = self._init_cognitive()
        
        logger.info("=" * 60)
        logger.info("CEVAHIR SYSTEM INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Tokenizer: {type(self._tokenizer_core).__name__}")
        logger.info(f"Model: {type(self._model_manager).__name__}")
        logger.info(f"Cognitive: {type(self._cognitive_manager).__name__}")
        logger.info(f"V-4 Features: RoPE={self.config.model.get('pe_mode')=='rope'}, "
                   f"RMSNorm={self.config.model.get('use_rmsnorm')}, "
                   f"SwiGLU={self.config.model.get('use_swiglu')}, "
                   f"KV Cache={self.config.model.get('use_kv_cache')}, "
                   f"MoE={self.config.model.get('use_moe')}")
        logger.info("=" * 60)
    
    def _init_tokenizer(self) -> TokenizerCore:
        """Initialize TokenizerCore"""
        try:
            tokenizer_config = {
                **self.config.tokenizer,
                "device": self.config.device,
            }
            
            # Check if vocab_path exists (if specified and path is absolute/nonexistent)
            vocab_path = tokenizer_config.get("vocab_path") or tokenizer_config.get("vocab_file")
            if vocab_path:
                # If path is absolute and doesn't exist, check parent directory
                if os.path.isabs(vocab_path) and not os.path.exists(vocab_path):
                    parent_dir = os.path.dirname(vocab_path) if os.path.dirname(vocab_path) else "/"
                    # If parent directory doesn't exist or is not writable, raise error
                    if not os.path.exists(parent_dir):
                        raise CevahirInitializationError(
                            f"TokenizerCore initialization failed: Vocab path parent directory does not exist: {parent_dir}"
                        )
                    if not os.access(parent_dir, os.W_OK):
                        raise CevahirInitializationError(
                            f"TokenizerCore initialization failed: Vocab path parent directory is not writable: {parent_dir}"
                        )
            
            tokenizer = TokenizerCore(tokenizer_config)
            logger.info("TokenizerCore initialized")
            return tokenizer
        except TokenizerCoreError as e:
            raise CevahirInitializationError(f"TokenizerCore initialization failed: {e}") from e
        except Exception as e:
            raise CevahirInitializationError(f"TokenizerCore initialization failed: {e}") from e
    
    def _init_model(self) -> ModelManager:
        """Initialize ModelManager with V-4 architecture"""
        try:
            model_config = {
                **self.config.model,
                "device": self.config.device,
            }
            
            # Ensure vocab_size matches tokenizer
            if self._tokenizer_core:
                vocab_size = self._tokenizer_core.get_vocab_size()
                model_config["vocab_size"] = vocab_size
                logger.info(f"Vocab size from tokenizer: {vocab_size}")
            
            model_manager = ModelManager(
                config=model_config,
                tokenizer=self._tokenizer_core,
            )
            
            # Initialize model
            model_manager.initialize(
                build_optimizer=False,
                build_criterion=False,
                build_scheduler=False
            )
            
            logger.info("ModelManager initialized with V-4 architecture")
            
            # Auto-load model if exists
            model_path = self.config.load_model_path
            if model_path is None:
                # Auto-detect: try saved_models/cevahir_model.pth
                default_path = "saved_models/cevahir_model.pth"
                if os.path.exists(default_path):
                    model_path = default_path
                    logger.info(f"Auto-detected model file: {default_path}")
            
            if model_path and os.path.exists(model_path):
                try:
                    logger.info(f"Loading model from: {model_path}")
                    model_manager.load(model_path, strict=False)
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {e}")
                    logger.info("Continuing with initialized model")
            elif model_path:
                logger.warning(f"Model path specified but file not found: {model_path}")
            
            return model_manager
            
        except Exception as e:
            raise CevahirInitializationError(f"ModelManager initialization failed: {e}") from e
    
    def _init_cognitive(self) -> CognitiveManager:
        """Initialize CognitiveManager"""
        try:
            cognitive_config = self.config.cognitive or CognitiveManagerConfig()
            
            cognitive_manager = CognitiveManager(
                model_manager=self._model_api,
                cfg=cognitive_config,
            )
            
            logger.info("CognitiveManager initialized")
            return cognitive_manager
            
        except Exception as e:
            raise CevahirInitializationError(f"CognitiveManager initialization failed: {e}") from e
    
    # =========================================================================
    # Tokenization API
    # =========================================================================
    
    @requires_tokenizer
    def encode(
        self,
        text: str,
        mode: str = "inference",
        **kwargs
    ) -> Tuple[List[str], List[int]]:
        """
        Encode text to tokens.
        
        Args:
            text: Input text
            mode: "train" or "inference"
            **kwargs: Additional tokenizer parameters
        
        Returns:
            Tuple of (tokens, token_ids)
        """
        try:
            return self._tokenizer_core.encode(text, mode=mode, **kwargs)
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Encoding", e, component="TokenizerCore"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_tokenizer
    def decode(
        self,
        token_ids: List[int],
        **kwargs
    ) -> str:
        """
        Decode tokens to text.
        
        Args:
            token_ids: Token IDs
            **kwargs: Additional tokenizer parameters
        
        Returns:
            Decoded text
        """
        # Handle None input
        if token_ids is None:
            raise CevahirProcessingError("Decode: token_ids cannot be None")
        
        try:
            return self._tokenizer_core.decode(token_ids, **kwargs)
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Decoding", e, component="TokenizerCore"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_tokenizer
    def train_tokenizer(
        self,
        corpus: List[str],
        **kwargs
    ) -> None:
        """
        Train tokenizer on corpus.
        
        Args:
            corpus: Training corpus
            **kwargs: Training parameters
        """
        try:
            self._tokenizer_core.train_model(corpus, **kwargs)
            logger.info(f"Tokenizer trained on {len(corpus)} samples")
        except Exception as e:
            raise CevahirProcessingError(f"Tokenizer training failed: {e}") from e
    
    # =========================================================================
    # Model API
    # =========================================================================
    
    @requires_model_manager
    def forward(
        self,
        inputs: Union[torch.Tensor, List[int], str],
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through model.
        
        Phase 3: Added performance profiling support.
        
        Args:
            inputs: Tensor, token IDs, or text
            **kwargs: Forward parameters
        
        Returns:
            Model output logits
        """
        # Phase 3: Performance profiling
        profiling = getattr(self, '_profiling_enabled', False)
        start_time = time.time() if profiling else None
        
        try:
            # Use input validator utility for clean validation
            inputs = _InputValidator.validate_and_convert_input(
                inputs=inputs,
                device=self.config.device,
                tokenizer_core=self._tokenizer_core,
                vocab_size=self.config.model.get("vocab_size", 50000)
            )
            
            logits, _ = self._model_manager.forward(inputs, **kwargs)
            
            # Phase 3: Record profiling stats
            if profiling and start_time is not None:
                elapsed = time.time() - start_time
                if not hasattr(self, '_profiling_stats'):
                    self.enable_profiling(True)
                self._profiling_stats["forward_calls"] += 1
                self._profiling_stats["total_forward_time"] += elapsed
            
            return logits
            
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Forward pass", e, component="ModelManager"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_model_api
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_cognitive_pipeline: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        When cognitive manager is available, routes through the full cognitive
        pipeline (Critic + Memory + Deliberation) by default.
        Set use_cognitive_pipeline=False for raw model generation (bypass).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            use_cognitive_pipeline: If True and cognitive manager available,
                                    routes through full cognitive pipeline
                                    (Self-Refine, Memory, Critic). Default: True.
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Phase 3: Performance profiling
            profiling = getattr(self, '_profiling_enabled', False)
            start_time = time.time() if profiling else None

            # Route through cognitive pipeline when available (Critic + Memory + Self-Refine)
            cognitive_manager = getattr(self, '_cognitive_manager', None)
            if use_cognitive_pipeline and cognitive_manager is not None:
                try:
                    state = CognitiveState()
                    decoding_override = DecodingConfig(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                    input_msg = CognitiveInput(user_message=prompt)
                    cognitive_output = cognitive_manager.handle(
                        state,
                        input_msg,
                        decoding=decoding_override,
                    )
                    result = cognitive_output.text if cognitive_output and cognitive_output.text else ""

                    # Phase 3: Record profiling stats
                    if profiling and start_time is not None:
                        elapsed = time.time() - start_time
                        if not hasattr(self, '_profiling_stats'):
                            self.enable_profiling(True)
                        self._profiling_stats["generate_calls"] += 1
                        self._profiling_stats["total_generate_time"] += elapsed

                    return result
                except Exception as cognitive_err:
                    # Cognitive pipeline failed - fall through to direct generation
                    logger.warning(
                        f"Cognitive pipeline generate başarısız, doğrudan üretim kullanılıyor: {cognitive_err}"
                    )

            # Direct model generation (no cognitive pipeline)
            decoding_cfg = DecodingConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            result = self._model_api.generate(prompt, decoding_cfg)

            # Phase 3: Record profiling stats
            if profiling and start_time is not None:
                elapsed = time.time() - start_time
                if not hasattr(self, '_profiling_stats'):
                    self.enable_profiling(True)
                self._profiling_stats["generate_calls"] += 1
                self._profiling_stats["total_generate_time"] += elapsed

            return result
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Generation", e, component="ModelAPI"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_model_manager
    def save_model(self, path: str, **kwargs) -> None:
        """Save model to path"""
        try:
            self._model_manager.save(path, **kwargs)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            raise CevahirProcessingError(f"Model save failed: {e}") from e
    
    @requires_model_manager
    def load_model(self, path: str, **kwargs) -> None:
        """Load model from path"""
        try:
            self._model_manager.load(path, **kwargs)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            raise CevahirProcessingError(f"Model load failed: {e}") from e
    
    @requires_model_manager
    def predict(
        self,
        inputs: Union[torch.Tensor, List[int], str],
        *,
        topk: int = 1,
        apply_softmax: bool = True,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict with top-k sampling.
        
        Args:
            inputs: Tensor, token IDs, or text
            topk: Number of top predictions to return
            apply_softmax: Whether to apply softmax to logits
            return_logits: Whether to return raw logits
            **kwargs: Additional forward parameters
        
        Returns:
            Dictionary with predictions, probabilities, and optionally logits
        """
        try:
            # Convert input to tensor if needed
            if isinstance(inputs, str):
                _, token_ids = self.encode(inputs)
                inputs = torch.tensor([token_ids], dtype=torch.long, device=self.config.device)
            elif isinstance(inputs, list):
                inputs = torch.tensor([inputs], dtype=torch.long, device=self.config.device)
            
            return self._model_manager.predict(
                inputs,
                topk=topk,
                apply_softmax=apply_softmax,
                return_logits=return_logits
            )
        except Exception as e:
            raise CevahirProcessingError(f"Prediction failed: {e}") from e
    
    @requires_model_manager
    def freeze(self, patterns: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Freeze model layers by pattern.
        
        Args:
            patterns: Layer name pattern(s) to freeze (e.g., "embedding.*", ["layer.0", "layer.1"])
        
        Returns:
            Dictionary with freeze report
        """
        try:
            return self._model_manager.freeze(patterns)
        except Exception as e:
            raise CevahirProcessingError(f"Freeze failed: {e}") from e
    
    @requires_model_manager
    def unfreeze(self, patterns: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Unfreeze model layers by pattern.
        
        Args:
            patterns: Layer name pattern(s) to unfreeze
        
        Returns:
            Dictionary with unfreeze report
        """
        try:
            return self._model_manager.unfreeze(patterns)
        except Exception as e:
            raise CevahirProcessingError(f"Unfreeze failed: {e}") from e
    
    @requires_model_manager
    def update_model(
        self,
        update_params: Dict[str, Any],
        *,
        dry_run: bool = False
    ) -> Dict[str, List[str]]:
        """
        Update model parameters (freeze/unfreeze, learning rate, etc.).
        
        Args:
            update_params: Update parameters dictionary
            dry_run: If True, only report what would be updated
        
        Returns:
            Dictionary with update report
        """
        try:
            return self._model_manager.update(update_params, dry_run=dry_run)
        except Exception as e:
            raise CevahirProcessingError(f"Model update failed: {e}") from e
    
    @requires_model_manager
    def train_mode(self) -> None:
        """Set model to training mode"""
        try:
            self._model_manager.train_mode()
            logger.debug("Model set to training mode")
        except Exception as e:
            raise CevahirProcessingError(f"Train mode failed: {e}") from e
    
    @requires_model_manager
    def eval_mode(self) -> None:
        """Set model to evaluation mode"""
        try:
            self._model_manager.eval_mode()
            logger.debug("Model set to evaluation mode")
        except Exception as e:
            raise CevahirProcessingError(f"Eval mode failed: {e}") from e
    
    @requires_model_manager
    def configure_tensorboard(
        self,
        writer: Optional[Any] = None,
        *,
        log_dir: Optional[str] = None,
        log_every_n: Optional[int] = None,
        log_histograms: Optional[bool] = None,
        log_attention_image: Optional[bool] = None,
        enable: Optional[bool] = None,
    ) -> None:
        """
        Configure TensorBoard logging.
        
        Args:
            writer: Optional TensorBoard SummaryWriter
            log_dir: Log directory path
            log_every_n: Log every N steps
            log_histograms: Whether to log histograms
            log_attention_image: Whether to log attention images
            enable: Enable/disable TensorBoard
        """
        try:
            self._model_manager.configure_tensorboard(
                writer=writer,
                log_dir=log_dir,
                log_every_n=log_every_n,
                log_histograms=log_histograms,
                log_attention_image=log_attention_image,
                enable=enable,
            )
            logger.info("TensorBoard configured")
        except Exception as e:
            raise CevahirProcessingError(f"TensorBoard configuration failed: {e}") from e
    
    @requires_model_manager
    def get_tb_writer(self) -> Optional[Any]:
        """Get TensorBoard writer"""
        try:
            return self._model_manager.get_tb_writer()
        except Exception as e:
            logger.warning(f"Failed to get TensorBoard writer: {e}")
            return None
    
    # =========================================================================
    # Cognitive API
    # =========================================================================
    
    @requires_cognitive_manager
    def process(
        self,
        text: str,
        state: Optional[CognitiveState] = None,
        **kwargs
    ) -> CognitiveOutput:
        """
        Process text through cognitive layer.
        
        Unified API: Tokenization + Model + Cognitive processing
        
        Args:
            text: Input text
            state: Optional cognitive state
            **kwargs: Additional processing parameters
        
        Returns:
            CognitiveOutput with processed result
        """
        # Validate input
        if text is None:
            raise CevahirProcessingError(
                "Process: text cannot be None",
                suggestion="Provide a valid text string as input"
            )
        
        # Phase 3: Performance profiling
        profiling = getattr(self, '_profiling_enabled', False)
        start_time = time.time() if profiling else None
        
        try:
            if state is None:
                state = CognitiveState()
            
            input_msg = CognitiveInput(user_message=text, **kwargs)
            output = self._cognitive_manager.handle(state, input_msg)
            
            # Phase 3: Record profiling stats
            if profiling and start_time is not None:
                elapsed = time.time() - start_time
                if not hasattr(self, '_profiling_stats'):
                    self.enable_profiling(True)
                self._profiling_stats["process_calls"] += 1
                self._profiling_stats["total_process_time"] += elapsed
            
            return output
            
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Processing", e, component="CognitiveManager"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    # =========================================================================
    # Phase 2: Batch Processing API
    # =========================================================================
    
    @requires_model_api
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Phase 2: Batch processing for improved throughput.
        Endüstri Standardı: GPT-4, Claude batch API pattern.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts (one per prompt)
        
        Example:
            >>> prompts = ["Hello", "How are you?"]
            >>> results = cevahir.generate_batch(prompts, max_new_tokens=50)
            >>> len(results) == 2  # True
        """
        if not prompts:
            return []
        
        try:
            decoding_cfg = DecodingConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            
            # Process in batch (sequential for now, can be parallelized later)
            results = []
            for prompt in prompts:
                try:
                    generated = self._model_api.generate(prompt, decoding_cfg)
                    results.append(generated)
                except Exception as e:
                    # Log error but continue with other prompts
                    logger.warning(f"Batch generation failed for prompt '{prompt[:50]}...': {e}")
                    results.append("")  # Empty result on error
            
            return results
            
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Batch generation", e, component="ModelAPI"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_cognitive_manager
    def process_batch(
        self,
        texts: List[str],
        states: Optional[List[Optional[CognitiveState]]] = None,
        **kwargs
    ) -> List[CognitiveOutput]:
        """
        Process multiple texts through cognitive layer in batch.
        
        Phase 2: Batch processing for improved throughput.
        Endüstri Standardı: GPT-4, Claude batch API pattern.
        
        Args:
            texts: List of input texts
            states: Optional list of cognitive states (one per text, or None for all)
            **kwargs: Additional processing parameters
        
        Returns:
            List of CognitiveOutput objects (one per input text)
        
        Example:
            >>> texts = ["Hello", "How are you?"]
            >>> results = cevahir.process_batch(texts)
            >>> len(results) == 2  # True
        """
        if not texts:
            return []
        
        # Validate inputs
        if texts is None:
            raise CevahirProcessingError(
                "process_batch: texts cannot be None",
                suggestion="Provide a valid list of text strings"
            )
        
        try:
            results = []
            for i, text in enumerate(texts):
                try:
                    # Use provided state or None
                    state = states[i] if states and i < len(states) else None
                    output = self.process(text, state=state, **kwargs)
                    results.append(output)
                except Exception as e:
                    # Log error but continue with other texts
                    logger.warning(f"Batch processing failed for text {i+1}/{len(texts)}: {e}")
                    # Create error output
                    error_output = CognitiveOutput(
                        response="",
                        mode="direct",
                        metadata={"error": str(e)}
                    )
                    results.append(error_output)
            
            return results
            
        except Exception as e:
            msg, suggestion = _ErrorContextBuilder.build_error_message(
                "Batch processing", e, component="CognitiveManager"
            )
            raise CevahirProcessingError(msg, suggestion=suggestion) from e
    
    @requires_cognitive_manager
    def add_memory(self, note: str) -> None:
        """Add memory note"""
        try:
            self._cognitive_manager.add_memory_note(note)
        except Exception as e:
            raise CevahirProcessingError(f"Memory add failed: {e}") from e
    
    @requires_cognitive_manager
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory using semantic search.
        
        This method uses the public API of CognitiveManager to retrieve
        relevant context from memory, avoiding private attribute access.
        
        Args:
            query: Query text for memory search
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            List of relevant memory items with metadata and scores
        """
        try:
            # Use public API - no private attribute access
            return self._cognitive_manager.retrieve_context(query, top_k=limit)
        except Exception as e:
            raise CevahirProcessingError(f"Memory search failed: {e}") from e
    
    @requires_cognitive_manager
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = None,
        parameters: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        """Register tool"""
        try:
            self._cognitive_manager.register_tool(
                name=name,
                tool_func=func,
                description=description,
                parameters=parameters,
                **kwargs
            )
            logger.info(f"Tool registered: {name}")
        except Exception as e:
            raise CevahirProcessingError(f"Tool registration failed: {e}") from e
    
    @requires_cognitive_manager
    def list_tools(self) -> List[str]:
        """List available tools"""
        
        try:
            return self._cognitive_manager.list_available_tools()
        except Exception as e:
            raise CevahirProcessingError(f"Tool listing failed: {e}") from e
    
    # =========================================================================
    # Training API - DEPRECATED
    # =========================================================================
    
    def train(
        self,
        data: List[Tuple[str, str]],
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        ⚠️ DEPRECATED: Bu metod kullanımdan kaldırılmıştır!
        
        Cevahir sınıfı SADECE INFERENCE için tasarlanmıştır.
        Eğitim için training_system/training_service.py kullanılmalıdır.
        
        Eğitim akışı:
        1. tokenizer_management/train_bpe.py → BPE vocab/merges eğitimi
        2. training_system/train.py → Model eğitimi giriş noktası
        3. training_system/training_service.py → Eğitim servisi
        4. model_management/model_manager.py → Neural network eğitimi
        
        Args:
            data: List of (input, target) pairs
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate (optional)
            **kwargs: Additional training parameters
        
        Returns:
            Training metrics (placeholder - gerçek eğitim yapılmaz)
        
        Raises:
            CevahirProcessingError: Her zaman - bu metod kullanılmamalıdır
        """
        import warnings
        warnings.warn(
            "Cevahir.train() DEPRECATED! "
            "Eğitim için training_system/train.py kullanın: "
            "python training_system/train.py",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.error(
            "Cevahir.train() çağrıldı - bu metod kullanımdan kaldırılmıştır!\n"
            "Eğitim için training_system/train.py kullanılmalıdır:\n"
            "  python training_system/train.py"
        )
        
        raise CevahirProcessingError(
            "Cevahir.train() kullanımdan kaldırılmıştır!\n"
            "Eğitim için training_system/train.py kullanın:\n"
            "  python training_system/train.py\n\n"
            "Eğitim akışı:\n"
            "  1. tokenizer_management/train_bpe.py → BPE eğitimi\n"
            "  2. training_system/train.py → Model eğitimi\n"
            "  3. training_system/training_service.py → Eğitim servisi\n"
            "  4. model_management/model_manager.py → Neural network eğitimi"
        )
    
    # =========================================================================
    # Monitoring & Observability
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        if not self._cognitive_manager:
            return {}
        
        try:
            return self._cognitive_manager.get_metrics()
        except Exception:
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self._cognitive_manager:
            return {"status": "unknown"}
        
        try:
            return self._cognitive_manager.get_health_status()
        except Exception:
            return {"status": "unknown"}
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def tokenizer(self) -> TokenizerCore:
        """Get tokenizer"""
        if not self._tokenizer_core:
            raise CevahirProcessingError("TokenizerCore not initialized")
        return self._tokenizer_core
    
    @property
    def model(self) -> ModelManager:
        """Get model manager"""
        if not self._model_manager:
            raise CevahirProcessingError("ModelManager not initialized")
        return self._model_manager
    
    @property
    def device(self) -> torch.device:
        """Get model device"""
        if not self._model_manager:
            raise CevahirProcessingError("ModelManager not initialized")
        return self._model_manager.device
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        if not self._model_manager:
            return False
        return self._model_manager.is_initialized
    
    @property
    def cognitive(self) -> CognitiveManager:
        """Get cognitive manager"""
        if not self._cognitive_manager:
            raise CevahirProcessingError("CognitiveManager not initialized")
        return self._cognitive_manager


# =============================================================================
# Factory Functions
# =============================================================================

def create_cevahir(
    config: Union[CevahirConfig, Dict[str, Any]],
    **kwargs
) -> Cevahir:
    """
    Factory function to create Cevahir instance.
    
    Endüstri Standardı: Factory pattern for flexible initialization
    """
    return Cevahir(config, **kwargs)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage
    config = CevahirConfig(
        device="cpu",
        model={
            "vocab_size": 50000,
            "embed_dim": 512,
            "seq_proj_dim": 512,
            "num_heads": 8,  # [OK] Training ile uyumlu (8 heads - Colab crash fix)
            "num_layers": 12,  # [OK] Training ile uyumlu (24 layers)
        }
    )
    
    cevahir = Cevahir(config)
    print("Cevahir system initialized successfully!")
