# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: model_manager.py
Modül: model_management
Görev: Model Manager - Modeli başlatma, kaydetme, yükleme, güncelleme ve ileri-yayılım
       (forward/inference) işlemlerini merkezi olarak yöneten sınıf. SOLID prensipleri,
       test edilebilir küçük metotlar, esneklik (eğitim/inference modları, opsiyonel
       bileşenler) ve güvenlik (tip doğrulama, açık hata mesajları, dikkatli logging)
       sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (model yönetimi), test edilebilir küçük metotlar
- Design Patterns: Manager Pattern (model yönetimi)
- Endüstri Standartları: Model management best practices

KULLANIM:
- Model başlatma için
- Model kaydetme/yükleme için
- Model güncelleme için
- Forward/inference işlemleri için

BAĞIMLILIKLAR:
- ModelInitializer: Model başlatma
- ModelLoader: Model yükleme
- ModelSaver: Model kaydetme
- ModelUpdater: Model güncelleme
- torch: PyTorch işlemleri

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
import sys
import logging
from typing import Any, Dict, Optional, Tuple, Type, Union, List

import torch
from torch import nn
import torch.optim as optim

# --- Proje yolu ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- Logger (root'u bozmadan) ---
manager_logger = logging.getLogger("ModelManager")
if not manager_logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _h.setFormatter(_f)
    manager_logger.addHandler(_h)
    manager_logger.setLevel(logging.INFO)

# --- Yerel importlar ---
from model_management.model_initializer import ModelInitializer
from model_management.model_saver import ModelSaver
from model_management.model_updater import ModelUpdater

# --- Yeni ileri seviye modüller ---
from model_management.exceptions import (
    ModelNotInitializedError,
    ModelBuildError,
    ForwardError,
    OOMRecoveryError,
    CheckpointError,
)
from model_management.profiler import ModelProfiler
from model_management.health_monitor import ModelHealthMonitor, HealthReport

# Varsayılan model sınıfı (opsiyonel)
try:
    from src.neural_network import CevahirNeuralNetwork  # type: ignore
except Exception as e:
    manager_logger.warning(f"src.neural_network import edilemedi: {e}")
    CevahirNeuralNetwork = None  # type: ignore

# TensorBoard opsiyonel
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore


def _ensure(v: Any, name: str) -> Any:
    if v is None:
        raise ModelNotInitializedError(component=name)
    return v


class ModelManager:
    """
    Modelle ilgili tüm işlemleri yöneten çekirdek sınıf.

    Yetenekler:
      - build_model / build_optimizer / build_criterion / build_scheduler
      - initialize (zincirleme başlatma)
      - train/eval mod yönetimi
      - forward (inference destekli)
      - predict (top-k, logits/softmax)
      - save/load (toplu checkpoint)
      - update (freeze/unfreeze, lr vs.)
      - TensorBoard: tek yerden writer yönetimi, modele ve eğitime enjekte
    
    V-2/V-3/V-4 Mimarisi Desteği:
      - ModelInitializer otomatik olarak model sınıfının __init__ imzasını okur
      - Config'teki tüm parametreler (V-2/V-3/V-4) otomatik geçirilir
      - V-2 parametreleri: num_layers, ffn_dim, pre_norm, causal_mask
      - V-3 parametreleri: use_flash_attention, pe_mode, use_gradient_checkpointing, tie_weights
      - V-4 parametreleri: use_rope, use_rmsnorm, use_swiglu, use_kv_cache, use_advanced_checkpointing, 
                           quantization_type, use_moe, num_experts, moe_top_k
      - Tüm parametreler config'te varsa otomatik olarak kullanılır
      - Backward compatibility: Tüm parametreler opsiyonel (default değerler mevcut)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_class: Optional[Type[nn.Module]] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        initializer: Type[ModelInitializer] = ModelInitializer,
        saver: Type[ModelSaver] = ModelSaver,
        updater: Type[ModelUpdater] = ModelUpdater,
        # Multimodal desteği
        tokenizer: Optional[Any] = None,
        audio_processor: Optional[Any] = None,
        vision_processor: Optional[Any] = None,
    ) -> None:
        self.config = dict(config)  # koruyucu kopya
        self.model_class = model_class or CevahirNeuralNetwork

        # cihaz - Colab için güçlendirilmiş kontrol
        if device is not None:
            self._device = torch.device(device)
        else:
            dev = str(self.config.get("device", "")).strip().lower()
            if dev == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
                # Colab için GPU'yu aktif et
                try:
                    torch.cuda.set_device(0)
                    manager_logger.info(f"[OK] GPU[0] aktif edildi (ModelManager)")
                except Exception as e:
                    manager_logger.warning(f"⚠️ GPU aktif edilemedi: {e}")
            elif dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif dev in {"cpu", ""}:
                self._device = torch.device("cpu")
            else:
                # CUDA varsa kesinlikle kullan
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    try:
                        torch.cuda.set_device(0)
                        manager_logger.info(f"[OK] GPU[0] otomatik aktif edildi (ModelManager)")
                    except Exception:
                        pass
                else:
                    self._device = torch.device("cpu")
                    manager_logger.warning("⚠️ GPU kullanılamıyor, CPU modunda çalışılacak (ModelManager)")

        # Bileşenler
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        
        # Multimodal bileşenler
        self.tokenizer: Optional[Any] = tokenizer
        self.audio_processor: Optional[Any] = audio_processor
        self.vision_processor: Optional[Any] = vision_processor

        # Yardımcı sınıflar
        self._Initializer = initializer
        self._Saver = saver
        self._Updater = updater

        # TensorBoard yönetimi
        self._tb_enabled: bool = bool(self.config.get("use_tensorboard", False))
        self._tb_log_dir: str = str(self.config.get("tb_log_dir", os.path.join("runs", "cevahir")))
        self._tb_writer: Optional[Any] = None  # SummaryWriter veya benzeri (fake) obje
        self._tb_log_every_n: int = int(self.config.get("tb_log_every_n", 10))
        self._tb_log_histograms: bool = bool(self.config.get("tb_log_histograms", False))
        self._tb_log_attention_image: bool = bool(self.config.get("tb_log_attention_image", False))

    # ---------------------------------------------------------------------
    # Kurulum / Başlatma
    # ---------------------------------------------------------------------
    def build_model(self) -> nn.Module:
        # [OK] Model build edildikten sonra writer'ı attach et
        """
        Modeli oluşturur ve cihaza taşır.
        
        V-2/V-3/V-4 Mimarisi:
          - Config'teki tüm parametreler (V-2/V-3/V-4) ModelInitializer tarafından otomatik olarak 
            model sınıfına geçirilir.
          - ModelInitializer, model sınıfının __init__ imzasını okuyarak uygun parametreleri filtreler.
          - V-2 parametreleri: num_layers, ffn_dim, pre_norm, causal_mask
          - V-3 parametreleri: use_flash_attention, pe_mode, use_gradient_checkpointing, tie_weights
          - V-4 parametreleri: use_rope, use_rmsnorm, use_swiglu, use_kv_cache, use_advanced_checkpointing,
                               quantization_type, use_moe, num_experts, moe_top_k
          - Tüm parametreler config'te varsa otomatik kullanılır, yoksa default değerler kullanılır.
          - V4 özellikleri (RoPE, RMSNorm, SwiGLU, KV Cache, Advanced Checkpointing, Quantization, MoE)
            config'te belirtildiğinde otomatik olarak aktif hale gelir.
        """
        if self.model_class is None:
            raise RuntimeError("model_class belirtilmeli veya src.neural_network import edilebilir olmalı.")
        model = self._Initializer.build_model(self.model_class, {**self.config, "device": str(self._device)})
        self.model = model

        # Model oluşturulur oluşturulmaz TB writer bağlı ise modele enjekte edelim
        if self._tb_writer is not None:
            self._attach_writer_to_model(self._tb_writer)

        # V-2/V-3 parametrelerini logla (varsa)
        v2_params = {}
        for param in ["num_layers", "ffn_dim", "pre_norm", "causal_mask"]:
            if param in self.config:
                v2_params[param] = self.config[param]
        
        # [OK] V3 parametrelerini logla (varsa)
        v3_params = {}
        for param in ["use_flash_attention", "pe_mode", "use_gradient_checkpointing", "tie_weights"]:
            if param in self.config:
                v3_params[param] = self.config[param]
        
        # [OK] V4 parametrelerini logla (varsa)
        v4_params = {}
        for param in ["use_rope", "use_rmsnorm", "use_swiglu", "use_kv_cache", "use_advanced_checkpointing", 
                      "quantization_type", "use_moe", "num_experts", "moe_top_k"]:
            if param in self.config:
                v4_params[param] = self.config[param]
        
        if v2_params or v3_params or v4_params:
            params_str = ""
            if v2_params:
                params_str += f"V-2 params: {v2_params}"
            if v3_params:
                if params_str:
                    params_str += ", "
                params_str += f"V-3 params: {v3_params}"
            if v4_params:
                if params_str:
                    params_str += ", "
                params_str += f"V-4 params: {v4_params}"
            manager_logger.info(
                f"Model oluşturuldu (V-2/V-3/V-4): {type(model).__name__} -> {self._device}, {params_str}"
            )
        else:
            manager_logger.info(f"Model oluşturuldu ve cihaza taşındı: {type(model).__name__} -> {self._device}")

        # ── Otomatik Profil Raporu ────────────────────────────────────────────
        # Model oluşturulur oluşturulmaz parametre sayısı ve model boyutu loglanır.
        # FLOP tahmini de bu noktada yapılır (seq_len config'ten alınır).
        try:
            _seq = int(self.config.get("max_seq_length", 512))
            _report = ModelProfiler.full_report(
                model,
                seq_len=_seq,
                batch_size=1,
                run_timing=False,   # Eğitim başlangıcında timing yapma (zaman kaybı)
            )
            # Parametre sayısını config'e yaz (checkpoint meta verisi için)
            self.config["_total_params"] = _report["params"].total
            self.config["_model_size_mb"] = _report["size_mb"]
        except Exception as prof_exc:
            manager_logger.debug(f"Profil raporu oluşturulamadı: {prof_exc}")

        return model

    def build_optimizer(self) -> optim.Optimizer:
        model = _ensure(self.model, "model")
        self.optimizer = self._Initializer.initialize_optimizer(model, self.config)
        return self.optimizer

    def build_criterion(self) -> nn.Module:
        self.criterion = self._Initializer.initialize_criterion(self.config)
        return self.criterion

    def build_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        optimizer = _ensure(self.optimizer, "optimizer")
        self.scheduler = self._Initializer.initialize_scheduler(optimizer, self.config)
        return self.scheduler

    def initialize(
        self,
        *,
        build_optimizer: bool = True,
        build_criterion: bool = True,
        build_scheduler: bool = True,
        reset: bool = False,
    ) -> "ModelManager":
        if reset:
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.scheduler = None
            # Reset sırasında writer'ı da sıfırla
            self._tb_writer = None

        # TensorBoard writer'ı sıfırlama - sadece reset=True olduğunda sıfırlanmalı
        # Writer önceden attach edilmişse korunmalı

        if self.model is None:
            self.build_model()

        # writer önceden oluşturulduysa modele bağla
        if self._tb_writer is not None:
            self._attach_writer_to_model(self._tb_writer)

        if build_optimizer and self.optimizer is None:
            self.build_optimizer()

        if build_criterion and self.criterion is None:
            self.build_criterion()

        if build_scheduler and self.scheduler is None:
            self.build_scheduler()

        return self

    # ---------------------------------------------------------------------
    # TensorBoard API
    # ---------------------------------------------------------------------
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
        TB ayarlarını günceller ve writer'ı bağlar. Dışarıdan writer geçebilirsin.
        """
        if enable is not None:
            self._tb_enabled = bool(enable)
        if log_dir is not None:
            self._tb_log_dir = str(log_dir)
        if log_every_n is not None:
            self._tb_log_every_n = int(log_every_n)
        if log_histograms is not None:
            self._tb_log_histograms = bool(log_histograms)
        if log_attention_image is not None:
            self._tb_log_attention_image = bool(log_attention_image)

        if writer is not None:
            self.attach_tb_writer(writer)
        elif self._tb_enabled and self._tb_writer is None and SummaryWriter is not None:
            # iç writer oluştur
            os.makedirs(self._tb_log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=self._tb_log_dir)  # type: ignore
            self._attach_writer_to_model(self._tb_writer)

    def attach_tb_writer(self, writer: Any) -> None:
        """
        Dışarıdan (ör. testlerde FakeWriter) bir writer enjekte et.
        """
        self._tb_writer = writer
        self._tb_enabled = True
        # modele aktar
        self._attach_writer_to_model(writer)

    def detach_tb_writer(self) -> None:
        """
        Writer ile ilişkiyi kes (model + manager).
        """
        if self.model is not None and hasattr(self.model, "set_tb_writer"):
            try:
                self.model.set_tb_writer(None)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._tb_writer = None

    def close_tensorboard(self) -> None:
        """
        İç writer'ı kapatır (dış writer ise dokunmaz).
        """
        if self._tb_writer is not None:
            # SummaryWriter veya close() metodu olan herhangi bir writer için
            if SummaryWriter is not None and isinstance(self._tb_writer, SummaryWriter):  # type: ignore[arg-type]
                try:
                    self._tb_writer.flush()
                    self._tb_writer.close()
                except Exception:
                    pass
            elif hasattr(self._tb_writer, "close"):
                # FakeWriter gibi close() metodu olan diğer writer'lar için
                try:
                    self._tb_writer.close()
                except Exception:
                    pass
        self._tb_writer = None

    def _attach_writer_to_model(self, writer: Any) -> None:
        model = self.model
        if model is None:
            return

        # 1) writer'ı enjekte et
        if hasattr(model, "set_tb_writer"):
            try:
                model.set_tb_writer(writer)  # harici writer
            except Exception as e:
                manager_logger.warning(f"Model'e TB writer set edilemedi: {e}")
        else:
            # set_tb_writer yoksa direkt _tb_writer attribute'unu set et
            try:
                model._tb_writer = writer
            except Exception as e:
                manager_logger.warning(f"Model'e TB writer attribute set edilemedi: {e}")

        # 2) TB konfigürasyonunu doğru isimlerle aktar
        try:
            if hasattr(model, "set_tb_config"):
                model.set_tb_config(
                    log_every_n=self._tb_log_every_n,
                    log_histograms=self._tb_log_histograms,
                    log_attention_image=self._tb_log_attention_image,
                )
            else:
                # Geriye dönük: özel alanları da deneriz
                for attr, val in [
                    ("_tb_log_every_n", self._tb_log_every_n),
                    ("_tb_log_histograms", self._tb_log_histograms),
                    ("_tb_log_attention_image", self._tb_log_attention_image),
                ]:
                    if hasattr(model, attr):
                        setattr(model, attr, val)
        except Exception:
            pass

    def get_tb_writer(self) -> Optional[Any]:
        return self._tb_writer
    # ---------------------------------------------------------------------
    # Mod Yönetimi
    # ---------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_initialized(self) -> bool:
        return self.model is not None

    def train_mode(self) -> None:
        model = _ensure(self.model, "model")
        model.train(True)

    def eval_mode(self) -> None:
        model = _ensure(self.model, "model")
        model.train(False)

    # ---------------------------------------------------------------------
    # Forward / Predict
    # ---------------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        inference: Optional[bool] = None,
        return_aux: bool = True,
        expected_vocab: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[bool] = None,
        # [OK] V4: KV Cache parametreleri (endüstri standardı: GPT-4, Claude, Gemini)
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model = _ensure(self.model, "model")

        prev_mode = model.training
        if inference is True:
            model.train(False)
        elif inference is False:
            model.train(True)

        use_no_grad = (inference is True)

        try:
            with torch.set_grad_enabled(not use_no_grad):
                # Input'u device'a taşı
                inputs_device = inputs.to(self.device)
                # Mask varsa device'a taşı ve shape'i düzelt
                if mask is not None:
                    mask = mask.to(self.device)
                    # Padding mask (B, T) → Attention mask (B, T, T) dönüşümü
                    # Endüstri standardı: GPT, BERT padding mask'i otomatik attention mask'e çevirir
                    if mask.dim() == 2:
                        batch_size, seq_len = mask.shape
                        # Padding mask: True = valid token, False = padding token
                        # Attention mask: True = mask (engelle), False = allow
                        # Her batch için: padding token'lara attention verilmemeli
                        attention_mask = torch.zeros(batch_size, seq_len, seq_len, device=mask.device, dtype=torch.bool)
                        for i in range(batch_size):
                            # Valid token sayısını bul
                            if mask.dtype == torch.bool:
                                valid_len = int(mask[i].sum().item())
                            else:
                                valid_len = int((mask[i] > 0.5).sum().item())
                            # Valid positions'da attention allow (False), padding'de mask (True)
                            # Her query position için, padding key positions'ı mask'le
                            attention_mask[i, :, :valid_len] = False  # Valid keys: allow
                            if valid_len < seq_len:
                                attention_mask[i, :, valid_len:] = True  # Padding keys: mask
                        mask = attention_mask
                # Model forward (mask, causal_mask ve KV Cache parametreleri)
                # [OK] V4: KV Cache desteği eklendi
                # [OK] Tüm parametreler Neural Network'un forward metoduna geçirilmeli
                # Neural Network.forward() imzası:
                #   forward(x, mask=None, causal_mask=None, use_cache=False, cache_position=None)
                
                forward_params = {}
                
                # Mask parametresi - Neural Network'te var, her zaman geçir
                if hasattr(model, "forward") and "mask" in model.forward.__code__.co_varnames:
                    forward_params["mask"] = mask  # None olsa bile geçir (default None)
                
                # Causal mask parametresi - Neural Network'te var, her zaman geçir
                if hasattr(model, "forward") and "causal_mask" in model.forward.__code__.co_varnames:
                    forward_params["causal_mask"] = causal_mask  # None olsa bile geçir (default None)
                
                # [OK] V4: KV Cache parametreleri - Neural Network'te var, her zaman geçir
                if hasattr(model, "forward") and "use_cache" in model.forward.__code__.co_varnames:
                    forward_params["use_cache"] = use_cache  # False default, ama her zaman geçir
                
                if hasattr(model, "forward") and "cache_position" in model.forward.__code__.co_varnames:
                    # Cache position'ı device'a taşı (None olsa bile)
                    if cache_position is not None:
                        cache_position = cache_position.to(self.device)
                    forward_params["cache_position"] = cache_position  # None olsa bile geçir (default None)
                
                # Forward çağrısı - Tüm parametreleri geçir
                # Neural Network'un forward metodunda default değerler var, bu yüzden
                # tüm parametreleri geçirmek güvenli (None olsa bile)
                if forward_params:
                    outputs = model(inputs_device, **forward_params)
                else:
                    # Fallback: Eğer model.forward() parametreleri desteklemiyorsa
                    outputs = model(inputs_device)

            logits: torch.Tensor
            aux: Optional[torch.Tensor] = None

            if isinstance(outputs, tuple):
                logits = outputs[0]
                if return_aux:
                    aux = outputs[1] if isinstance(outputs[1], torch.Tensor) else None
            else:
                logits = outputs

            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Model çıktısı tensör olmalı; gelen={type(logits)}")

            vocab = expected_vocab if expected_vocab is not None else int(self.config.get("vocab_size", 0) or 0)
            if vocab > 0 and logits.ndim >= 3 and logits.shape[-1] != vocab:
                manager_logger.warning(
                    f"[FORWARD] Son eksen vocab ile uyuşmuyor! beklenen={vocab}, gelen={logits.shape[-1]}"
                )

            # [OK] V4: KV Cache kullanımı logla
            cache_info = ""
            if use_cache:
                cache_info = f", KV Cache: ON"
                if cache_position is not None:
                    cache_info += f" (position={cache_position.shape})"

            manager_logger.debug(
                f"[FORWARD] Tamam. logits={tuple(logits.shape)}"
                + (f", aux={tuple(aux.shape)}" if isinstance(aux, torch.Tensor) else ", aux=None")
                + cache_info
            )
            return logits, aux

        except torch.cuda.OutOfMemoryError as oom_err:
            # ── OOM Kurtarma ──────────────────────────────────────────────────
            # Strateji:
            #   1. Cache'i temizle ve bilgilendirici hata fırlat.
            #   Model eğitim döngüsü bu exception'ı yakalayıp batch'i atlayabilir.
            manager_logger.error(
                f"[FORWARD] ❌ CUDA OOM! "
                f"Tahsis={torch.cuda.memory_allocated() / 1e9:.2f} GB — "
                f"cache temizleniyor..."
            )
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            raise OOMRecoveryError(
                allocated_gb=allocated_gb,
                required_gb=0.0,   # torch OOM mesajından kesin değer çıkarmak güç
                recovery_attempted=True,
            ) from oom_err

        except Exception as e:
            manager_logger.error(f"[FORWARD] Hata: {e}", exc_info=True)
            raise ForwardError(reason=str(e), input_shape=tuple(inputs.shape)) from e
        finally:
            if inference is not None and model.training != prev_mode:
                model.train(prev_mode)

    def clear_kv_cache(self) -> None:
        """Tüm layer'lardaki KV cache'i temizler. Her yeni generate() öncesi çağrılmalı."""
        model = self.model
        if model is not None and hasattr(model, "clear_kv_cache"):
            model.clear_kv_cache()

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        *,
        topk: int = 1,
        apply_softmax: bool = True,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        logits, _ = self.forward(inputs, inference=True, return_aux=False)
        out: Dict[str, Any] = {}

        probs_or_logits = torch.softmax(logits, dim=-1) if apply_softmax else logits
        if apply_softmax:
            out["probs"] = probs_or_logits

        if isinstance(topk, int) and topk > 0:
            k = min(int(topk), int(probs_or_logits.shape[-1]))
            values, indices = torch.topk(probs_or_logits, k=k, dim=-1)
            out["topk_values"] = values
            out["topk_indices"] = indices

        if return_logits:
            out["logits"] = logits

        return out

    # ---------------------------------------------------------------------
    # Kaydet / Yükle
    # ---------------------------------------------------------------------
    def save(
        self,
        save_path: Optional[str] = None,
        *,
        epoch: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        model = _ensure(self.model, "model")
        optimizer = self.optimizer
        scheduler = self.scheduler

        if save_path is None:
            save_path = os.path.join(os.getcwd(), "saved_models", "test_models", "cevahir_model.pth")
        if epoch is None:
            epoch = int(self.config.get("current_epoch", 0))

        self.config["current_epoch"] = epoch
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ModelSaver.save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            additional_info={**(additional_info or {}), "config": self.config, "epoch": epoch},
            save_dir=os.path.dirname(save_path),
            model_name=os.path.basename(save_path),
        )
        manager_logger.info(f"Model kaydedildi: {save_path} (epoch={epoch})")
        return save_path

    def load(
        self,
        load_path: Optional[str] = None,
        *,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
        weights_only: Optional[bool] = None,
    ) -> None:
        if load_path is None:
            load_path = os.path.join("saved_models", "cevahir_model.pth")
        if map_location is None:
            map_location = self.device

        if self.model is None:
            self.build_model()
        if self.optimizer is None:
            self.build_optimizer()
        if self.scheduler is None:
            try:
                self.build_scheduler()
            except Exception:
                pass

        # PyTorch sürümlerinde weights_only opsiyonel olabilir
        load_kwargs = {"map_location": map_location}
        try:
            if weights_only is not None:
                checkpoint = torch.load(load_path, weights_only=bool(weights_only), **load_kwargs)  # type: ignore
            else:
                checkpoint = torch.load(load_path, **load_kwargs)
        except TypeError:
            checkpoint = torch.load(load_path, **load_kwargs)

        try:
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"], strict=strict)

                if self.optimizer is not None and checkpoint.get("optimizer_state") is not None and not weights_only:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    # [OK] Learning rate'i config'ten güncelle (checkpoint'teki eski LR'yi override et)
                    new_lr = self.config.get("learning_rate")
                    if new_lr is not None:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = float(new_lr)
                        manager_logger.info(f"Learning rate güncellendi: {new_lr} (checkpoint'teki eski LR override edildi)")

                if self.scheduler is not None and checkpoint.get("scheduler_state") is not None and not weights_only:
                    try:
                        self.scheduler.load_state_dict(checkpoint["scheduler_state"])  # type: ignore
                    except Exception as e:
                        manager_logger.warning(f"Scheduler state yüklenemedi: {e}")

                add_info = checkpoint.get("additional_info") or {}
                if isinstance(add_info, dict):
                    self.config.update(add_info.get("config", {}))
                    # [OK] Epoch bilgisini additional_info'dan veya direkt checkpoint'ten al
                    epoch_from_add_info = add_info.get("epoch")
                    if epoch_from_add_info is not None:
                        self.config["current_epoch"] = int(epoch_from_add_info)
                    else:
                        self.config["current_epoch"] = int(self.config.get("current_epoch", 0))

            elif isinstance(checkpoint, dict):
                state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
                if state is None:
                    raise ValueError("Checkpoint içinde model ağırlıkları bulunamadı (state_dict/model_state_dict).")
                self.model.load_state_dict(state, strict=strict)

                if self.optimizer is not None and "optimizer_state_dict" in checkpoint and not weights_only:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        # [OK] Learning rate'i config'ten güncelle (checkpoint'teki eski LR'yi override et)
                        new_lr = self.config.get("learning_rate")
                        if new_lr is not None:
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = float(new_lr)
                            manager_logger.info(f"Learning rate güncellendi: {new_lr} (checkpoint'teki eski LR override edildi)")
                    except (ValueError, KeyError, RuntimeError) as e:
                        manager_logger.warning(f"⚠️ Optimizer state dict yüklenemedi: {e}")
                        manager_logger.warning("⚠️ Model weights yüklendi, ancak optimizer state atlandı (yeniden eğitim gerekebilir)")
                        # Optimizer state dict yüklenemese bile model weights yüklendi, devam et
                if self.scheduler is not None and "scheduler_state_dict" in checkpoint and not weights_only:
                    try:
                        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore
                    except Exception as e:
                        manager_logger.warning(f"Scheduler state yüklenemedi: {e}")

                if "config" in checkpoint:
                    self.config.update(checkpoint["config"])
                # [OK] Epoch bilgisini checkpoint'ten al (metadata veya direkt)
                epoch_value = checkpoint.get("epoch")
                if epoch_value is not None:
                    self.config["current_epoch"] = int(epoch_value)
                else:
                    self.config["current_epoch"] = int(self.config.get("current_epoch", 0))

            elif isinstance(checkpoint, nn.Module):
                self.model = checkpoint.to(self.device)
            else:
                raise RuntimeError("Desteklenmeyen checkpoint formatı.")

            manager_logger.info(f"Model yüklendi: {load_path} (epoch={self.config.get('current_epoch', 0)})")

        except FileNotFoundError:
            manager_logger.error(f"Model dosyası bulunamadı: {load_path}")
            raise
        except Exception as e:
            manager_logger.error(f"Model yüklenemedi: {e}", exc_info=True)
            raise RuntimeError("Model yükleme işlemi sırasında hata oluştu.") from e


    # ---------------------------------------------------------------------
    # Güncelleme
    # ---------------------------------------------------------------------
    def update(self, update_params: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, List[str]]:
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        pre_report = {"model": [], "optimizer": [], "scheduler": []}

        if model and isinstance(update_params, dict) and "model" in update_params:
            mops = dict(update_params.get("model") or {})
            freeze_list = mops.get("freeze") or []
            unfreeze_list = mops.get("unfreeze") or []

            if any(p in (".*", "*", "__ALL__") for p in freeze_list):
                if dry_run:
                    pre_report["model"].append("freeze:*")
                else:
                    for p in model.parameters():
                        p.requires_grad = False
                    pre_report["model"].append("freeze:*")
                mops["freeze"] = []
            if any(p in (".*", "*", "__ALL__") for p in unfreeze_list):
                if dry_run:
                    pre_report["model"].append("unfreeze:*")
                else:
                    for p in model.parameters():
                        p.requires_grad = True
                    pre_report["model"].append("unfreeze:*")
                mops["unfreeze"] = []
            update_params = dict(update_params)
            update_params["model"] = mops

        report = self._Updater.bulk_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            update_params=update_params,
            dry_run=dry_run,
            filter_frozen_params=True,
        )

        try:
            as_dict = report.as_dict()
        except Exception:
            as_dict = dict(report) if isinstance(report, dict) else {}

        for k in pre_report:
            as_dict.setdefault(k, [])
            as_dict[k] = list(dict.fromkeys(pre_report[k] + as_dict[k]))

        manager_logger.info(f"Güncelleme tamamlandı. Rapor: {as_dict}")
        return as_dict

    def freeze(self, patterns: Union[str, List[str]]) -> Dict[str, List[str]]:
        pats = patterns if isinstance(patterns, list) else [patterns]
        return self.update({"model": {"freeze": pats}})

    def unfreeze(self, patterns: Union[str, List[str]]) -> Dict[str, List[str]]:
        pats = patterns if isinstance(patterns, list) else [patterns]
        return self.update({"model": {"unfreeze": pats}})

    # ---------------------------------------------------------------------
    # Multimodal API
    # ---------------------------------------------------------------------
    
    def process_audio(self, audio_data: bytes) -> str:
        """Ses verisini işle ve metne çevir"""
        if self.audio_processor is None:
            return "Ses işleme modülü bulunamadı."
        
        try:
            return self.audio_processor.audio_to_text(audio_data)
        except Exception as e:
            manager_logger.warning(f"Ses işleme hatası: {e}")
            return "Ses işlenemedi."
    
    def process_image(self, image_data: bytes) -> str:
        """Görüntü verisini işle ve metne çevir"""
        if self.vision_processor is None:
            return "Görüntü işleme modülü bulunamadı."
        
        try:
            return self.vision_processor.image_to_text(image_data)
        except Exception as e:
            manager_logger.warning(f"Görüntü işleme hatası: {e}")
            return "Görüntü işlenemedi."
    
    def process_multimodal(self, text: str = None, audio: bytes = None, image: bytes = None) -> str:
        """Çok modaliteli veriyi işle"""
        processed_parts = []
        
        # Text processing
        if text:
            processed_parts.append(f"Metin: {text}")
        
        # Audio processing
        if audio:
            audio_text = self.process_audio(audio)
            processed_parts.append(f"Ses: {audio_text}")
        
        # Image processing
        if image:
            image_text = self.process_image(image)
            processed_parts.append(f"Görüntü: {image_text}")
        
        if not processed_parts:
            return "Hiçbir veri işlenemedi."
        
        return " | ".join(processed_parts)
    
    def entropy_estimate(self, text: str) -> float:
        """Metin entropisini tahmin et"""
        if not text:
            return 0.5
        
        # Basit entropi hesaplama
        unique_chars = len(set(text))
        total_chars = len(text)
        if total_chars == 0:
            return 0.5
        
        entropy = unique_chars / total_chars
        return max(0.1, min(1.0, entropy))
    
    def generate(self, prompt: str, decoding_cfg=None) -> str:
        """Metin üretimi"""
        if self.model is None:
            return "Model henüz yüklenmedi."
        
        try:
            # Tokenize
            if self.tokenizer:
                tokens, token_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor([token_ids], dtype=torch.long)
            else:
                return "Tokenizer bulunamadı."
            
            # Generate with proper decoding
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Simple argmax for now - model needs more training
                predicted_ids = torch.argmax(output, dim=-1)[0]
                response = self.tokenizer.decode(predicted_ids.tolist())
            
            return response
            
        except Exception as e:
            return f"Üretim hatası: {e}"
    
    def score(self, prompt: str, candidate: str) -> float:
        """Metin skorlama"""
        # Basit skorlama - uzunluk bazlı
        return float(len(candidate)) / max(1, len(prompt))

    # ---------------------------------------------------------------------
    # Profiler & Health Monitor API
    # ---------------------------------------------------------------------

    def profile(
        self,
        *,
        seq_len: int = 512,
        batch_size: int = 1,
        run_timing: bool = False,
    ) -> Dict[str, Any]:
        """
        Model üzerinde tam profil raporu çalıştırır.
        build_model() çağrısında otomatik çalışır; ek çağrı ayrıntılı analiz için.

        Returns:
            {
              "params":  ParamStats,
              "memory":  MemorySnapshot,
              "flops":   FlopEstimate,
              "timing":  TimingResult | None,
              "size_mb": float,
            }
        """
        model = _ensure(self.model, "model")
        return ModelProfiler.full_report(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            device=str(self._device),
            run_timing=run_timing,
        )

    def health_check(
        self,
        sample_input: Optional[torch.Tensor] = None,
        *,
        check_gradients: bool = True,
        check_weights: bool = True,
        check_attention: bool = True,
        raise_on_critical: bool = False,
    ) -> HealthReport:
        """
        Model sağlık kontrolü — gradient, ağırlık dağılımı ve attention entropisi.

        Tipik kullanım:
            # Her N epoch'ta bir:
            report = manager.health_check()
            if not report.is_healthy:
                logger.warning(report.summary())

        Args:
            sample_input    : Attention entropy için örnek input (opsiyonel).
            check_gradients : Gradient akışını denetle.
            check_weights   : Ağırlık dağılımını denetle.
            check_attention : Attention entropisini denetle.
            raise_on_critical: CRITICAL durumda HealthCheckError fırlat.

        Returns:
            HealthReport veri sınıfı.
        """
        model = _ensure(self.model, "model")
        return ModelHealthMonitor.full_health_check(
            model,
            sample_input=sample_input,
            check_gradients=check_gradients,
            check_weights=check_weights,
            check_attention=check_attention,
            log=True,
            raise_on_critical=raise_on_critical,
        )

    def quick_gradient_check(self) -> Tuple[bool, str]:
        """
        Her batch sonrası hızlı NaN/Inf gradient kontrolü.
        True döndürürse batch güvenli; False ise batch'i atla.
        """
        model = _ensure(self.model, "model")
        return ModelHealthMonitor.quick_gradient_check(model)

    def log_gradient_norms(self, step: int, *, top_n: int = 10) -> Dict[str, float]:
        """
        Tüm katmanların gradient normlarını loglar ve opsiyonel TensorBoard'a yazar.
        Epoch sonu debug için idealdir.
        """
        model = _ensure(self.model, "model")
        return ModelHealthMonitor.log_gradient_norms(
            model, step, tb_writer=self._tb_writer, top_n=top_n
        )

    # ---------------------------------------------------------------------
    # Dunder Metodları (Pythonic API)
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:
        model_name = type(self.model).__name__ if self.model is not None else "None"
        total_params = self.config.get("_total_params", "?")
        size_mb = self.config.get("_model_size_mb", "?")
        param_str = (
            f"{total_params / 1e6:.1f}M" if isinstance(total_params, int) else str(total_params)
        )
        size_str = (
            f"{size_mb:.0f} MB" if isinstance(size_mb, float) else str(size_mb)
        )
        return (
            f"ModelManager(\n"
            f"  model      = {model_name} [{param_str} params, {size_str}]\n"
            f"  device     = {self._device}\n"
            f"  optimizer  = {type(self.optimizer).__name__ if self.optimizer else 'None'}\n"
            f"  scheduler  = {type(self.scheduler).__name__ if self.scheduler else 'None'}\n"
            f"  tb_enabled = {self._tb_enabled}\n"
            f")"
        )

    def __enter__(self) -> "ModelManager":
        """Context manager girişi — with ModelManager(cfg) as mm: ..."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager çıkışı — TensorBoard writer'ı otomatik kapat."""
        self.close_tensorboard()
        if exc_type is not None:
            manager_logger.error(
                f"ModelManager context manager'dan hata ile çıkıldı: "
                f"{exc_type.__name__}: {exc_val}"
            )
        # False → exception'ı yukarı ilet (bastırma)
        return None
