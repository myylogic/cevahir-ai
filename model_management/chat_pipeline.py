# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: chat_pipeline.py
Modül: model_management
Görev: Chat Pipeline - Cevahir inference sistemi kullanarak chat pipeline.
       Model yükleme, inference işlemleri, tokenization, generation ve chat
       interface sağlar. Kullanıcı girdilerini işler ve model çıktılarını
       döndürür.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (chat pipeline)
- Design Patterns: Pipeline Pattern (chat pipeline)
- Endüstri Standartları: Chat interface best practices

KULLANIM:
- Chat interface için
- Model inference için
- Tokenization ve generation için
- Kullanıcı etkileşimi için

BAĞIMLILIKLAR:
- Cevahir: Model sınıfı
- CevahirConfig: Model config
- torch: PyTorch işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import sys
import os
import logging
from typing import Dict, Any, Optional

import torch

# PYTHONPATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model.cevahir import Cevahir, CevahirConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("chat_pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("ChatPipeline")

class ChatPipeline:
    """
    Chat pipeline using Cevahir inference system.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)

        # Device
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Model path: checkpoint veya cevahir_model.pth (eğitim config ile uyumlu olmalı)
        load_model_path = self.config.get("load_path")
        if not load_model_path:
            # Önce son checkpoint, yoksa cevahir_model.pth
            checkpoint_dir = os.path.join("saved_models", "checkpoints")
            if os.path.isdir(checkpoint_dir):
                import glob
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
                if checkpoints:
                    load_model_path = max(checkpoints, key=lambda p: int(p.split("_")[-1].replace(".pth", "")))
                    logger.info(f"Son checkpoint kullanılıyor: {load_model_path}")
            if not load_model_path or not os.path.exists(load_model_path or ""):
                load_model_path = os.path.join("saved_models", "cevahir_model.pth")

        # Checkpoint yoksa uyarı (eğitilmemiş model = boş/anlamsız cevaplar)
        if not os.path.exists(load_model_path or ""):
            logger.warning(
                "Model dosyası bulunamadı: %s — Eğitilmiş checkpoint yüklü değil, cevaplar rastgele/anlamsız olacak. "
                "Colab'ta eğittiyseniz saved_models/checkpoints veya cevahir_model.pth dosyasını bu klasöre kopyalayın.",
                load_model_path,
            )

        # vocab_size: tokenizer ile aynı olmalı (vocab.json boyutu); yoksa embedding uyumsuz → bozuk çıktı
        vocab_size_cfg = self.config.get("vocab_size")
        if vocab_size_cfg is None:
            try:
                vocab_path = self.config.get("vocab_path") or self.config.get("vocab_file") or "data/vocab_lib/vocab.json"
                if os.path.exists(vocab_path):
                    import json
                    with open(vocab_path, encoding="utf-8") as f:
                        vocab_size_cfg = len(json.load(f))
                    logger.info("vocab_size tokenizer ile eşlendi: %s", vocab_size_cfg)
            except Exception:
                pass
        if vocab_size_cfg is None:
            vocab_size_cfg = 60000  # Eğitimde kullanılan standart; 16125 eski/yanlış olabilir
        model_cfg = dict(self.config.get("model_override") or {})
        model_cfg.setdefault("vocab_size", vocab_size_cfg)
        # train.py ile aynı mimari olmalı (checkpoint yükleme); güncel: embed_dim=384, num_heads=6, num_layers=8
        model_cfg.setdefault("embed_dim", self.config.get("embed_dim", 384))
        model_cfg.setdefault("seq_proj_dim", self.config.get("seq_proj_dim", 384))
        model_cfg.setdefault("num_heads", self.config.get("num_heads", 6))
        model_cfg.setdefault("num_layers", self.config.get("num_layers", 8))
        model_cfg.setdefault("ffn_dim", None)
        model_cfg.setdefault("pre_norm", True)
        model_cfg.setdefault("causal_mask", True)
        model_cfg.setdefault("use_flash_attention", False)
        model_cfg.setdefault("pe_mode", "rope")
        model_cfg.setdefault("use_gradient_checkpointing", False)  # inference'da kapalı
        model_cfg.setdefault("tie_weights", True)
        model_cfg.setdefault("use_rmsnorm", True)
        model_cfg.setdefault("use_swiglu", True)
        model_cfg.setdefault("use_kv_cache", True)
        model_cfg.setdefault("max_cache_len", 2048)
        model_cfg.setdefault("use_advanced_checkpointing", False)
        model_cfg.setdefault("checkpointing_strategy", "selective")
        model_cfg.setdefault("quantization_type", "none")
        model_cfg.setdefault("use_moe", False)
        model_cfg.setdefault("num_experts", 8)
        model_cfg.setdefault("moe_top_k", 2)
        model_cfg.setdefault("use_tensorboard", False)

        cevahir_config = CevahirConfig(
            device=device,
            tokenizer={
                "vocab_path": self.config.get("vocab_path") or self.config.get("vocab_file") or "data/vocab_lib/vocab.json",
                "merges_path": self.config.get("merges_path") or self.config.get("merges_file") or "data/merges_lib/merges.txt",
                "data_dir": self.config.get("data_dir"),
                "use_gpu": device == "cuda",
                "batch_size": 32,
            },
            load_model_path=load_model_path,
            model=model_cfg,
        )

        try:
            logger.info("Cevahir sistemi başlatılıyor...")
            self.cevahir = Cevahir(cevahir_config)
            logger.info("[OK] Cevahir sistemi başarıyla başlatıldı!")
        except Exception as e:
            logger.error(f"Cevahir başlatılamadı: {e}", exc_info=True)
            raise

        # Generation parametreleri
        self.max_new_tokens = int(self.config.get("max_response_length", 128))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.top_p = float(self.config.get("top_p", 0.9))
        self.top_k = int(self.config.get("top_k", 50))
        self.repetition_penalty = float(self.config.get("repetition_penalty", 1.1))

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Generate response using Cevahir.generate()
        """
        try:
            if max_new_tokens is None:
                max_new_tokens = self.max_new_tokens
            
            # Direct generation (bypass cognitive layer for simplicity)
            generated_text = self.cevahir.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation hatası: {e}", exc_info=True)
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response.
        """
        try:
            return self.generate(user_input, max_new_tokens=self.max_new_tokens)
        except Exception as e:
            logger.error(f"Girdi işlenirken hata: {e}", exc_info=True)
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def run(self) -> None:
        print("\nCevahir AI'ya hoş geldin! (Çıkmak için 'kapat' veya 'çık')\n")
        while True:
            try:
                user_input = input("Sen: ").strip()
                if user_input.lower() in ("kapat", "çık"):
                    print("Cevahir: Görüşürüz! 👋")
                    break
                response = self.process_input(user_input)
                print(f"Cevahir: {response}\n")
            except KeyboardInterrupt:
                print("\nCevahir: Görüşürüz! 👋")
                break
            except Exception as e:
                logger.error(f"Sohbet döngüsünde hata: {e}", exc_info=True)
                print("Cevahir: Bir hata oluştu, lütfen tekrar deneyin. 😕")


def main() -> None:
    logger.info("ChatPipeline başlatılıyor...")

    config: Dict[str, Any] = {
        "data_dir": "education",
        "vocab_path": os.path.join("data", "vocab_lib", "vocab.json"),
        "merges_path": os.path.join("data", "merges_lib", "merges.txt"),

        "max_response_length": 80,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Colab epoch testi ile uyumlu + biraz daha tutarlı cevap için
        "temperature": 0.7,    # 0.7 = daha deterministik (0.75/1.0'dan); saçma token azalır
        "top_k": 40,           # 40 = sadece en güçlü tokenlar (50/100 yerine); anlamsız kelime azalır
        "top_p": 0.85,         # nucleus sampling (0.9'dan biraz sıkı)
        "repetition_penalty": 1.5,  # Colab ile aynı; tekrar azaltır (hava hava, ver ver vb.)
        "load_path": os.path.join("saved_models", "cevahir_model.pth"),
        # Eğitilen model mimarisi (train.py ile aynı; checkpoint tam yüklensin)
        "num_heads": 6,
        "num_layers": 8,
        "embed_dim": 384,
        "seq_proj_dim": 384,
    }

    try:
        pipeline = ChatPipeline(config)
        logger.info("[OK] ChatPipeline hazır! Sohbet başlatılıyor...")
        pipeline.run()
    except Exception as e:
        logger.error(f"ChatPipeline başlatılamadı: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
