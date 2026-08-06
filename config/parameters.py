# -*- coding: utf-8 -*-
"""
Global Parameters for Cevahir AI Engine
========================================

Endüstri Standardı: Merkezi yapılandırma yönetimi
- Tüm sistem genelinde kullanılan sabitler
- Environment variable override desteği
- Type-safe defaults
"""

import os
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).parent.parent.absolute()

# Device configuration (CUDA/CPU)
import torch
DEVICE = os.getenv("CEVAHIR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Logging path
LOGGING_PATH = os.getenv("CEVAHIR_LOGGING_PATH", str(BASE_DIR / "logs"))
os.makedirs(LOGGING_PATH, exist_ok=True)

# Model save path
MODEL_SAVE_PATH = os.getenv("CEVAHIR_MODEL_SAVE_PATH", str(BASE_DIR / "saved_models"))
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Data paths
DATA_PATH = os.getenv("CEVAHIR_DATA_PATH", str(BASE_DIR / "data"))
os.makedirs(DATA_PATH, exist_ok=True)

# Cache paths
CACHE_PATH = os.getenv("CEVAHIR_CACHE_PATH", str(BASE_DIR / "cache"))
os.makedirs(CACHE_PATH, exist_ok=True)

# Tokenizer paths
TOKENIZER_PATH = os.getenv("CEVAHIR_TOKENIZER_PATH", str(BASE_DIR / "tokenizer_management" / "bpe"))
os.makedirs(TOKENIZER_PATH, exist_ok=True)

# Database configuration
DATABASE_URL = os.getenv("CEVAHIR_DATABASE_URL", f"sqlite:///{BASE_DIR}/database/cevahir.db")
DATABASE_ECHO = os.getenv("CEVAHIR_DATABASE_ECHO", "false").lower() == "true"

# API Configuration
API_HOST = os.getenv("CEVAHIR_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("CEVAHIR_API_PORT", "8000"))
API_DEBUG = os.getenv("CEVAHIR_API_DEBUG", "false").lower() == "true"

# Security
SECRET_KEY = os.getenv("CEVAHIR_SECRET_KEY", "cevahir-ai-dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("CEVAHIR_JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("CEVAHIR_JWT_EXPIRATION_MINUTES", "1440"))  # 24 hours

# Rate limiting
RATE_LIMIT_DEFAULT = os.getenv("CEVAHIR_RATE_LIMIT_DEFAULT", "100 per minute")
RATE_LIMIT_STORAGE = os.getenv("CEVAHIR_RATE_LIMIT_STORAGE", "memory://")

# Cognitive layer
COGNITIVE_MEMORY_ENABLED = os.getenv("CEVAHIR_COGNITIVE_MEMORY", "true").lower() == "true"
COGNITIVE_TOOLS_ENABLED = os.getenv("CEVAHIR_COGNITIVE_TOOLS", "true").lower() == "true"
COGNITIVE_MAX_ITERATIONS = int(os.getenv("CEVAHIR_COGNITIVE_MAX_ITERATIONS", "10"))

# Training defaults
TRAINING_BATCH_SIZE = int(os.getenv("CEVAHIR_TRAINING_BATCH_SIZE", "32"))
TRAINING_LEARNING_RATE = float(os.getenv("CEVAHIR_TRAINING_LR", "1e-4"))
TRAINING_MAX_STEPS = int(os.getenv("CEVAHIR_TRAINING_MAX_STEPS", "100000"))

print(f"[CONFIG] Cevahir AI initialized on {DEVICE}")
print(f"[CONFIG] Logging: {LOGGING_PATH}")
print(f"[CONFIG] Models: {MODEL_SAVE_PATH}")
