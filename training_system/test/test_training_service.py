# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import tempfile
import torch
import pytest
import numpy as np

# training_service import edilebilsin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training_system.training_service import TrainingService


# ================================
# Yardımcılar
# ================================
def _write_txt(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json_qa(path: str, qa_list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)


def _vocab(service: TrainingService):
    return service.tokenizer_core.get_vocab()


def _vocab_size(service: TrainingService) -> int:
    return len(_vocab(service))


def _pad_id(service: TrainingService) -> int:
    vocab = _vocab(service)
    val = vocab.get("<PAD>", 0)
    if isinstance(val, dict):
        return int(val.get("id", 0))
    return int(val or 0)


def _first_batch_max(loader) -> int:
    for xb, yb in loader:
        return max(int(xb.max().cpu()), int(yb.max().cpu()))
    return -1


def _output_last_dim_from_predict(service: TrainingService, seq_len: int = 8) -> int:
    pad = _pad_id(service)
    dummy = torch.full((1, seq_len), int(pad), dtype=torch.long)
    out = service.predict(dummy)
    if out is None:
        return 0
    if isinstance(out, tuple):
        out = out[0]
    # Beklenen genellikle (B, T, V)
    if hasattr(out, "shape") and len(out.shape) >= 3:
        return int(out.shape[-1])
    # Bazı varyantlar (B, V) dönebilir
    if hasattr(out, "shape") and len(out.shape) >= 2:
        return int(out.shape[-1])
    return 0


# ================================
# PyTest fixture: geçici ortam + servis
# ================================
@pytest.fixture(scope="module")
def tmp_env_and_service():
    """
    Geçici bir data dizini kurar, küçük bir korpus yazar ve
    TrainingService'i bu dizinle başlatır.
    """
    tmp = tempfile.mkdtemp(prefix="ts_oob_")
    try:
        data_dir = os.path.join(tmp, "education")
        os.makedirs(data_dir, exist_ok=True)

        # Küçük korpus (text + QA-json)
        _write_txt(
            os.path.join(data_dir, "a.txt"),
            "Merhaba dünya.\nBugün hava çok güzel.\nTransformer denemesi yapıyoruz."
        )
        _write_json_qa(
            os.path.join(data_dir, "b.json"),
            [
                {"question": "Selam", "answer": "Dünya", "source_file": "b.json"},
                {"question": "Bugün hava nasıl?", "answer": "Güzel", "source_file": "b.json"},
            ],
        )

        # BPE dosyaları temp içinde
        vocab_path = os.path.join(tmp, "data", "vocab", "vocab.json")
        merges_path = os.path.join(tmp, "data", "merges", "merges.txt")

        # MODELE GEREKEN PARAMLAR (küçük ve bölünebilir değerler)
        embed_dim = 64
        seq_proj_dim = 64
        num_heads = 4     # 64 % 4 == 0
        dropout = 0.1

        cfg = {
            # Cihaz sade olsun
            "device": "cpu",

            # ÖNEMLİ: Hem data_dir hem data_directory ver → TokenizerCore hangi ismi bekliyorsa bulsun
            "data_dir": data_dir,
            "data_directory": data_dir,

            "vocab_path": vocab_path,
            "merges_path": merges_path,

            # BPE'yi her seferinde taze kur
            "bpe_rebuild": True,
            "merge_operations": 60,
            "bpe_min_frequency": 1,
            "bpe_max_iter": 100000,
            "bpe_include_syllables": True,
            "bpe_include_sep": True,
            "bpe_include_whole_words": True,

            # Eğitim tokenizasyonu
            "train_include_whole_words": True,
            "train_include_syllables": True,
            "train_include_sep": True,

            # Eğitim ayarları
            "batch_size": 4,
            "max_seq_length": 64,
            "learning_rate": 1e-4,
            "use_tensorboard": False,
            "torch_compile": False,  # testte derleme yok

            # Model yapısı (CevahirNeuralNetwork için zorunlu)
            "embed_dim": embed_dim,
            "seq_proj_dim": seq_proj_dim,
            "num_heads": num_heads,
            "dropout": dropout,

            # (opsiyonel) guard’lar; projeye göre yok sayılabilir
            "strict_range_check": True,
        }

        service = TrainingService(cfg)
        yield tmp, data_dir, service

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================
# Test 1: Başlatma
# ================================
def test_training_service_initialization(tmp_env_and_service):
    _, _, service = tmp_env_and_service
    assert service.model_manager.model is not None, "Model init edilmemiş."
    assert hasattr(service.tokenizer_core, "data_loader"), "TokenizerCore.data_loader yok."


# ================================
# Test 2: prepare_data + aralık + çıkış-boyutu
# ================================
def test_training_service_prepare_data_and_ranges(tmp_env_and_service):
    _, _, service = tmp_env_and_service

    train_loader, val_loader, seq_len = service._prepare_data()
    assert hasattr(train_loader, "__iter__")
    assert hasattr(val_loader, "__iter__")
    assert isinstance(seq_len, int)

    vocab_size = _vocab_size(service)
    assert vocab_size > 0

    # Model çıkış son-boyutu vocab ile eşit olmalı
    out_dim = _output_last_dim_from_predict(service, seq_len=8)
    assert out_dim == vocab_size, f"Model out_dim ({out_dim}) != vocab_size ({vocab_size})"

    # İlk batch'te max id kontrolü
    t_max = _first_batch_max(train_loader)
    v_max = _first_batch_max(val_loader)
    assert t_max < vocab_size, f"Train OOB: max_id={t_max} >= vocab={vocab_size}"
    assert v_max < vocab_size, f"Val OOB: max_id={v_max} >= vocab={vocab_size}"

    # Batch yapısı
    for batch in train_loader:
        assert isinstance(batch, tuple) and len(batch) == 2
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], torch.Tensor)
        break


# ================================
# Test 3: Kısa eğitim turu
# ================================
def test_training_service_train_process(tmp_env_and_service):
    _, _, service = tmp_env_and_service
    service.train()
    assert service.training_manager is not None


# ================================
# Test 4: Tahmin
# ================================
def test_training_service_prediction(tmp_env_and_service):
    _, _, service = tmp_env_and_service
    vocab_size = _vocab_size(service)
    seq_len = 64
    dummy_input = torch.randint(0, max(vocab_size, 2), (1, seq_len), dtype=torch.long)
    output = service.predict(dummy_input)
    assert output is not None


# ================================
# Test 5: Vocab-ID hizalaması (tüm batch'ler)
# ================================
def test_vocab_id_alignment(tmp_env_and_service):
    _, _, service = tmp_env_and_service
    train_loader, val_loader, _ = service._prepare_data()

    vocab_size = _vocab_size(service)
    assert vocab_size > 0

    for loader, name in [(train_loader, "train"), (val_loader, "val")]:
        for i, (inp, tgt) in enumerate(loader):
            inp_max = int(inp.max().cpu())
            tgt_max = int(tgt.max().cpu())
            if inp_max >= vocab_size or tgt_max >= vocab_size:
                raise AssertionError(
                    f"[{name}][batch {i}] Out of bounds: "
                    f"max_input_id={inp_max}, max_target_id={tgt_max}, vocab_size={vocab_size}"
                )


# ================================
# Test 6: (Opsiyonel) Negatif OOB senaryosu
# ================================
def test_oob_negative_scenario(tmp_env_and_service):
    """
    Kötü niyetli bir dosya bırakıldığında iki olası davranışı test eder:
    - Guard varsa: _prepare_data() AssertionError ile düşebilir → PASS
    - Guard yoksa: dosya atlanır, veri hazırlanır ve OOB oluşmaz → PASS
    """
    import numpy as np
    import os

    tmp_root, data_dir, _ = tmp_env_and_service

    # Kasıtlı 'kötü' dosya (LOADER_REGISTRY .npy desteklemediği için atlanacak)
    bad = np.full((8,), 9999, dtype=np.int64)
    np.save(os.path.join(data_dir, "leak.npy"), bad)

    vocab_path = os.path.join(tmp_root, "data", "vocab", "vocab.json")
    merges_path = os.path.join(tmp_root, "data", "merges", "merges.txt")

    cfg = {
        "device": "cpu",
        "data_dir": data_dir,
        "data_directory": data_dir,
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "bpe_rebuild": True,
        "merge_operations": 40,

        # Model paramları (küçük ve bölünebilir)
        "embed_dim": 64,
        "seq_proj_dim": 64,
        "num_heads": 4,
        "dropout": 0.1,

        # ÖNEMLİ: model ctor 'learning_rate' istiyor
        "learning_rate": 1e-4,

        # Tokenizasyon / eğitim ayarları
        "train_include_whole_words": True,
        "train_include_syllables": True,
        "train_include_sep": True,
        "batch_size": 4,
        "max_seq_length": 64,
        "use_tensorboard": False,
        "torch_compile": False,

        # Guard varsa proje içinden yakalanabilir
        "strict_range_check": True,
    }

    svc = TrainingService(cfg)

    # 1) Eğer guard tetikleniyorsa _prepare_data() AssertionError atabilir → PASS
    try:
        train_loader, val_loader, _ = svc._prepare_data()
    except AssertionError:
        # Negatif senaryoda beklenen koruma davranışı (varsa)
        return

    # 2) Aksi halde .npy zaten atlanmıştır; OOB olmamalı
    vocab_size = len(svc.tokenizer_core.get_vocab())
    assert vocab_size > 0
    for loader in (train_loader, val_loader):
        for i, (inp, tgt) in enumerate(loader):
            assert int(inp.max().cpu()) < vocab_size, f"OOB in train/val batch {i} (input)"
            assert int(tgt.max().cpu()) < vocab_size, f"OOB in train/val batch {i} (target)"