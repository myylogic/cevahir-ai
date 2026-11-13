"""
FAZE 5.1: Generation Süreci Kontrolü Testleri

Bu test modülü, generation/inference sürecinin doğru çalışıp çalışmadığını kontrol eder.

Test Edilecekler:
1. Prompt encoding doğru mu?
2. Model forward pass generation sırasında doğru mu?
3. Sampling parametreleri doğru mu? (temperature, top-k, top-p)
4. EOS token doğru handle ediliyor mu?
5. Generation loop doğru mu? (autoregressive generation)
6. Training_service.py'deki basit generation doğru mu?

Kod Konumu:
- training_system/v2/core/training_service.py → _test_model_after_epoch() (basit generation)
- model/cevahir.py → CevahirModelAPI.generate() (tam generation API)

Not: Testler generation mekaniğini küçük, rastgele ağırlıklı modellerle ve repo'daki
BPE dosyalarıyla doğrular; eğitilmiş bir checkpoint gerektirmez. Çıktı kalitesi kapsam dışıdır.
"""

import logging
import os
import sys

import pytest
import torch

# Root dizini path'e ekle
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from tokenizer_management.core.tokenizer_core import TokenizerCore
from src.neural_network import CevahirNeuralNetwork

logger = logging.getLogger(__name__)

# Repo'daki eğitilmiş BPE dosyaları. Mutlak yol kullanılır: TokenizerCore göreli yollarda
# CWD altında dizin oluşturur, testler ise repo ağacına yazmamalı.
VOCAB_PATH = os.path.join(REPO_ROOT, "data", "vocab_lib", "vocab.json")
MERGES_PATH = os.path.join(REPO_ROOT, "data", "merges_lib", "merges.txt")

TEST_PROMPTS = [
    "selam",
    "merhaba nasılsın",
    "ben bir yapay zeka modeliyim",
]


@pytest.fixture(scope="module")
def tokenizer_core():
    """Repo'daki vocab/merges ile TokenizerCore (modül başına bir kez yüklenir).

    Eski sürüm tokenizer config'ini kaldırılmış config_management.ConfigLoader ile
    yüklüyordu; artık CevahirConfig'in varsayılan tokenizer ayarları kullanılıyor.
    """
    return TokenizerCore({
        "vocab_path": VOCAB_PATH,
        "merges_path": MERGES_PATH,
        "use_gpu": False,
        "batch_size": 32,
        "max_unk_ratio": 0.01,
        # Salt okunur: vocab/merges eksik olsa bile yeni dosya oluşturulmaz
        "read_only": True,
    })


def _build_minimal_model(vocab_size: int, embed_dim: int = 128) -> CevahirNeuralNetwork:
    """Generation testleri için minimal, rastgele ağırlıklı model (eval mode)."""
    # learning_rate ve dropout artık zorunlu constructor argümanları (API değişikliği);
    # model yalnızca eval/no_grad altında kullanıldığı için dropout=0.0.
    model = CevahirNeuralNetwork(
        learning_rate=1e-3,
        dropout=0.0,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=embed_dim,
        num_heads=8,
        num_layers=2,
        ffn_dim=512,
    )
    model.eval()  # Generation için eval mode
    return model


def _eos_token_id(tokenizer_core) -> int:
    """Gerçek vocab'deki <EOS> token ID'si (vocab değerleri dict veya int olabilir)."""
    eos = tokenizer_core.get_vocab().get("<EOS>")
    eos_id = eos.get("id") if isinstance(eos, dict) else eos
    assert isinstance(eos_id, int), f"Vocab'de <EOS> token ID'si bulunamadı: {eos!r}"
    return eos_id


def test_prompt_encoding(tokenizer_core):
    """Test 1: Prompt encoding kontrolü"""
    for prompt in TEST_PROMPTS:
        tokens, token_ids = tokenizer_core.encode(prompt, mode="inference")

        # Encoding başarılı mı?
        assert len(token_ids) > 0, f"Prompt '{prompt}' encode edilemedi!"
        assert len(tokens) > 0, f"Prompt '{prompt}' tokenize edilemedi!"

        # Decode kontrolü (roundtrip test)
        decoded = tokenizer_core.decode(token_ids, method="bpe", remove_specials=True)
        assert decoded == prompt, (
            f"Roundtrip başarısız: '{prompt}' → tokens={tokens}, ids={token_ids} → '{decoded}'"
        )


def test_model_forward_pass_generation():
    """Test 2: Model forward pass generation sırasında kontrolü"""
    vocab_size = 1000
    seq_len = 10
    model = _build_minimal_model(vocab_size)

    # Test input (basit token IDs)
    test_input = torch.randint(0, vocab_size, (1, seq_len))

    # Forward pass (generation mode - inference=True equivalent)
    with torch.no_grad():
        output = model(test_input)
    logits = output[0] if isinstance(output, tuple) else output

    # Shape kontrolü
    assert logits.shape == (1, seq_len, vocab_size), \
        f"Output shape yanlış! Beklenen: (1, {seq_len}, {vocab_size}), Alınan: {tuple(logits.shape)}"

    # NaN/Inf kontrolü
    assert not torch.isnan(logits).any(), "Output NaN içeriyor!"
    assert not torch.isinf(logits).any(), "Output Inf içeriyor!"

    # Son token'ın logits'i (generation için önemli) ve softmax kontrolü
    last_logits = logits[0, -1, :]
    assert last_logits.shape == (vocab_size,), f"Last token logits shape yanlış: {tuple(last_logits.shape)}"
    probs = torch.softmax(last_logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 0.01, f"Probabilities sum != 1.0! ({probs.sum().item():.6f})"


def test_sampling_process():
    """Test 3: Sampling parametreleri kontrolü (temperature, top-k, top-p)"""
    vocab_size = 1000

    # Test logits (rastgele, tekrarlanabilir)
    generator = torch.Generator().manual_seed(0)
    test_logits = torch.randn(vocab_size, generator=generator)
    # Top-k / top-p sıcaklıksız (T=1.0) dağılım üzerinde yapılır. Eski sürüm burada
    # sıcaklık döngüsünden kalan T=2.0 dağılımını yanlışlıkla kullanıyordu.
    base_probs = torch.softmax(test_logits, dim=-1)

    # Temperature kontrolü: temperature düşükse, daha "sharp" distribution olmalı
    temperatures = [0.5, 1.0, 1.5, 2.0]
    entropies = []
    max_probs = []
    for temp in temperatures:
        probs = torch.softmax(test_logits / temp, dim=-1)
        entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        max_probs.append(probs.max().item())
    assert all(a < b for a, b in zip(entropies, entropies[1:])), \
        f"Entropy temperature ile artmalı: {dict(zip(temperatures, entropies))}"
    assert all(a > b for a, b in zip(max_probs, max_probs[1:])), \
        f"Max probability temperature ile azalmalı: {dict(zip(temperatures, max_probs))}"

    # Top-k kontrolü
    for k in [1, 5, 10, 50, 100]:
        _, top_k_indices = torch.topk(base_probs, k=k)
        # Top-k dışındaki logits'leri -inf yap (mask)
        top_k_logits = test_logits.clone()
        mask = torch.ones_like(test_logits, dtype=torch.bool)
        mask[top_k_indices] = False
        top_k_logits[mask] = float('-inf')

        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        assert abs(top_k_probs.sum().item() - 1.0) < 0.01, f"Top-k={k} probabilities sum != 1.0!"
        non_zero_count = int((top_k_probs > 0).sum().item())
        assert non_zero_count == k, f"Top-k={k}: {non_zero_count} token olasılık taşıyor"

    # Top-p (nucleus sampling) kontrolü
    probs_sorted, indices_sorted = torch.sort(base_probs, descending=True)
    cumsum = torch.cumsum(probs_sorted, dim=0)
    for p in [0.1, 0.5, 0.9, 0.95, 1.0]:
        # p threshold'undan küçük olan ilk n token'ı seç (en az 1 token)
        keep_count = max(int((cumsum <= p).sum().item()), 1)

        top_p_logits = test_logits.clone()
        top_p_mask = torch.ones_like(test_logits, dtype=torch.bool)
        top_p_mask[indices_sorted[:keep_count]] = False
        top_p_logits[top_p_mask] = float('-inf')

        top_p_probs = torch.softmax(top_p_logits, dim=-1)
        assert abs(top_p_probs.sum().item() - 1.0) < 0.01, f"Top-p={p} probabilities sum != 1.0!"
        non_zero_count = int((top_p_probs > 0).sum().item())
        assert non_zero_count == keep_count, \
            f"Top-p={p}: {non_zero_count} token olasılık taşıyor, beklenen {keep_count}"


def test_eos_token_handling(tokenizer_core):
    """Test 4: EOS token handling kontrolü"""
    # EOS token ID gerçek vocab'den alınır
    eos_id = _eos_token_id(tokenizer_core)

    vocab_size = 1000
    assert eos_id < vocab_size
    max_new_tokens = 10
    prompt_ids = [1, 2]  # Başlangıç token'ları (BOS, ilk token)

    def simulate_generation(eos_step):
        """Generation loop simülasyonu; EOS yalnızca eos_step adımında üretilir.

        Rastgele örneklemede EOS nadiren çıktığı için durma davranışı deterministik
        olarak sınanır (eos_step=None → EOS hiç üretilmez).
        """
        generator = torch.Generator().manual_seed(0)
        generated_ids = list(prompt_ids)
        eos_found = False
        for step in range(max_new_tokens):
            # Simüle edilmiş logits (rastgele)
            logits = torch.randn(vocab_size, generator=generator)
            logits[eos_id] = 1e4 if step == eos_step else float('-inf')
            probs = torch.softmax(logits, dim=-1)

            # Sample
            next_token = torch.multinomial(probs, 1, generator=generator).item()
            generated_ids.append(next_token)

            # EOS kontrolü: EOS bulunursa generation durmalı
            if next_token == eos_id:
                eos_found = True
                break
        return generated_ids[len(prompt_ids):], eos_found

    # EOS üretilirse generation durmalı
    eos_step = 4
    new_tokens, eos_found = simulate_generation(eos_step)
    assert eos_found, "Üretilen EOS token'ı algılanmadı"
    assert len(new_tokens) == eos_step + 1, f"EOS sonrası generation durmadı: {new_tokens}"
    assert new_tokens[-1] == eos_id and eos_id not in new_tokens[:-1], new_tokens

    # EOS üretilmezse generation max_new_tokens'a kadar sürmeli
    new_tokens, eos_found = simulate_generation(None)
    assert not eos_found
    assert len(new_tokens) == max_new_tokens, f"Beklenen {max_new_tokens} token, alınan {len(new_tokens)}"
    assert eos_id not in new_tokens


def test_autoregressive_generation_loop():
    """Test 5: Autoregressive generation loop kontrolü"""
    vocab_size = 1000
    prompt_length = 5
    max_new_tokens = 10
    model = _build_minimal_model(vocab_size)

    # Prompt (başlangıç token'ları)
    prompt = torch.randint(0, vocab_size, (1, prompt_length))
    generated_ids = prompt[0].tolist()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Current input (tüm generated sequence)
            current_input = torch.tensor([generated_ids], dtype=torch.long)

            # Forward pass
            output = model(current_input)
            logits = output[0] if isinstance(output, tuple) else output
            assert logits.shape == (1, len(generated_ids), vocab_size), \
                f"Step {step + 1}: logits shape yanlış: {tuple(logits.shape)}"

            # Son token'ın logits'i → sampling (greedy - argmax)
            next_logits = logits[0, -1, :]
            next_token = torch.argmax(next_logits).item()
            assert 0 <= next_token < vocab_size, f"Step {step + 1}: geçersiz token ID {next_token}"
            generated_ids.append(next_token)

    # Kontroller
    assert len(generated_ids) == prompt_length + max_new_tokens, \
        f"Generated sequence length yanlış! Beklenen: {prompt_length + max_new_tokens}, Alınan: {len(generated_ids)}"
    assert generated_ids[:prompt_length] == prompt[0].tolist(), "Prompt token'ları değişmemeli"


def test_training_service_generation(tokenizer_core, caplog):
    """Test 6: training_service.py'deki basit generation kontrolü

    Eski sürüm "gerçek model ve tokenizer gerekli" diyerek True döndürüp atlıyordu.
    Tokenizer repo'da mevcut ve generation mekaniği eğitilmiş ağırlık gerektirmez, bu
    yüzden Test 2/5 gibi minimal rastgele model kullanılır.
    _test_model_after_epoch() prompt başına hataları yakalayıp yalnızca logladığından
    sonuç log kayıtlarından doğrulanır.
    """
    from model_management.model_manager import ModelManager
    from training_system.v2.core import training_service as training_service_module

    vocab_size = tokenizer_core.get_vocab_size()
    model_manager = ModelManager({"device": "cpu", "vocab_size": vocab_size}, tokenizer=tokenizer_core)

    # Tam TrainingService kurulumu data_dir, veri cache'i (CWD altına yazar) ve optimizer
    # ister; test edilen metot yalnızca aşağıdaki alanları kullanır.
    service = training_service_module.TrainingService.__new__(training_service_module.TrainingService)
    service.config = {}
    service.logger = training_service_module.service_logger
    service.device = model_manager.device
    service.model_manager = model_manager
    service.tokenizer_core = tokenizer_core

    # Ağırlıklar ve örnekleme tekrarlanabilir olsun; global RNG durumu test sonrası geri yüklenir
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model_manager.model = _build_minimal_model(vocab_size, embed_dim=32)
        model_manager.model.train()  # Eğitim sırasında çağrılır; mod geri alınmalı
        with caplog.at_level(logging.INFO, logger=service.logger.name):
            max_eos_prob = service._test_model_after_epoch(
                epoch=1, train_loss=0.0, val_loss=0.0, test_prompts=TEST_PROMPTS
            )

    messages = [record.getMessage() for record in caplog.records if record.name == service.logger.name]
    errors = [m for m in messages if "için hata" in m or "tokenize edilemedi" in m or "initialize edilmemiş" in m]
    assert not errors, f"Generation hataları: {errors}"

    responses = [m.strip() for m in messages if m.strip().startswith("Response:")]
    assert len(responses) == len(TEST_PROMPTS), f"Her prompt için bir cevap bekleniyordu: {messages}"
    token_counts = [int(m.split(":", 1)[1]) for m in messages if m.strip().startswith("Tokens:")]
    assert len(token_counts) == len(TEST_PROMPTS) and all(0 < n <= 30 for n in token_counts), \
        f"Üretilen token sayıları beklenmedik: {token_counts}"

    assert 0.0 <= max_eos_prob <= 1.0, f"EOS olasılığı [0, 1] aralığında olmalı: {max_eos_prob}"
    assert model_manager.model.training, "Model generation sonrası eğitim moduna geri alınmadı"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
