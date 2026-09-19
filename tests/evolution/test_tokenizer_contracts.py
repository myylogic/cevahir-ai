"""Real shipped vocabulary regressions; no tokenizer training or asset writes."""
import hashlib
import json
from pathlib import Path

import pytest
from tokenizer_management.bpe.bpe_manager import BPEManager

ROOT = Path(__file__).resolve().parents[2]


def manager(**overrides):
    return BPEManager(vocab_file=str(ROOT / "data/vocab_lib/vocab.json"),
                      merges_file=str(ROOT / "data/merges_lib/merges.txt"), use_gpu=False,
                      config={"read_only": True, **overrides})


def test_shipped_token_ids_unchanged_for_all_baseline_probes():
    bpe = manager()
    baseline = json.loads((ROOT / "benchmarks/results/tokenizer_baseline.json").read_text(encoding="utf-8"))
    for language in baseline["languages"].values():
        for example in language["examples"]:
            assert bpe.encode(example["text"], mode="inference", add_special_tokens=False)[1] == example["ids"]
    for name, relative in (("vocab", "data/vocab_lib/vocab.json"), ("merges", "data/merges_lib/merges.txt")):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == baseline["artifacts"]["sha256"][name]


def test_configuration_variants_are_isolated_and_reach_components():
    first = manager(cleanup_punctuation_spaces=True)
    second = manager(cleanup_punctuation_spaces=False)
    assert first is not second
    assert manager(cleanup_punctuation_spaces=True) is first
    for component in (first.encoder, first.decoder, first.trainer):
        assert component.config["cleanup_punctuation_spaces"] is True
    _, ids = first.encode("Merhaba dünya!", mode="inference", add_special_tokens=False)
    assert first.decode(ids) == "Merhaba dünya!"
    assert second.decode(ids) == "Merhaba dünya !"


def test_tokenizer_core_passes_nested_configuration_to_existing_manager():
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    core = TokenizerCore({
        "vocab_path": str(ROOT / "data/vocab_lib/vocab.json"),
        "merges_path": str(ROOT / "data/merges_lib/merges.txt"),
        "read_only": True, "use_gpu": False,
        "bpe_config": {"cleanup_punctuation_spaces": False},
    })
    assert core.tokenizer.decoder.config["cleanup_punctuation_spaces"] is False


@pytest.mark.parametrize("character", list("İıŞşĞğÜüÖöÇçIi"))
def test_turkish_unicode_case_roundtrip(character):
    bpe = manager()
    _, ids = bpe.encode(character, mode="inference", add_special_tokens=False)
    assert bpe.decode(ids) == character


def test_missing_readonly_assets_fail_without_creating_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPEManager(vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt"), config={"read_only": True})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("text", ["🧬", "漢字", "Merhaba 🧬 dünya", "\u0378"])
def test_strict_text_loss_policy_rejects_dropped_characters(text):
    from tokenizer_management.bpe.bpe_manager import TokenizerTextLossError
    with pytest.raises(TokenizerTextLossError) as error:
        manager(text_loss_policy="error").encode(text, add_special_tokens=False)
    assert error.value.dropped_codepoints


def test_loss_warning_exposes_codepoints_without_echoing_private_text(caplog):
    bpe = manager(text_loss_policy="warn")
    bpe.encode("private-secret 🧬", add_special_tokens=False)
    assert "U+1F9EC" in caplog.text
    assert "private-secret" not in caplog.text


def test_explicit_special_token_choice_overrides_configuration():
    bpe = manager(add_special_tokens=True)
    tokens, _ = bpe.encode("Merhaba", add_special_tokens=False)
    assert "<BOS>" not in tokens
    assert bpe.encode("Merhaba")[0][0] == "<BOS>"
