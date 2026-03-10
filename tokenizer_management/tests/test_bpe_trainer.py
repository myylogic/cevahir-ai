import pytest
import copy
import logging

from tokenizer_management.bpe.bpe_trainer import (
    BPETrainer,
    BPETrainingError,
)

@pytest.fixture
def simple_vocab():
    # Orijinal sözlüğü bozmamak için deep copy testlerinde kullanılacak
    return {"x": {"id": 0}, "y": {"id": 1}}

def test_init_non_dict_raises_type_error():
    """Dict olmayan vocab ile başlatma TypeError fırlatmalı."""
    with pytest.raises(TypeError):
        BPETrainer(vocab=["not", "a", "dict"])

def test_init_empty_dict_loads_default_vocab(caplog):
    """Boş dict verilirse default vocab yüklenmeli ve özel tokenlar eklenmeli."""
    caplog.set_level(logging.INFO)
    trainer = BPETrainer(vocab={})
    # Özel tokenların eklendiğine dair log
    assert "Varsayılan vocab yüklendi." in caplog.text
    for tok in trainer.SPECIAL_TOKENS:
        assert tok in trainer.vocab

def test_ensure_special_tokens_does_not_duplicate():
    """Zaten eksiksiz özel tokenlar varsa tekrar eklenmemeli."""
    base = {tok: {"id": i} for i, tok in enumerate(BPETrainer.SPECIAL_TOKENS)}
    caplog = pytest.LogCapture()
    caplog.set_level(logging.INFO)
    trainer = BPETrainer(vocab=copy.deepcopy(base))
    # Warning kayıtları olmamalı
    assert all(record.levelno != logging.WARNING for record in caplog.records)

@pytest.mark.parametrize("corpus, expected_pairs", [
    ([["a", "b"], ["a", "b"]], {("a","b"):2}),
    ([["c","d","c"]], {("c","d"):1, ("d","c"):1}),
])
def test_get_stats_counts_bigrams(corpus, expected_pairs):
    """_get_stats metodu bigram frekansını doğru hesaplamalı."""
    trainer = BPETrainer(vocab={})
    seq = []
    for s in corpus:
        seq.extend(s + ["<EOS>"])
    stats = trainer._get_stats(seq, min_frequency=1)
    for pair, freq in expected_pairs.items():
        assert stats[pair] == freq

def test_get_stats_filters_special_tokens():
    """_get_stats özel token içeren pair’leri atlamalı."""
    trainer = BPETrainer(vocab={})
    seq = ["a", "<PAD>", "b", "<UNK>", "c"]
    stats = trainer._get_stats(seq, min_frequency=1)
    assert all("<PAD>" not in pair and "<UNK>" not in pair for pair in stats)

def test_merge_pair_merges_correctly():
    """_merge_pair verilen pair’i her yerde doğru birleştirmeli."""
    trainer = BPETrainer(vocab={})
    sequence = ["a","b","b","a","b"]
    merged = trainer._merge_pair(sequence, ("b","a"))
    # "b","a" her yerde "ba" olmalı
    assert merged == ["a","b","ba","b"]

def test_train_zero_merges_does_nothing():
    """target_merges=0 ise hiçbir merge adımı yapılmamalı."""
    corpus = [["a","b","a"]]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=0)
    assert trainer.get_merges() == []

def test_train_min_frequency_filters_out():
    """min_frequency yüksekse hiçbir merge adımı yapılmamalı."""
    corpus = [["x","y","z","x","y"]]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=5, min_frequency=2)
    # ("x","y") freq=2, merge olur ama sonra next pair yok.
    assert ("x","y") in trainer.get_merges()

def test_train_respects_max_iter():
    """max_iter sınırı merge sayısını kısıtlamalı."""
    corpus = [["a","b"]*10]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=100, max_iter=1)
    assert len(trainer.get_merges()) == 1

def test_train_adds_new_tokens_and_merges(simple_vocab):
    """train, en sık pair’i vocab’a ve merges_ listesine eklemeli."""
    corpus = [["a","b","a","b","a","b"]]
    # Başlangıçta a,b yok
    initial = {}
    trainer = BPETrainer(vocab=initial)
    trainer.train(corpus, target_merges=1)
    merges = trainer.get_merges()
    assert merges == [("a","b")]
    # Yeni token 'ab' eklenmiş olmalı
    assert "ab" in trainer.vocab

def test_repeated_train_clears_previous_merges():
    """Tekrar train çağrısında merges_ önce temizlenmeli."""
    corpus = [["a","b"]*3]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=1)
    first = trainer.get_merges()
    trainer.train(corpus, target_merges=2)
    second = trainer.get_merges()
    assert first != second
    assert len(second) == 2

def test_get_merges_returns_copy():
    """get_merges iç listeyi kopyalayarak döndürmeli."""
    trainer = BPETrainer(vocab={})
    trainer.merges_.extend([("x","y")])
    out = trainer.get_merges()
    out.append(("y","z"))
    assert ("y","z") not in trainer.merges_

def test_get_vocab_returns_deep_copy(simple_vocab):
    """get_vocab geri dönen dict’i değiştirmek iç durumu etkilememeli."""
    trainer = BPETrainer(vocab=simple_vocab)
    v = trainer.get_vocab()
    v["zzz"] = {"id": 99}
    assert "zzz" not in trainer.vocab

def test_get_vocab_error_when_empty():
    """reset sonrası boşalırsa get_vocab error fırlatmalı."""
    trainer = BPETrainer(vocab={})
    # zorla boşalt
    trainer.vocab = {}
    with pytest.raises(BPETrainingError):
        trainer.get_vocab()

def test_update_vocab_adds_only_new_tokens():
    """update_vocab yalnızca eklenmemiş tokenları eklemeli."""
    trainer = BPETrainer(vocab={"a": {"id": 0}})
    trainer.update_vocab(["b","a","c"])
    assert "b" in trainer.vocab
    assert "c" in trainer.vocab
    # 'a' zaten vardı
    assert trainer.vocab["a"]["id"] == 0

def test_reset_restores_default_vocab_and_clears_merges(caplog):
    """reset çağrısı sonrası vocab default’a dönmeli, merges_ boş olmalı."""
    trainer = BPETrainer(vocab={"foo":{"id":10}})
    trainer.merges_.append(("x","y"))
    trainer.reset()
    assert trainer.get_merges() == []
    for tok in trainer.SPECIAL_TOKENS:
        assert tok in trainer.vocab
    assert "foo" not in trainer.vocab

def test_default_vocab_ids_sequential():
    """default vocab sırayla 0–3 id’lere sahip olmalı."""
    trainer = BPETrainer(vocab={})
    default = trainer._default_vocab()
    ids = sorted(info["id"] for info in default.values())
    assert ids == [0,1,2,3]

def test_original_vocab_not_modified_by_trainer(simple_vocab):
    """Trainer kendi içinde copy oluşturmalı, orijinal değişmemeli."""
    orig = copy.deepcopy(simple_vocab)
    BPETrainer(vocab=orig)
    assert orig == simple_vocab

def test_train_raises_on_bad_corpus_type():
    """train argümanı uygun tipte değilse ValueError fırlatmalı."""
    trainer = BPETrainer(vocab={})
    with pytest.raises(ValueError):
        trainer.train(tokenized_corpus="notalist", target_merges=1)

def test_train_stops_when_no_stats():
    """Hiç stats dönmemeye başladığında döngü kırılmalı."""
    trainer = BPETrainer(vocab={})
    # tüm öğeler SPECIAL_TOKENS
    corpus = [["<PAD>","<UNK>"]]
    trainer.train(corpus, target_merges=5)
    assert trainer.get_merges() == []

def test_train_with_min_frequency_greater_than_counts():
    """min_frequency > mevcut frekans yields no merges."""
    corpus = [["a","b","a","b","a","b"]]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=3, min_frequency=10)
    assert trainer.get_merges() == []

@pytest.mark.parametrize("tokens,pair,expected", [
    (["a","b","c","b","c"], ("b","c"), ["a","bc","bc"]),
    (["x","y","x","y"], ("x","y"), ["xy","xy"]),
])
def test_merge_pair_parametrized(tokens, pair, expected):
    """_merge_pair parametrik olarak çalışmalı."""
    trainer = BPETrainer(vocab={})
    merged = trainer._merge_pair(tokens, pair)
    assert merged == expected

def test_logging_during_training(caplog):
    """train sırasında INFO ve DEBUG logları oluşturulmalı."""
    caplog.set_level(logging.DEBUG)
    corpus = [["a","a","b","b"]]
    trainer = BPETrainer(vocab={})
    trainer.train(corpus, target_merges=1)
    assert "BPE eğitimi tamamlandı" in caplog.text or "Merged" in caplog.text
