"""
TrainingService Data Processing Analizi Testi
==============================================
TrainingService'in cache'den gelen verileri nasıl işlediğini test eder.

Test Amacı:
- Cache'den yüklenen örneklerin doğru işlenip işlenmediği
- EOS ekleme mekanizması (EOS yalnız belgenin son kaydının hedef sonunda mı?)
- BOS ekleme mekanizması (BOS yalnız belgenin ilk kaydının başında mı?)
- Alignment kontrolü (autoregressive format doğru mu?)
- max_seq_length'e sığmayan kayıt kesilip sahte EOS ile kapatılmak yerine reddediliyor mu?
- Loss function'da EOS weight uygulanıyor mu?

GÜNCEL AKIŞ:
Eski test, cache'deki ham (inp, inp) çiftlerine BOS/EOS ekleyen eski
TrainingService._prepare_data mantığını test içinde yeniden yazıp (simüle edip) uyguluyordu.
Bu formatlama artık cache'e yazılmadan ÖNCE yapılıyor (TrainingService.prepare_from_cache →
format_data_for_autoregressive; training_system/prepare_cache.py da aynı biçimi üretir).
Cache'deki kayıtlar bu yüzden zaten formatlıdır (training_system/record_format.py):
    input  = [BOS, t1, ..., tN, PAD, ...]     (BOS yalnız belgenin ilk kaydında)
    target = [t1, ..., tN, EOS, PAD, ...]     (EOS yalnız belgenin son kaydında; target[t] == input[t+1])
Uzun belgenin ara parçaları BOS da EOS da almaz. max_seq_length'e sığmayan kayıt kesilip EOS ile
kapatılmaz (bu belgenin sonunu sahte yere taşırdı); ValueError verilir. Testler bu biçimi
doğrudan doğrular:
- Ürünün kendi formatlama/önbellek yolunu küçük bir geçici korpusla çalıştırarak (her ortamda),
- Varsa yerel eğitim cache'indeki kayıtlar üzerinde (.cache/preprocessed_data, repo'da yok;
  yoksa test atlanır).
Hiçbir test repo ağacına yazmaz; cache'ler pytest'in geçici dizinlerindedir.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest
import torch
import torch.nn.functional as F

# Proje kök dizinini sys.path'e ekle (dosya doğrudan çalıştırıldığında da import'lar çalışsın)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG
from training_system.data_cache import DataCache
# Eski yol training_system/training_service.py artık yok; V2 TrainingService
from training_system.v2.core.training_service import TrainingService

# Modül seviyesinde logging.basicConfig yok; standart logger yeterli (pytest yakalar).
logger = logging.getLogger("TrainingServiceDataProcessing")

# Yerel eğitim cache'i (training_system/prepare_cache.py üretir; .gitignore'da, repo'da yok)
LOCAL_CACHE_DIR = ROOT / ".cache" / "preprocessed_data"
MAX_ANALYZED_EXAMPLES = 1000  # İlk 1000 örneği işle (hızlı test için)

# Tokenizer örnekleri en fazla SERVICE_MAX_SEQ_LEN - 2 içerik token'ına böler (BOS/EOS payı).
SERVICE_MAX_SEQ_LEN = 64
# Formatlayıcıya daha kısa bir uzunluk verilince kayıtlar sığmaz; kesilmez, reddedilir.
TRUNCATED_MAX_SEQ_LEN = 16

# Ürün varsayılanı artık eos_token_weight=1.0 (CriterionManager: 0.1 EOS öğrenimini bozuyordu).
# 1.0 ağırlık etkisizdir; mekanizmanın gözlenebilmesi için eski testteki 0.1 kullanılır.
EOS_WEIGHT = 0.1
LABEL_SMOOTHING = 0.1
PAD_TOKEN_ID = 0

QA_PAIRS = [
    {"question": "Merhaba, nasılsın?",
     "answer": "İyiyim, teşekkür ederim. Sen nasılsın, bugün neler yaptın?"},
    {"question": "Yapay zeka nedir?",
     "answer": "Yapay zeka, insan zekasını taklit eden ve verilerden öğrenebilen bilgisayar "
               "sistemlerinin genel adıdır. Dil modelleri metindeki bir sonraki kelimeyi tahmin "
               "ederek öğrenir ve bu tahminleri art arda yaparak cümle kurar."},
    {"question": "Ankara hangi bölgededir?",
     "answer": "Ankara, İç Anadolu Bölgesinde yer alır ve Cumhuriyetin ilanından önce başkent "
               "olarak seçilmiştir."},
    {"question": "Bugün hava nasıl?",
     "answer": "Bugün hava güneşli ve ılık, akşama doğru hafif bir rüzgar bekleniyor."},
]
RAW_TEXT = " ".join(
    ["Bilim, evreni anlamak için gözlem ve deney kullanır. Her yeni bulgu eski fikirleri sınar."] * 8
)


def special_token_ids(tokenizer_core) -> Dict[str, int]:
    """Tokenizer vocab'ından PAD/BOS/EOS ID'lerini al."""
    vocab = tokenizer_core.get_vocab()

    def _id_of(token: str) -> int:
        val = vocab.get(token)
        assert val is not None, f"{token} vocab'ta yok"
        if isinstance(val, dict):
            return int(val["id"])
        return int(val)

    return {"pad_id": _id_of("<PAD>"), "bos_id": _id_of("<BOS>"), "eos_id": _id_of("<EOS>")}


def find_format_problems(inp: Sequence[int], tgt: Sequence[int], *, pad_id: int, bos_id: int, eos_id: int,
                         starts_document: bool = True, ends_document: bool = True) -> List[str]:
    """Formatlı bir eğitim örneğindeki sorunları listele (boş liste = doğru format).

    BOS yalnız belgeyi başlatan kaydın, EOS yalnız belgeyi bitiren kaydın içinde bulunur.
    """
    inp = [int(t) for t in inp]
    tgt = [int(t) for t in tgt]
    if len(inp) != len(tgt):
        return [f"Alignment hatası - Input len={len(inp)} != Target len={len(tgt)}"]

    problems = []
    if not inp or (inp[0] == bos_id) != starts_document:
        expected = "başlamalı" if starts_document else "başlamamalı (belge ortası)"
        problems.append(f"Input BOS ile {expected} (ilk token={inp[0] if inp else None})")

    content = [i for i, token in enumerate(tgt) if token != pad_id]
    if not content:
        return problems + ["Target tamamen PAD"]
    last = content[-1]  # son içerik pozisyonu (belge bitiyorsa EOS)
    if (tgt[last] == eos_id) != ends_document:
        expected = "EOS olmalı" if ends_document else "EOS olmamalı (belge sürüyor)"
        problems.append(f"Target'in son PAD-dışı token'ı {expected} (pozisyon {last}, id={tgt[last]})")
    if eos_id in tgt[:last] or bos_id in inp[1:last + 1]:
        problems.append("İçeriğin ortasında BOS/EOS var")
    if len(content) != last + 1:
        problems.append("Target'ta içerik arasında PAD var")
    if inp[1:last + 1] != tgt[:last]:
        problems.append("Autoregressive alignment hatası: target[t] != input[t+1]")
    if any(token != pad_id for token in inp[last + 1:]):
        problems.append("Input, içerikten sonra PAD-dışı token içeriyor")
    return problems


def collect_format_failures(records, ids: Dict[str, int]) -> List[Tuple[int, List[str]]]:
    """Cache sırasındaki (input, target[, source_id]) kayıtlarından hatalı formatlı olanları döndür.

    Belge sınırları kayıt sırasından okunur: aynı source_id'nin ardışık kayıtları bir belgedir.
    """
    from training_system.record_format import record_boundaries
    failures = []
    for idx, (item, (starts, ends)) in enumerate(zip(records, record_boundaries(records))):
        problems = find_format_problems(item[0], item[1], **ids, starts_document=starts, ends_document=ends)
        if problems:
            failures.append((idx, problems))
    return failures


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seeded_rng():
    """Dummy logits/target deterministik olsun; global RNG durumu test sonrası geri yüklenir."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


def _bpe_paths() -> Tuple[str, str]:
    vocab_path, merges_path = BPE_CONFIG["vocab_file"], BPE_CONFIG["merges_file"]
    for path in (vocab_path, merges_path):
        # Depoda izlenen dosyalar: yoksa checkout bozuktur; sessizce atlamak yerine başarısız ol.
        assert Path(path).is_file(), f"BPE dosyası bulunamadı: {path}"
    return vocab_path, merges_path


@pytest.fixture(scope="module")
def tokenizer_core():
    """Repo'daki BPE vocab/merges ile salt-okunur TokenizerCore (özel token ID'leri için)."""
    vocab_path, merges_path = _bpe_paths()
    return TokenizerCore({
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "use_gpu": False,  # CPU için
        "read_only": True,  # repo'daki vocab/merges dosyalarına asla yazma
    })


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    """Küçük eğitim korpusu: farklı uzunlukta QA çiftleri (JSON) + uzun bir düz metin (TXT)."""
    path = tmp_path_factory.mktemp("education")
    (path / "sohbet.json").write_text(json.dumps(QA_PAIRS, ensure_ascii=False), encoding="utf-8")
    (path / "bilim.txt").write_text(RAW_TEXT, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def training_service(corpus_dir, tmp_path_factory):
    """Küçük modelli gerçek TrainingService (V2): tokenizer, DataCache, ModelManager, criterion."""
    from model_management.config_schema import tiny_model_config

    vocab_path, merges_path = _bpe_paths()
    config = tiny_model_config(num_layers=1, max_seq_length=SERVICE_MAX_SEQ_LEN, max_cache_len=SERVICE_MAX_SEQ_LEN)
    config.update(
        data_dir=str(corpus_dir),
        cache_dir=str(tmp_path_factory.mktemp("preprocessed_data")),
        vocab_path=vocab_path,
        merges_path=merges_path,
        read_only=True,  # repo'daki vocab/merges dosyalarına asla yazma
        use_gpu=False,
        device="cpu",
        pad_token_id=PAD_TOKEN_ID,
        eos_token_weight=EOS_WEIGHT,
        label_smoothing=LABEL_SMOOTHING,
        train_include_whole_words=True,
        train_include_syllables=False,
        train_include_sep=False,
    )
    return TrainingService(config)


# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------

def test_prepare_from_cache_formats_cached_examples(training_service, corpus_dir, tmp_path):
    """
    TrainingService.prepare_from_cache: cache'e yazılan ve eğitime verilen örneklerde BOS yalnız belgenin
    ilk kaydının başında, EOS yalnız son kaydının hedef sonunda ve target[t] == input[t+1] mi?
    """
    max_seq_len = SERVICE_MAX_SEQ_LEN
    ids = special_token_ids(training_service.tokenizer_core)
    cache_dir = tmp_path / "preprocessed_data"
    data_cache = DataCache(data_dir=str(corpus_dir), cache_dir=str(cache_dir), cache_enabled=True)
    config = dict(training_service.config, max_seq_length=max_seq_len)

    train_data, val_data, vocab_size = training_service.prepare_from_cache(
        data_cache=data_cache,
        tokenizer_core=training_service.tokenizer_core,
        config=config,
    )
    assert vocab_size == len(training_service.tokenizer_core.get_vocab())

    # Cache'den yüklenen örnekler: eski testteki gibi cache dosyasını doğrudan oku
    cache_files = sorted(cache_dir.glob("cached_data_*.pkl"))
    assert len(cache_files) == 1, f"Tek cache dosyası bekleniyordu: {[p.name for p in cache_files]}"
    with open(cache_files[0], "rb") as f:
        cached = pickle.load(f)
    assert cached, f"Cache boş: {cache_files[0].name}"

    failures = collect_format_failures(cached, ids)
    assert not failures, (
        f"{len(failures)}/{len(cached)} cache örneği hatalı formatlı (special ids={ids}); ilk 5: {failures[:5]}"
    )
    lengths = {len(item[0]) for item in cached}
    assert lengths == {max_seq_len}, f"Örnekler max_seq_length={max_seq_len} uzunluğa PAD'lenmeli: {lengths}"

    # Uzun düz metin birden çok kayda bölünür: belge başına tek BOS (ilk kayıt) ve tek EOS (son kayıt)
    per_source = {}
    for inp, tgt, source_id in cached:
        per_source.setdefault(source_id, []).append(
            (int(inp[0]) == ids["bos_id"], ids["eos_id"] in [int(t) for t in tgt]))
    assert max(len(flags) for flags in per_source.values()) >= 2, "Uzun metin tek kayda sığdı; parça sınırı sınanmadı"
    for source_id, flags in per_source.items():
        assert [bos for bos, _ in flags] == [True] + [False] * (len(flags) - 1), (source_id, flags)
        assert [eos for _, eos in flags] == [False] * (len(flags) - 1) + [True], (source_id, flags)

    # Eğitime verilen (train/val split sonrası) tensörler: cache'deki kayıtların aynısı (yalnız sıra değişir)
    split_examples = sorted((x.tolist(), y.tolist()) for x, y in list(train_data) + list(val_data))
    assert split_examples == sorted(([int(t) for t in item[0]], [int(t) for t in item[1]]) for item in cached), (
        f"Split sonrası örnekler ({len(split_examples)}) cache'deki {len(cached)} kayıtla aynı değil"
    )


def test_prepare_from_cache_rejects_records_longer_than_max_seq_len(training_service, corpus_dir, tmp_path):
    """
    Tokenizer kayıtları SERVICE_MAX_SEQ_LEN'e göre parçalar; daha kısa bir max_seq_length ile biçimlemek
    kayıtları sığdıramaz. Eski kod kesip son hedefi EOS yapıyordu (belgenin sonunu sahte yere taşıyan,
    kuyruğu sessizce atan bir kayıt); şimdi hata verilir ve cache yazılmaz.
    """
    cache_dir = tmp_path / "preprocessed_data"
    data_cache = DataCache(data_dir=str(corpus_dir), cache_dir=str(cache_dir), cache_enabled=True)
    config = dict(training_service.config, max_seq_length=TRUNCATED_MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="chunk it first"):
        training_service.prepare_from_cache(
            data_cache=data_cache,
            tokenizer_core=training_service.tokenizer_core,
            config=config,
        )
    assert not list(cache_dir.glob("cached_data_*.pkl")), "Reddedilen biçimleme cache yazmamalı"


def test_local_training_cache_examples_are_formatted(request):
    """
    Yerel eğitim cache'indeki örnekler (ilk 1000) autoregressive formatta mı?
    Cache repo'da yoktur (training_system/prepare_cache.py üretir); yoksa test atlanır.
    """
    cache_files = sorted(LOCAL_CACHE_DIR.glob("cached_data_*.pkl"), key=lambda p: p.stat().st_mtime)
    if not cache_files:
        pytest.skip(
            f"Yerel eğitim cache'i yok: {LOCAL_CACHE_DIR}/cached_data_*.pkl "
            "(training_system/prepare_cache.py ile oluşturulur)"
        )

    # En son üretilen cache dosyasını yükle
    cache_file = cache_files[-1]
    ids = special_token_ids(request.getfixturevalue("tokenizer_core"))
    with open(cache_file, "rb") as f:
        raw_data = pickle.load(f)
    assert raw_data, f"Cache boş: {cache_file.name}"

    samples = raw_data[:MAX_ANALYZED_EXAMPLES]
    failures = collect_format_failures(samples, ids)
    assert not failures, (
        f"{cache_file.name}: {len(failures)}/{len(samples)} örnek hatalı formatlı "
        f"(special ids={ids}); ilk 5: {failures[:5]}"
    )


def test_loss_function_eos_weight(training_service):
    """
    Loss function'da EOS weight'in uygulanıp uygulanmadığını test et.
    Eski test TrainingService'teki loss function kurulumunu kendi içinde yeniden yazıyordu; bu kurulum
    artık CriterionManager'da ve TrainingService.criterion üzerinden doğrudan test edilir.
    """
    criterion = training_service.criterion
    vocab = training_service.tokenizer_core.get_vocab()
    vocab_size = len(vocab)

    # EOS ID'yi al (TrainingService ile aynı yöntem)
    eos_data = vocab.get("<EOS>")
    eos_id = eos_data.get("id") if isinstance(eos_data, dict) else eos_data
    assert isinstance(eos_id, int) and 0 <= eos_id < vocab_size, (
        f"EOS ID geçersiz: {eos_id}, vocab_size={vocab_size}"
    )

    # TrainingService criterion'ı ModelManager'a da veriyor (eğitimde kullanılan criterion bu)
    assert training_service.model_manager.criterion is criterion

    # EOS weight tensor'u: EOS için yapılandırılan değer, diğer token'lar 1.0
    loss_weights = criterion.weight
    assert loss_weights is not None and tuple(loss_weights.shape) == (vocab_size,), (
        f"Loss weight tensor'u yok ya da boyutu yanlış: {None if loss_weights is None else tuple(loss_weights.shape)}"
    )
    assert abs(loss_weights[eos_id].item() - EOS_WEIGHT) < 1e-6, (
        f"EOS weight uygulanmamış: weight[{eos_id}]={loss_weights[eos_id].item()}, beklenen {EOS_WEIGHT}"
    )
    other_weights = torch.cat([loss_weights[:eos_id], loss_weights[eos_id + 1:]])
    assert torch.all(other_weights == 1.0), "EOS dışındaki token weight'leri 1.0 olmalı"
    assert criterion.label_smoothing == pytest.approx(LABEL_SMOOTHING), (
        f"Label smoothing: {criterion.label_smoothing}, beklenen {LABEL_SMOOTHING}"
    )
    assert criterion.ignore_index == PAD_TOKEN_ID, f"ignore_index PAD olmalı: {criterion.ignore_index}"

    # Test: Dummy logits ve target ile loss hesapla
    batch_size = 4
    seq_len = 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(5, vocab_size, (batch_size, seq_len))
    targets[:, -1] = eos_id  # Son pozisyona EOS koy

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    loss = criterion(logits_flat, targets_flat)
    loss_without_eos_weight = F.cross_entropy(
        logits_flat, targets_flat, label_smoothing=LABEL_SMOOTHING, ignore_index=PAD_TOKEN_ID
    )
    logger.info("Test loss: %.4f (EOS weight'siz: %.4f), EOS pozisyonları: %d/%d",
                loss.item(), loss_without_eos_weight.item(), batch_size, batch_size * seq_len)

    assert torch.isfinite(loss), f"Loss sonlu değil: {loss.item()}"
    assert abs(loss.item() - loss_without_eos_weight.item()) > 1e-5, (
        "EOS weight loss değerini değiştirmiyor - loss function'da UYGULANMIYOR"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
