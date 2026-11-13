"""
KRİTİK TEST: TrainingManager Loss Function Doğrulama
=====================================================

Bu test, TrainingManager'ın loss hesaplamasında EOS weight'in gerçekten uygulanıp uygulanmadığını doğrular.

ENDÜSTRİ STANDARTLARI:
- ISO/IEC/IEEE 29119-3: Unit Test Coverage
- Test: Loss function'ın doğru kullanıldığını doğrula
- Test: EOS weight'in gradient'lere etkisini doğrula
- Test: Training loop'ta criterion kullanılıyor mu?

KRİTİK SORU:
- TrainingService'te EOS weight'li criterion oluşturuluyor
- AMA TrainingManager'da kullanılıyor mu?
- Eski TrainingManager'ın _compute_masked_loss_and_acc metodu F.cross_entropy kullanıyordu -
  criterion kullanılmıyor muydu?

GÜNCEL DURUM (V2):
- TrainingManager training_management/v2/core/training_manager.py'ye taşındı.
- Loss artık LossComputation.compute_loss ile hesaplanıyor (_compute_masked_loss_and_acc'nin
  yerini aldı); criterion'ın weight ve label_smoothing değerlerini kullanır, PAD'i maskeler ve
  geçerli token sayısına böler (token-ortalaması). nn.CrossEntropyLoss(reduction="mean") ise
  ağırlıklı ortalama alır (hedeflerin weight toplamına böler); bu yüzden iki değer EOS weight
  1.0'dan farklıyken birebir eşit değildir, aynı token terimlerini kullanırlar.
"""

import copy
import logging
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

# Proje kök dizinini sys.path'e ekle (dosya doğrudan çalıştırıldığında da import'lar çalışsın)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Eski yol training_management/training_manager.py artık yok; V2 TrainingManager
from training_management.v2.core.training_manager import TrainingManager
from training_management.v2.core.loss_computation import LossComputation

# Modül seviyesinde logging.basicConfig yok; standart logger yeterli (pytest yakalar).
logger = logging.getLogger("TrainingManagerLossFunctionTest")

# float32 toplamlarında beklenen sayısal hata ~1e-6
TOLERANCE = 1e-5


class _StdTrainingLogger(logging.LoggerAdapter):
    """TrainingManager bileşenlerinin beklediği log_* arayüzü, standart logging'e yazar.

    Varsayılan TrainingLogger süreç genelinde tekildir (singleton) ve tensorboard
    kuruluysa ./logs/tb altına yazar; testler repo ağacına yazmamalı.
    """

    def __init__(self, name: str):
        super().__init__(logging.getLogger(name), {})

    def log_info(self, msg, *args, **kwargs):
        self.info(msg)

    def log_warning(self, msg, *args, **kwargs):
        self.warning(msg)

    def log_error(self, msg, *args, exc_info=False, **kwargs):
        self.error(msg, exc_info=exc_info)

    def log_debug(self, msg, *args, **kwargs):
        self.debug(msg)


@pytest.fixture(autouse=True)
def _seeded_rng():
    """Mock model ve dummy veri deterministik olsun; global RNG durumu test sonrası geri yüklenir."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


class TestTrainingManagerLossFunction:
    """TrainingManager loss function doğrulama testi"""

    def setup_method(self):
        self.vocab_size = 60000
        self.eos_id = 3
        self.pad_id = 0
        self.bos_id = 2
        # Ürün varsayılanı artık eos_token_weight=1.0 (CriterionManager: 0.1 EOS öğrenimini
        # bozuyordu). 1.0 ağırlık etkisizdir; mekanizmanın gözlenebilmesi için 0.1 kullanılır.
        self.eos_weight = 0.1
        self.label_smoothing = 0.1

    def create_mock_model(self):
        """Mock model oluştur"""
        class MockModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size
                self.embedding = torch.nn.Embedding(1000, 512)
                self.linear = torch.nn.Linear(512, vocab_size)

            def forward(self, inputs):
                # Basit forward pass - [B, T] → [B, T, V]
                x = self.embedding(inputs)  # [B, T, 512]
                logits = self.linear(x)  # [B, T, V]
                return logits

        return MockModel(self.vocab_size)

    def create_criterion_with_eos_weight(self):
        """EOS weight'li criterion oluştur (TrainingService gibi)"""
        loss_weights = torch.ones(self.vocab_size)
        loss_weights[self.eos_id] = self.eos_weight

        return CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.pad_id,
            reduction="mean"
        )

    def create_dummy_batch(self, batch_size: int = 4, seq_len: int = 10):
        """Dummy input/target; her satırın son içerik pozisyonu EOS, bir satırda PAD dolgusu var."""
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        # Özel token'ları (0-4) rastgele içerikten hariç tut; EOS/PAD yerleri belirli olsun
        targets = torch.randint(5, self.vocab_size, (batch_size, seq_len))
        targets[:, -1] = self.eos_id  # Son pozisyona EOS koy
        # Kısa bir örnek: EOS'tan sonra PAD (criterion ignore_index / TrainingManager maskesi)
        targets[1, seq_len - 4] = self.eos_id
        targets[1, seq_len - 3:] = self.pad_id
        return inputs, targets

    def test_training_manager_loss_computation(self, tmp_path):
        """
        TrainingManager'ın loss hesaplamasını test et
        KRİTİK: TrainingManager criterion'ın EOS weight'ini ve label smoothing'ini kullanıyor mu,
        yoksa düz F.cross_entropy mi?
        """
        # Mock model
        model = self.create_mock_model()

        # EOS weight'li criterion (TrainingService gibi)
        criterion = self.create_criterion_with_eos_weight()

        # TrainingManager config
        config = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_id,
            "device": "cpu",
            "use_amp": False,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "epochs": 1,
            "enable_file_logging": False,
            # Repo ağacına yazmamak için: varsayılan ./checkpoints ve ./runs yerine
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "enable_tensorboard": False,
        }

        # Dummy data
        inputs, targets = self.create_dummy_batch()
        dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=inputs.size(0), shuffle=False)
        val_loader = DataLoader(dataset, batch_size=inputs.size(0), shuffle=False)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # TrainingManager oluştur
        training_manager = TrainingManager(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,  # EOS weight'li criterion veriyoruz
            config=config,
            logger=_StdTrainingLogger(logger.name),
        )

        # KRİTİK: Training loop'un kullandığı loss hesaplayıcı, verdiğimiz criterion'ı taşıyor mu?
        assert training_manager.loss_computation.criterion is criterion, (
            "TrainingManager verilen (EOS weight'li) criterion'ı kullanmıyor"
        )
        assert training_manager.training_loop.loss_computation is training_manager.loss_computation, (
            "Training loop, TrainingManager'ın LossComputation'ını kullanmıyor"
        )

        # Model'den logits al
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # TrainingManager'ın loss hesaplama yolu (eski _compute_masked_loss_and_acc)
        loss_tm, acc_tm, ppl_tm = training_manager.loss_computation.compute_loss(
            logits=logits,
            targets=targets,
            pad_id=self.pad_id
        )

        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        valid = targets_flat != self.pad_id
        num_valid = int(valid.sum())

        # MANUEL HESAPLAMA: Criterion ile (EOS weight'li, hedef weight toplamına bölünmüş ortalama)
        loss_criterion = criterion(logits_flat, targets_flat)

        # MANUEL HESAPLAMA: F.cross_entropy ile (EOS weight YOK, label smoothing YOK)
        loss_f_cross_entropy = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_id)

        # MANUEL HESAPLAMA: criterion'ın token terimleri (weight + label smoothing), token-ortalaması
        per_token_weighted = F.cross_entropy(
            logits_flat, targets_flat,
            weight=criterion.weight, label_smoothing=self.label_smoothing, reduction="none",
        )
        expected_tm_loss = per_token_weighted[valid].sum() / num_valid

        # Aynı hesap EOS weight olmadan (label smoothing aynı) - tek fark EOS weight
        per_token_unweighted = F.cross_entropy(
            logits_flat, targets_flat, label_smoothing=self.label_smoothing, reduction="none",
        )
        expected_without_eos_weight = per_token_unweighted[valid].sum() / num_valid

        eos_positions = int((targets_flat == self.eos_id).sum())
        logger.info(
            "Loss karşılaştırması: TM=%.6f, criterion=%.6f, F.cross_entropy=%.6f, "
            "beklenen (weight'li token-ortalaması)=%.6f, EOS weight'siz=%.6f, EOS pozisyonu=%d/%d",
            loss_tm.item(), loss_criterion.item(), loss_f_cross_entropy.item(),
            expected_tm_loss.item(), expected_without_eos_weight.item(), eos_positions, num_valid,
        )

        # 1) TrainingManager loss'u = criterion'ın weight + label smoothing'iyle token-ortalaması
        assert math.isclose(loss_tm.item(), expected_tm_loss.item(), rel_tol=TOLERANCE, abs_tol=TOLERANCE), (
            f"TrainingManager loss'u ({loss_tm.item():.6f}) criterion'ın weight/label_smoothing'iyle "
            f"beklenen değerden ({expected_tm_loss.item():.6f}) farklı"
        )

        # 2) Criterion ile aynı token terimleri: criterion weight toplamına, TM geçerli token sayısına böler
        weight_sum = float(criterion.weight[targets_flat[valid]].sum())
        assert math.isclose(loss_tm.item() * num_valid, loss_criterion.item() * weight_sum, rel_tol=TOLERANCE), (
            f"TM toplam loss ({loss_tm.item() * num_valid:.6f}) criterion'ın toplam loss'una "
            f"({loss_criterion.item() * weight_sum:.6f}) eşit değil - criterion terimleri kullanılmıyor"
        )

        # 3) EOS weight gerçekten uygulanıyor: EOS weight'siz aynı hesaptan farklı
        assert abs(loss_tm.item() - expected_without_eos_weight.item()) > TOLERANCE, (
            "TrainingManager loss'u EOS weight'siz loss ile aynı - EOS weight UYGULANMIYOR"
        )

        # 4) TrainingManager düz F.cross_entropy kullanmıyor (eski V1 hatası)
        assert abs(loss_tm.item() - loss_f_cross_entropy.item()) > TOLERANCE, (
            "KRİTİK HATA: TrainingManager criterion kullanmıyor, düz F.cross_entropy kullanıyor"
        )

        # Accuracy ve perplexity aynı (maskeli) loss'tan türetilir
        assert 0.0 <= acc_tm <= 1.0, f"Accuracy aralık dışı: {acc_tm}"
        assert math.isclose(ppl_tm, math.exp(min(20.0, loss_tm.item())), rel_tol=1e-6), (
            f"Perplexity ({ppl_tm}) loss'tan türetilmemiş (exp(loss)={math.exp(min(20.0, loss_tm.item()))})"
        )

    def test_gradient_impact(self):
        """
        EOS weight'in gradient'lere etkisini test et (TrainingManager'ın loss hesaplama yolu ile)
        """
        # Mock model. Karşılaştırma aynı başlangıç ağırlıklarıyla yapılmalı: eski test ikinci
        # modeli ayrıca rastgele başlatıyordu, gradient farkı EOS weight'ten değil init'ten geliyordu.
        model = self.create_mock_model()
        model2 = copy.deepcopy(model)

        # EOS weight'li criterion
        criterion = self.create_criterion_with_eos_weight()

        # EOS weight olmadan criterion (karşılaştırma için). Label smoothing aynı tutulur ki
        # tek fark EOS weight olsun (eski test label smoothing'i de kaldırıyordu).
        criterion_no_weight = CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
        )

        # Dummy data
        inputs, targets = self.create_dummy_batch()

        def loss_and_grad_norm(net, crit):
            net.train()
            net.zero_grad()
            # TrainingManager/TrainingLoop'un kullandığı LossComputation
            loss, _, _ = LossComputation(crit).compute_loss(net(inputs), targets, pad_id=self.pad_id)
            loss.backward()
            grads = {name: p.grad.detach().clone() for name, p in net.named_parameters() if p.grad is not None}
            total_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads.values()))
            return loss.item(), total_norm, grads

        loss_w, grad_norm_w, grads_w = loss_and_grad_norm(model, criterion)
        loss_nw, grad_norm_nw, grads_nw = loss_and_grad_norm(model2, criterion_no_weight)

        logger.info(
            "Gradient analizi: loss (EOS weight'li)=%.6f, loss (EOS weight YOK)=%.6f, "
            "grad norm (EOS weight'li)=%.6f, grad norm (EOS weight YOK)=%.6f",
            loss_w, loss_nw, grad_norm_w, grad_norm_nw,
        )

        assert math.isfinite(loss_w) and math.isfinite(loss_nw), f"Loss sonlu değil: {loss_w}, {loss_nw}"
        assert math.isfinite(grad_norm_w) and math.isfinite(grad_norm_nw), (
            f"Gradient norm sonlu değil: {grad_norm_w}, {grad_norm_nw}"
        )
        assert set(grads_w) == set(grads_nw) == {name for name, _ in model.named_parameters()}, (
            "Tüm parametreler gradient almalı"
        )

        # EOS weight loss'u ve gradient'leri değiştirmeli
        assert abs(loss_w - loss_nw) > TOLERANCE, "EOS weight loss'u değiştirmiyor"
        assert abs(grad_norm_w - grad_norm_nw) > TOLERANCE, (
            f"EOS weight gradient norm'unu değiştirmiyor ({grad_norm_w:.6f} vs {grad_norm_nw:.6f})"
        )
        assert not torch.allclose(grads_w["linear.weight"], grads_nw["linear.weight"]), (
            "EOS weight çıkış katmanı gradient'lerini değiştirmiyor"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
