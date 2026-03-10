import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.attention_utils_module.attention_initializer import AttentionInitializer


@pytest.fixture
def initializer():
    """
    Testler için bir AttentionInitializer örneği döndürür.
    """
    return AttentionInitializer(initialization_type="xavier", seed=42, verbose=False)


def test_initialize_weights_xavier(initializer):
    """
    Xavier başlatma türü için ağırlık başlatma testini yapar.
    """
    tensor = torch.empty(4, 4)
    initialized_tensor = initializer.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert not torch.isnan(initialized_tensor).any()
    assert not torch.isinf(initialized_tensor).any()


def test_initialize_weights_he():
    """
    He başlatma türü için ağırlık başlatma testini yapar.
    """
    initializer = AttentionInitializer(initialization_type="he", seed=42, verbose=False)
    tensor = torch.empty(4, 4)
    initialized_tensor = initializer.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert not torch.isnan(initialized_tensor).any()
    assert not torch.isinf(initialized_tensor).any()


def test_initialize_param_matrix(initializer):
    """
    Parametre matrisinin doğru şekilde başlatıldığını test eder.
    """
    input_dim, output_dim = 4, 5
    param_matrix = initializer.initialize_param_matrix(input_dim, output_dim)
    assert param_matrix.shape == (input_dim, output_dim)
    assert not torch.isnan(param_matrix).any()
    assert not torch.isinf(param_matrix).any()


def test_initialize_bias(initializer):
    """
    Bias vektörünün doğru şekilde başlatıldığını test eder.
    """
    size = 5
    bias = initializer.initialize_bias(size)
    assert bias.shape == (size,)
    assert torch.equal(bias, torch.zeros(size))


def test_invalid_initialization_type():
    """
    Geçersiz başlatma türüyle oluşturulan initializer'ın hata verdiğini test eder.
    """
    with pytest.raises(ValueError):
        AttentionInitializer(initialization_type="invalid")


def test_invalid_tensor_type(initializer):
    """
    Geçersiz bir veri tipiyle ağırlık başlatma fonksiyonunun hata verdiğini test eder.
    """
    with pytest.raises(TypeError):
        initializer.initialize_weights([1, 2, 3])  # Tensor yerine liste verildi


def test_validate_tensor_valid(initializer):
    """
    Geçerli bir tensörün doğrulandığını test eder.
    """
    tensor = torch.rand(4, 4)
    assert initializer.validate_tensor(tensor)


def test_validate_tensor_nan(initializer):
    """
    NaN değerler içeren tensörün doğrulamada hata verdiğini test eder.
    """
    tensor = torch.tensor([[float('nan'), 1.0]])
    with pytest.raises(ValueError):
        initializer.validate_tensor(tensor)


def test_validate_tensor_inf(initializer):
    """
    Sonsuz değerler içeren tensörün doğrulamada hata verdiğini test eder.
    """
    tensor = torch.tensor([[float('inf'), 1.0]])
    with pytest.raises(ValueError):
        initializer.validate_tensor(tensor)


def test_log_initialization_details(initializer, capsys):
    """
    Başlatma detaylarının doğru şekilde loglandığını test eder.
    """
    initializer.verbose = True
    tensor = torch.rand(4, 4)
    initializer.log_initialization_details(tensor, description="Test Tensor")
    captured = capsys.readouterr()
    assert "Test Tensor başlatıldı:" in captured.out
    assert "Şekil" in captured.out
