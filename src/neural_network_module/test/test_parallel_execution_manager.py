import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_manager import ParallelExecutionManager

@pytest.fixture
def test_tensor():
    return torch.randn(32, 10, 128)

@pytest.fixture
def parallel_execution_manager():
    task_dim = 2  # 
    return ParallelExecutionManager(num_tasks=4, task_dim=task_dim, learning_rate=0.01, scale_range=(0, 1), log_level=logging.DEBUG)


def test_initialize(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.initialize(tensor)
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_scale_min_max(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.scale(tensor, method="min_max")
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.min() >= 0
        assert task.max() <= 1
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_scale_standard(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.scale(tensor, method="standard")
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.mean().abs() < 1e-6
        assert task.std().abs() - 1 < 1e-6
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_scale_robust(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.scale(tensor, method="robust")
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_schedule(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.schedule(tensor)
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_balance(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    tasks = parallel_execution_manager.balance(tensor)
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_optimize(parallel_execution_manager):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = parallel_execution_manager.optimize(tensor, gradients)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device


def test_invalid_tensor_type(parallel_execution_manager):
    with pytest.raises(TypeError):
        parallel_execution_manager.initialize("invalid_input")

    with pytest.raises(TypeError):
        parallel_execution_manager.scale("invalid_input", method="min_max")

    with pytest.raises(TypeError):
        parallel_execution_manager.schedule("invalid_input")

    with pytest.raises(TypeError):
        parallel_execution_manager.balance("invalid_input")

    with pytest.raises(TypeError):
        parallel_execution_manager.optimize("invalid_input", torch.randn(100, 64))

    with pytest.raises(TypeError):
        parallel_execution_manager.optimize(torch.randn(100, 64), "invalid_input")

def test_log_execution(parallel_execution_manager, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)

    parallel_execution_manager.initialize(tensor)
    parallel_execution_manager.scale(tensor, method="min_max")
    parallel_execution_manager.schedule(tensor)
    parallel_execution_manager.balance(tensor)
    parallel_execution_manager.optimize(tensor, gradients)

    assert "Initialization completed." in caplog.text
    assert "Scaling completed." in caplog.text
    assert "Scheduling completed." in caplog.text
    assert "Load Balancing completed." in caplog.text
    assert "Optimization completed." in caplog.text

def test_large_tensor_parallel_execution(parallel_execution_manager):
    """
    Büyük boyutlu tensörlerle paralel çalıştırma test edilir.
    """
    tensor = torch.randn(10000, 512)  # Büyük tensör
    tasks = parallel_execution_manager.initialize(tensor)

    assert len(tasks) == 4
    assert sum(task.size(0) for task in tasks) == tensor.size(0)  # Boyut korunmalı


def test_parallel_execution_memory_leak(parallel_execution_manager):
    """
    Paralel işlemler sırasında bellek sızıntısı olup olmadığını test eder.
    """
    import gc

    tensor = torch.randn(5000, 512)
    before_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    _ = parallel_execution_manager.initialize(tensor)
    gc.collect()  # Garbage Collector çalıştır

    after_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert after_memory <= before_memory  # Bellek sızıntısı olmamalı


def test_parallel_execution_exception_handling(parallel_execution_manager):
    """
    Paralel işlemler sırasında oluşabilecek hataların düzgün şekilde yakalanıp yakalanmadığını test eder.
    """
    tensor = torch.randn(100, 64)

    try:
        tasks = parallel_execution_manager.initialize(tensor)
        assert len(tasks) == 4
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_parallel_execution_with_mismatched_gradients(parallel_execution_manager):
    """
    Optimizasyon sırasında tensör ve gradyan boyutlarının eşleşmediği durumda hata alınıp alınmadığını test eder.
    """
    tensor = torch.randn(100, 64)
    gradients = torch.randn(50, 64)  # Farklı boyutta gradient (hata bekliyoruz)

    with pytest.raises(RuntimeError):
        parallel_execution_manager.optimize(tensor, gradients)




def test_parallel_execution_with_non_divisible_tasks(parallel_execution_manager):
    """
    Paralel işlemlerde görev sayısının, tensör boyutuna tam bölünmediği durumda nasıl çalıştığını test eder.
    """
    tensor = torch.randn(103, 64)  # 103 sayısı 4'e tam bölünmüyor
    tasks = parallel_execution_manager.initialize(tensor)

    assert len(tasks) == 4
    assert sum(task.size(0) for task in tasks) == 103  # Toplam eleman sayısı korunmalı

def test_parallel_execution_with_batch_processing(parallel_execution_manager):
    """
    Farklı tensörlerin batch halinde işlenmesi test edilir.
    """
    batch_size = 5
    tensors = [torch.randn(100, 64) for _ in range(batch_size)]

    results = []
    for tensor in tensors:
        tasks = parallel_execution_manager.initialize(tensor)
        results.append(tasks)

    assert len(results) == batch_size
    for tasks in results:
        assert len(tasks) == 4  # Her tensör 4 parçaya bölünmeli

# 1. Paralel bölme işlemi sonrası toplam batch sayısı korunuyor mu?
def test_parallel_task_splitting_correctness(parallel_execution_manager, test_tensor):
    tasks = parallel_execution_manager.initialize(test_tensor)
    assert sum(t.shape[0] for t in tasks) == test_tensor.shape[0], "Paralel bölme sonrası toplam batch boyutu değişti!"

# 2. Paralel işlemler sonrası çıktı şekli yanlış mı?
def test_parallel_execution_output_shape_issue(parallel_execution_manager, test_tensor):
    tasks = parallel_execution_manager.initialize(test_tensor)
    merged_output = torch.cat(tasks, dim=-1)
    expected_dim = 128 * parallel_execution_manager.initializer.num_tasks
    assert merged_output.shape[-1] == expected_dim, f"Yanlış birleştirme! Beklenen {expected_dim}, ama {merged_output.shape[-1]}"

# 3. Paralel yük dengeleme sonucunda her task dengeli mi?
def test_parallel_execution_load_balancing_consistency(parallel_execution_manager, test_tensor):
    tasks = parallel_execution_manager.balance(test_tensor)
    avg_size = test_tensor.shape[0] // 4
    assert all(abs(t.shape[0] - avg_size) <= 1 for t in tasks), "Paralel yük dengesi hatalı!"

# 4. num_tasks değeri değiştirildiğinde çıktı boyutu doğru hesaplanıyor mu?
@pytest.mark.parametrize("num_tasks", [2, 4, 8, 16])
def test_parallel_execution_with_different_num_tasks(num_tasks):
    manager = ParallelExecutionManager(num_tasks=num_tasks, task_dim=2, learning_rate=0.01, scale_range=(0, 1), log_level=logging.DEBUG)
    tensor = torch.randn(32, 10, 128)
    tasks = manager.initialize(tensor)
    total_size = sum(task.shape[0] for task in tasks)
    assert total_size == tensor.shape[0], f"Yanlış çıktı boyutu! Beklenen {tensor.shape[0]}, ama {total_size}"

# 5. Paralel optimize işleminde tensör ve gradient boyutları uyumlu mu?
def test_parallel_execution_optimize_shape(parallel_execution_manager, test_tensor):
    gradients = torch.randn_like(test_tensor)
    optimized_tensor = parallel_execution_manager.optimize(test_tensor, gradients)
    assert optimized_tensor.shape == test_tensor.shape, "Optimizasyon sonrası tensör boyutu değişmemeli!"

# 6. num_tasks değiştikçe çıktılar sabitleniyor mu?
@pytest.mark.parametrize("num_tasks", [2, 4, 8, 16])
def test_parallel_execution_output_consistency(num_tasks):
    manager = ParallelExecutionManager(num_tasks=num_tasks, task_dim=2, learning_rate=0.01, scale_range=(0, 1), log_level=logging.DEBUG)
    tensor = torch.randn(32, 10, 128)
    tasks = manager.initialize(tensor)
    merged_output = torch.cat(tasks, dim=-1)
    assert merged_output.shape[-1] == 128 * num_tasks, f"Yanlış çıktı boyutu! Beklenen {128 * num_tasks}, ama {merged_output.shape[-1]}"

# 7. Bellek tüketimi paralel işlemler sonrası artıyor mu?
def test_parallel_execution_memory_consumption(parallel_execution_manager, test_tensor):
    import gc
    before_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    _ = parallel_execution_manager.initialize(test_tensor)
    gc.collect()
    after_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert after_memory <= before_memory, "Bellek tüketimi anormal şekilde arttı!"



# 10. Paralel optimizasyonda yanlış gradyan boyutu hata veriyor mu?
def test_parallel_execution_mismatched_gradient_shapes(parallel_execution_manager, test_tensor):
    wrong_gradients = torch.randn(16, 10, 128)
    with pytest.raises(RuntimeError):
        parallel_execution_manager.optimize(test_tensor, wrong_gradients)
        

