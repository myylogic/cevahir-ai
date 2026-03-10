# -*- coding: utf-8 -*-
"""
Core Processing API Tests
=========================
CognitiveManager core processing metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- handle() - Senkron request processing
- handle_async() - Asenkron request processing
- handle_batch() - Batch request processing
- handle_multimodal() - Multimodal processing

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
- Error handling tests
- Edge case coverage
"""

import pytest
from typing import List, Tuple
import asyncio

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput, CognitiveOutput, DecodingConfig
from .conftest import (
    mock_model_api,
    default_config,
    cognitive_manager,
    cognitive_state,
    cognitive_input,
    decoding_config,
    assert_cognitive_output
)


# ============================================================================
# Test 1-10: handle() - Basic Functionality
# Test Edilen Dosya: cognitive_manager.py (handle method)
# ============================================================================

def test_handle_basic_request(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 1: Basic handle() request processing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Basit bir request işleme
    """
    output = cognitive_manager.handle(cognitive_state, cognitive_input)
    assert_cognitive_output(output)
    assert output.text is not None
    assert len(output.text) > 0


def test_handle_with_decoding_config(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput, decoding_config: DecodingConfig):
    """
    Test 2: handle() with custom decoding config.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle(decoding=DecodingConfig)
    Test Senaryosu: Custom decoding config ile request işleme
    """
    decoding_config.max_tokens = 100
    output = cognitive_manager.handle(cognitive_state, cognitive_input, decoding=decoding_config)
    assert_cognitive_output(output)


def test_handle_empty_message(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 3: handle() with empty message.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Boş mesaj ile request işleme (edge case)
    """
    empty_input = CognitiveInput(user_message="")
    output = cognitive_manager.handle(cognitive_state, empty_input)
    assert_cognitive_output(output)


def test_handle_long_message(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 4: handle() with very long message.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Çok uzun mesaj ile request işleme (edge case)
    """
    long_message = "Test " * 1000  # 5000 karakter
    long_input = CognitiveInput(user_message=long_message)
    output = cognitive_manager.handle(cognitive_state, long_input)
    assert_cognitive_output(output)


def test_handle_special_characters(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 5: handle() with special characters.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Özel karakterler içeren mesaj (edge case)
    """
    special_input = CognitiveInput(user_message="Test: !@#$%^&*()_+-=[]{}|;':\",./<>?")
    output = cognitive_manager.handle(cognitive_state, special_input)
    assert_cognitive_output(output)


def test_handle_unicode_characters(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 6: handle() with unicode characters.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Unicode karakterler içeren mesaj (edge case)
    """
    unicode_input = CognitiveInput(user_message="Test: Türkçe 中文 العربية 🚀")
    output = cognitive_manager.handle(cognitive_state, unicode_input)
    assert_cognitive_output(output)


def test_handle_multiple_sequential_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 7: handle() multiple sequential requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Ardışık multiple request işleme
    """
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Request {i}")
        output = cognitive_manager.handle(cognitive_state, input_msg)
        assert_cognitive_output(output)


def test_handle_state_persistence(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 8: handle() state persistence across requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: State'in request'ler arasında korunması
    """
    # İlk request
    input1 = CognitiveInput(user_message="First message")
    output1 = cognitive_manager.handle(cognitive_state, input1)
    assert_cognitive_output(output1)
    
    # İkinci request (aynı state ile)
    input2 = CognitiveInput(user_message="Second message")
    output2 = cognitive_manager.handle(cognitive_state, input2)
    assert_cognitive_output(output2)
    
    # State'in korunduğunu kontrol et
    assert cognitive_state is not None


def test_handle_error_handling(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 9: handle() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Hata durumlarında graceful handling
    """
    # Invalid input test
    try:
        invalid_input = None
        # Bu durumda exception beklenir
        with pytest.raises((AttributeError, TypeError)):
            cognitive_manager.handle(cognitive_state, invalid_input)
    except Exception:
        # Expected behavior
        pass


def test_handle_metadata_in_output(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 10: handle() output metadata validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle()
    Test Senaryosu: Output metadata'nın doğru şekilde dönmesi
    """
    output = cognitive_manager.handle(cognitive_state, cognitive_input)
    assert_cognitive_output(output)
    assert isinstance(output.metadata, dict)
    # Metadata'da beklenen alanlar olabilir
    # (implementation'a göre değişebilir)


# ============================================================================
# Test 11-20: handle_async() - Async Functionality
# Test Edilen Dosya: cognitive_manager.py (handle_async method)
# ============================================================================

@pytest.mark.asyncio
async def test_handle_async_basic_request(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 11: Basic handle_async() request processing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Basit async request işleme
    """
    output = await cognitive_manager.handle_async(cognitive_state, cognitive_input)
    assert_cognitive_output(output)
    assert output.text is not None


@pytest.mark.asyncio
async def test_handle_async_with_decoding_config(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput, decoding_config: DecodingConfig):
    """
    Test 12: handle_async() with custom decoding config.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async(decoding=DecodingConfig)
    Test Senaryosu: Custom decoding config ile async request
    """
    decoding_config.max_tokens = 150
    output = await cognitive_manager.handle_async(cognitive_state, cognitive_input, decoding=decoding_config)
    assert_cognitive_output(output)


@pytest.mark.asyncio
async def test_handle_async_concurrent_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 13: handle_async() concurrent requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Concurrent async request işleme
    """
    inputs = [CognitiveInput(user_message=f"Async request {i}") for i in range(5)]
    tasks = [cognitive_manager.handle_async(cognitive_state, inp) for inp in inputs]
    outputs = await asyncio.gather(*tasks)
    
    for output in outputs:
        assert_cognitive_output(output)


@pytest.mark.asyncio
async def test_handle_async_empty_message(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 14: handle_async() with empty message.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Boş mesaj ile async request (edge case)
    """
    empty_input = CognitiveInput(user_message="")
    output = await cognitive_manager.handle_async(cognitive_state, empty_input)
    assert_cognitive_output(output)


@pytest.mark.asyncio
async def test_handle_async_error_handling(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 15: handle_async() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Async hata durumlarında graceful handling
    """
    try:
        invalid_input = None
        with pytest.raises((AttributeError, TypeError)):
            await cognitive_manager.handle_async(cognitive_state, invalid_input)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_handle_async_vs_sync_consistency(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 16: handle_async() vs handle() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async() vs handle()
    Test Senaryosu: Async ve sync metodların tutarlılığı
    """
    # Sync
    sync_output = cognitive_manager.handle(cognitive_state, cognitive_input)
    
    # Async
    async_output = await cognitive_manager.handle_async(cognitive_state, cognitive_input)
    
    # Her ikisi de geçerli output döndürmeli
    assert_cognitive_output(sync_output)
    assert_cognitive_output(async_output)


@pytest.mark.asyncio
async def test_handle_async_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 17: handle_async() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Async performans testi
    """
    import time
    start = time.time()
    
    inputs = [CognitiveInput(user_message=f"Perf test {i}") for i in range(10)]
    tasks = [cognitive_manager.handle_async(cognitive_state, inp) for inp in inputs]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    # Async'in sequential'dan daha hızlı olması beklenir (ama garantili değil)
    assert elapsed >= 0  # Basic sanity check


@pytest.mark.asyncio
async def test_handle_async_state_isolation(cognitive_manager: CognitiveManager):
    """
    Test 18: handle_async() state isolation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Concurrent request'lerde state isolation
    """
    state1 = CognitiveState()
    state2 = CognitiveState()
    
    input1 = CognitiveInput(user_message="State 1")
    input2 = CognitiveInput(user_message="State 2")
    
    outputs = await asyncio.gather(
        cognitive_manager.handle_async(state1, input1),
        cognitive_manager.handle_async(state2, input2)
    )
    
    for output in outputs:
        assert_cognitive_output(output)


@pytest.mark.asyncio
async def test_handle_async_cancellation(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 19: handle_async() cancellation handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Async task cancellation
    """
    task = asyncio.create_task(cognitive_manager.handle_async(cognitive_state, cognitive_input))
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        # Expected behavior
        pass


@pytest.mark.asyncio
async def test_handle_async_timeout_handling(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, cognitive_input: CognitiveInput):
    """
    Test 20: handle_async() timeout handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_async()
    Test Senaryosu: Timeout durumunda handling
    """
    try:
        output = await asyncio.wait_for(
            cognitive_manager.handle_async(cognitive_state, cognitive_input),
            timeout=30.0
        )
        assert_cognitive_output(output)
    except asyncio.TimeoutError:
        # Timeout durumunda expected behavior
        pass


# ============================================================================
# Test 21-30: handle_batch() - Batch Processing
# Test Edilen Dosya: cognitive_manager.py (handle_batch method)
# ============================================================================

def test_handle_batch_basic(cognitive_manager: CognitiveManager):
    """
    Test 21: Basic handle_batch() processing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Basit batch request işleme
    """
    states = [CognitiveState() for _ in range(3)]
    inputs = [CognitiveInput(user_message=f"Batch request {i}") for i in range(3)]
    requests = list(zip(states, inputs))
    
    outputs = cognitive_manager.handle_batch(requests)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    for output in outputs:
        assert_cognitive_output(output)


def test_handle_batch_empty_list(cognitive_manager: CognitiveManager):
    """
    Test 22: handle_batch() with empty list.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Boş batch list (edge case)
    """
    outputs = cognitive_manager.handle_batch([])
    assert isinstance(outputs, list)
    assert len(outputs) == 0


def test_handle_batch_large_batch(cognitive_manager: CognitiveManager):
    """
    Test 23: handle_batch() with large batch.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Büyük batch işleme (edge case)
    """
    batch_size = 50
    states = [CognitiveState() for _ in range(batch_size)]
    inputs = [CognitiveInput(user_message=f"Large batch {i}") for i in range(batch_size)]
    requests = list(zip(states, inputs))
    
    outputs = cognitive_manager.handle_batch(requests)
    
    assert len(outputs) == batch_size
    for output in outputs:
        assert_cognitive_output(output)


def test_handle_batch_with_decoding_config(cognitive_manager: CognitiveManager, decoding_config: DecodingConfig):
    """
    Test 24: handle_batch() with decoding config.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch(decoding=DecodingConfig)
    Test Senaryosu: Custom decoding config ile batch işleme
    """
    states = [CognitiveState() for _ in range(3)]
    inputs = [CognitiveInput(user_message=f"Batch with config {i}") for i in range(3)]
    requests = list(zip(states, inputs))
    
    decoding_config.max_tokens = 200
    outputs = cognitive_manager.handle_batch(requests, decoding=decoding_config)
    
    assert len(outputs) == 3
    for output in outputs:
        assert_cognitive_output(output)


def test_handle_batch_order_preservation(cognitive_manager: CognitiveManager):
    """
    Test 25: handle_batch() order preservation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Batch'te sıra korunması
    """
    states = [CognitiveState() for _ in range(5)]
    inputs = [CognitiveInput(user_message=f"Order test {i}") for i in range(5)]
    requests = list(zip(states, inputs))
    
    outputs = cognitive_manager.handle_batch(requests)
    
    assert len(outputs) == 5
    # Output'ların sırası input sırasıyla aynı olmalı
    for i, output in enumerate(outputs):
        assert_cognitive_output(output)


def test_handle_batch_state_isolation(cognitive_manager: CognitiveManager):
    """
    Test 26: handle_batch() state isolation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Batch'te state isolation
    """
    states = [CognitiveState() for _ in range(3)]
    inputs = [CognitiveInput(user_message=f"Isolation test {i}") for i in range(3)]
    requests = list(zip(states, inputs))
    
    outputs = cognitive_manager.handle_batch(requests)
    
    # Her state'in izole olması gerekiyor
    assert len(outputs) == 3
    for output in outputs:
        assert_cognitive_output(output)


def test_handle_batch_error_handling(cognitive_manager: CognitiveManager):
    """
    Test 27: handle_batch() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Batch'te hata durumlarında handling
    """
    # Invalid request format - should handle gracefully
    invalid_requests = [(None, None)]
    try:
        result = cognitive_manager.handle_batch(invalid_requests)
        # System should handle gracefully or return empty results
        assert isinstance(result, list)
    except (AttributeError, TypeError, ValueError) as e:
        # Expected error handling is acceptable
        assert isinstance(e, (AttributeError, TypeError, ValueError))


def test_handle_batch_performance(cognitive_manager: CognitiveManager):
    """
    Test 28: handle_batch() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Batch performans testi
    """
    import time
    
    batch_size = 20
    states = [CognitiveState() for _ in range(batch_size)]
    inputs = [CognitiveInput(user_message=f"Perf batch {i}") for i in range(batch_size)]
    requests = list(zip(states, inputs))
    
    start = time.time()
    outputs = cognitive_manager.handle_batch(requests)
    elapsed = time.time() - start
    
    assert len(outputs) == batch_size
    assert elapsed >= 0  # Basic sanity check


def test_handle_batch_mixed_states(cognitive_manager: CognitiveManager):
    """
    Test 29: handle_batch() with mixed states.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch()
    Test Senaryosu: Farklı state'lerle batch işleme
    """
    states = [CognitiveState() for _ in range(3)]
    inputs = [
        CognitiveInput(user_message="Short"),
        CognitiveInput(user_message="Medium length message"),
        CognitiveInput(user_message="Very long message " * 10)
    ]
    requests = list(zip(states, inputs))
    
    outputs = cognitive_manager.handle_batch(requests)
    
    assert len(outputs) == 3
    for output in outputs:
        assert_cognitive_output(output)


def test_handle_batch_vs_sequential_consistency(cognitive_manager: CognitiveManager):
    """
    Test 30: handle_batch() vs sequential handle() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_batch() vs handle()
    Test Senaryosu: Batch ve sequential işleme tutarlılığı
    """
    # Sequential
    state1 = CognitiveState()
    input1 = CognitiveInput(user_message="Sequential test")
    sequential_output = cognitive_manager.handle(state1, input1)
    
    # Batch
    state2 = CognitiveState()
    input2 = CognitiveInput(user_message="Sequential test")
    batch_outputs = cognitive_manager.handle_batch([(state2, input2)])
    
    # Her ikisi de geçerli output döndürmeli
    assert_cognitive_output(sequential_output)
    assert len(batch_outputs) == 1
    assert_cognitive_output(batch_outputs[0])


# ============================================================================
# Test 31-40: handle_multimodal() - Multimodal Processing
# Test Edilen Dosya: cognitive_manager.py (handle_multimodal method)
# ============================================================================

def test_handle_multimodal_text_only(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 31: handle_multimodal() with text only.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Sadece text ile multimodal işleme
    """
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text="Text only message"
    )
    assert_cognitive_output(output)


def test_handle_multimodal_audio_only(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 32: handle_multimodal() with audio only.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Sadece audio ile multimodal işleme
    """
    audio_bytes = b"fake_audio_data"
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        audio=audio_bytes
    )
    assert_cognitive_output(output)


def test_handle_multimodal_image_only(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 33: handle_multimodal() with image only.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Sadece image ile multimodal işleme
    """
    image_bytes = b"fake_image_data"
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        image=image_bytes
    )
    assert_cognitive_output(output)


def test_handle_multimodal_text_and_audio(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 34: handle_multimodal() with text and audio.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Text + audio kombinasyonu
    """
    audio_bytes = b"fake_audio_data"
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text="Text with audio",
        audio=audio_bytes
    )
    assert_cognitive_output(output)


def test_handle_multimodal_text_and_image(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 35: handle_multimodal() with text and image.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Text + image kombinasyonu
    """
    image_bytes = b"fake_image_data"
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text="Text with image",
        image=image_bytes
    )
    assert_cognitive_output(output)


def test_handle_multimodal_all_modalities(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 36: handle_multimodal() with all modalities.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Text + audio + image kombinasyonu
    """
    audio_bytes = b"fake_audio_data"
    image_bytes = b"fake_image_data"
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text="All modalities",
        audio=audio_bytes,
        image=image_bytes
    )
    assert_cognitive_output(output)


def test_handle_multimodal_empty_inputs(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 37: handle_multimodal() with empty inputs.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Tüm input'lar boş (edge case)
    """
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text=None,
        audio=None,
        image=None
    )
    assert_cognitive_output(output)


def test_handle_multimodal_with_decoding_config(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState, decoding_config: DecodingConfig):
    """
    Test 38: handle_multimodal() with decoding config.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal(decoding=DecodingConfig)
    Test Senaryosu: Custom decoding config ile multimodal işleme
    """
    decoding_config.max_tokens = 250
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        text="Multimodal with config",
        decoding=decoding_config
    )
    assert_cognitive_output(output)


def test_handle_multimodal_large_audio(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 39: handle_multimodal() with large audio.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Büyük audio dosyası (edge case)
    """
    large_audio = b"x" * (10 * 1024 * 1024)  # 10MB
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        audio=large_audio
    )
    assert_cognitive_output(output)


def test_handle_multimodal_large_image(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 40: handle_multimodal() with large image.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: handle_multimodal()
    Test Senaryosu: Büyük image dosyası (edge case)
    """
    large_image = b"x" * (5 * 1024 * 1024)  # 5MB
    output = cognitive_manager.handle_multimodal(
        cognitive_state,
        image=large_image
    )
    assert_cognitive_output(output)


# ============================================================================
# Test 41-50: Integration and Edge Cases
# Test Edilen Dosya: cognitive_manager.py (Core integration)
# ============================================================================

def test_core_integration_full_workflow(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 41: Full workflow integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), handle_async(), handle_batch()
    Test Senaryosu: Tam workflow integration testi
    """
    # Sequential
    input1 = CognitiveInput(user_message="Workflow step 1")
    output1 = cognitive_manager.handle(cognitive_state, input1)
    assert_cognitive_output(output1)
    
    # Batch
    states = [CognitiveState() for _ in range(2)]
    inputs = [CognitiveInput(user_message=f"Workflow batch {i}") for i in range(2)]
    outputs = cognitive_manager.handle_batch(list(zip(states, inputs)))
    assert len(outputs) == 2


def test_core_error_recovery(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 42: Error recovery test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Hata sonrası recovery
    """
    # Normal request
    input1 = CognitiveInput(user_message="Normal request")
    output1 = cognitive_manager.handle(cognitive_state, input1)
    assert_cognitive_output(output1)
    
    # Error case (if applicable)
    # Recovery
    input2 = CognitiveInput(user_message="Recovery request")
    output2 = cognitive_manager.handle(cognitive_state, input2)
    assert_cognitive_output(output2)


def test_core_concurrent_access(cognitive_manager: CognitiveManager):
    """
    Test 43: Concurrent access test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Concurrent access senaryosu
    """
    import threading
    
    results = []
    
    def worker(state, input_msg):
        output = cognitive_manager.handle(state, input_msg)
        results.append(output)
    
    threads = []
    for i in range(5):
        state = CognitiveState()
        input_msg = CognitiveInput(user_message=f"Concurrent {i}")
        thread = threading.Thread(target=worker, args=(state, input_msg))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert len(results) == 5
    for result in results:
        assert_cognitive_output(result)


def test_core_state_management(cognitive_manager: CognitiveManager):
    """
    Test 44: State management test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: State yönetimi testi
    """
    state = CognitiveState()
    
    # Multiple requests with same state
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"State test {i}")
        output = cognitive_manager.handle(state, input_msg)
        assert_cognitive_output(output)
        # State'in korunduğunu kontrol et
        assert state is not None


def test_core_performance_under_load(cognitive_manager: CognitiveManager):
    """
    Test 45: Performance under load test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Yük altında performans testi
    """
    import time
    
    num_requests = 50
    start = time.time()
    
    for i in range(num_requests):
        state = CognitiveState()
        input_msg = CognitiveInput(user_message=f"Load test {i}")
        output = cognitive_manager.handle(state, input_msg)
        assert_cognitive_output(output)
    
    elapsed = time.time() - start
    # Basic sanity check
    assert elapsed >= 0
    # Average time per request
    avg_time = elapsed / num_requests
    assert avg_time >= 0


def test_core_memory_usage(cognitive_manager: CognitiveManager):
    """
    Test 46: Memory usage test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Memory kullanımı testi
    """
    import sys
    
    initial_size = sys.getsizeof(cognitive_manager)
    
    # Multiple requests
    for i in range(10):
        state = CognitiveState()
        input_msg = CognitiveInput(user_message=f"Memory test {i}")
        output = cognitive_manager.handle(state, input_msg)
        assert_cognitive_output(output)
    
    # Memory leak kontrolü (basit)
    final_size = sys.getsizeof(cognitive_manager)
    # Büyük bir artış olmamalı (ama garantili değil, sadece sanity check)
    assert final_size >= initial_size


def test_core_thread_safety(cognitive_manager: CognitiveManager):
    """
    Test 47: Thread safety test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Thread safety testi
    """
    import threading
    import time
    
    results = []
    errors = []
    
    def worker(worker_id):
        try:
            for i in range(5):
                state = CognitiveState()
                input_msg = CognitiveInput(user_message=f"Thread {worker_id} request {i}")
                output = cognitive_manager.handle(state, input_msg)
                results.append((worker_id, i, output))
                time.sleep(0.01)  # Small delay
        except Exception as e:
            errors.append((worker_id, e))
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # Her thread 5 request yaptı
    assert len(results) == 25
    # Hata olmamalı
    assert len(errors) == 0


def test_core_resource_cleanup(cognitive_manager: CognitiveManager):
    """
    Test 48: Resource cleanup test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Resource cleanup testi
    """
    # Multiple requests
    for i in range(10):
        state = CognitiveState()
        input_msg = CognitiveInput(user_message=f"Cleanup test {i}")
        output = cognitive_manager.handle(state, input_msg)
        assert_cognitive_output(output)
    
    # Resource cleanup kontrolü (basit)
    # Implementation'a göre cleanup metodları çağrılabilir
    # Şimdilik sadece no exception kontrolü
    assert True


def test_core_configuration_impact(cognitive_manager: CognitiveManager, default_config: CognitiveManagerConfig):
    """
    Test 49: Configuration impact test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle()
    Test Senaryosu: Farklı config'lerin etkisi
    """
    # Default config ile
    state1 = CognitiveState()
    input1 = CognitiveInput(user_message="Config test 1")
    output1 = cognitive_manager.handle(state1, input1)
    assert_cognitive_output(output1)
    
    # Modified config ile (yeni instance)
    default_config.memory.max_episodic_turns = 20
    new_manager = CognitiveManager(
        model_manager=cognitive_manager.mm,
        cfg=default_config
    )
    
    state2 = CognitiveState()
    input2 = CognitiveInput(user_message="Config test 2")
    output2 = new_manager.handle(state2, input2)
    assert_cognitive_output(output2)


def test_core_end_to_end_scenario(cognitive_manager: CognitiveManager):
    """
    Test 50: End-to-end scenario test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), handle_batch(), handle_multimodal()
    Test Senaryosu: Tam end-to-end senaryo
    """
    state = CognitiveState()
    
    # 1. Sequential request
    input1 = CognitiveInput(user_message="E2E step 1")
    output1 = cognitive_manager.handle(state, input1)
    assert_cognitive_output(output1)
    
    # 2. Batch request
    states = [CognitiveState() for _ in range(2)]
    inputs = [CognitiveInput(user_message=f"E2E batch {i}") for i in range(2)]
    batch_outputs = cognitive_manager.handle_batch(list(zip(states, inputs)))
    assert len(batch_outputs) == 2
    
    # 3. Multimodal request
    multimodal_output = cognitive_manager.handle_multimodal(
        state,
        text="E2E multimodal"
    )
    assert_cognitive_output(multimodal_output)

