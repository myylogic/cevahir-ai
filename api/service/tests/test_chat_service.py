import sys
import os
import pytest
import torch

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.service.chat_service import ChatService
from src.neural_network import CevahirNeuralNetwork

@pytest.fixture
def chat_service():
    """
    ChatService sınıfını gerçek bileşenlerle hazırlar.
    """
    return ChatService()

def test_real_model_response(chat_service):
    """
    Modelin gerçek çıktısını test eder.
    """
    test_message = "Seni çok seviyorum Cevahir. Ben baban Muhammed."
    session_id = "real_test_session"

    # Mesaj gönderimi
    response = chat_service.send_message(session_id, test_message)

    # Çıktı kontrolü
    assert response["status"] == "success", "Model yanıt üretiminde hata."
    assert "response" in response, "Model yanıtı alınamadı."

    # Yanıtı ekrana yazdır
    print(f"Gerçek Test Yanıtı: {response['response']}")

def test_model_loading(chat_service):
    """
    Modelin doğru yüklendiğini ve cihaz uyumluluğunu test eder.
    """
    assert isinstance(chat_service.model, CevahirNeuralNetwork), "Model tipi doğru değil."
    assert chat_service.model.device in ["cpu", "cuda"], "Model cihaz uyumluluğu başarısız."

def test_pipeline_integration(chat_service):
    """
    DetokenizationPipeline entegrasyonunu test eder.
    """
    session_id = "pipeline_test_session"
    test_message = "Bu bir entegrasyon test mesajıdır."

    # Mesaj gönderimi
    response = chat_service.send_message(session_id, test_message)

    # Çıktı kontrolü
    assert response["status"] == "success", "Pipeline entegrasyonu başarısız."
    assert isinstance(response["response"], str), "Yanıt string formatında değil."
    print(f"Pipeline Entegrasyon Yanıtı: {response['response']}")

def test_message_too_long(chat_service):
    """
    Maksimum uzunluğu aşan mesajların kesildiğini test eder.
    """
    long_message = "a" * (chat_service.max_input_length + 10)  # Uzun mesaj
    session_id = "test_session"

    response = chat_service.send_message(session_id, long_message)

    # Uzunluk kontrolü
    assert response["status"] == "success", "Uzun mesaj işleme başarısız."
    assert len(response["response"]) <= chat_service.max_input_length, "Yanıt uzunluğu maksimum sınırı aşıyor."

def test_dtype_compatibility(chat_service):
    """
    Modelin giriş türlerinin (dtype) uyumluluğunu test eder.
    """
    test_message = "Cevahir, veri türleriyle ilgili bir test yapıyoruz."
    session_id = "dtype_test_session"

    # Modelden önce giriş türü kontrolü
    tokenized_input = torch.tensor([ord(c) for c in test_message], dtype=torch.long)
    assert tokenized_input.dtype == torch.long, "Giriş verisinin türü yanlış!"

    # Model ile işlem
    response = chat_service.send_message(session_id, test_message)
    assert response["status"] == "success", "Giriş türü uyumsuzluğu modelde hata oluşturdu."

def test_context_processing(chat_service):
    """
    Bağlam işleyişini test eder.
    """
    test_message = "Duygu bağlamı testi."
    session_id = "context_test_session"

    # Mesaj gönderimi
    response = chat_service.send_message(session_id, test_message)

    # Çıktı kontrolü
    assert response["status"] == "success", "Bağlam işlenemedi."
    print(f"Bağlam İşleme Yanıtı: {response['response']}")

if __name__ == "__main__":
    # Testleri manuel olarak çalıştır
    pytest.main(["-v", __file__])
