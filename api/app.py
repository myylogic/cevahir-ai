import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import traceback
import socket
import time
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
import logging
from logging.handlers import RotatingFileHandler

# Config dosyasından parametreleri yükle
from config.parameters import DEVICE, LOGGING_PATH
from api.api_config import get_config

# YENİ: App factory import (tüm entegrasyon burada)
try:
    from api.app_factory import create_app as create_app_v3
    USE_NEW_FACTORY = True
except ImportError:
    USE_NEW_FACTORY = False
    # Eski blueprint kayıtlarını routes modülünden al
    from routes import register_blueprints


def is_port_in_use(port):
    """
    Belirtilen portun kullanımda olup olmadığını kontrol eder.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def create_app():
    """
    Flask uygulamasını oluştur ve yapılandırır.
    
    Endüstri Standardı: App Factory Pattern
    
    YENİ: Eğer app_factory.py mevcutsa, yeni entegre yapıyı kullanır.
    Eski yapı fallback olarak kullanılabilir.
    """
    # YENİ: Entegre app factory kullan (Database + Cevahir + ChattingManagement)
    if USE_NEW_FACTORY:
        logging.info("🚀 Using new integrated app factory (Database + Cevahir + ChattingManagement)")
        logging.info("   Industry-standard Flask API with JWT, Security, Monitoring, etc.")
        return create_app_v3()
    
    # ESKİ: Eski yapı (fallback - deprecated)
    logging.warning("⚠️ Using legacy app factory (old structure)")
    logging.warning("   Consider migrating to app_factory.py for industry-standard features")
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # Yapılandırmayı yükle
    config = get_config()
    app.config.from_object(config)

    # CORS desteği ekle
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Rate Limiting yapılandırması
    limiter = Limiter(
        get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.getenv("LIMITER_STORAGE_URI", "memory://")  # Redis URI kullanılabilir
    )
    limiter.init_app(app)

    # Blueprint'leri kaydet
    register_blueprints(app)

    # Loglama ayarlarını yapılandır
    setup_logging()

    # Hata yönetimini yapılandır
    setup_error_handlers(app)

    # Ana sayfa endpoint
    @app.route('/')
    def home():
        """
        Ana sayfayı render eder.
        """
        return render_template('index.html')

    # API rotalarını listeleyen endpoint
    @app.route('/api/routes')
    def list_routes():
        """
        Mevcut API rotalarını JSON formatında döner.
        """
        return jsonify({
            "message": "Cevahir API'ye hoş geldiniz.",
            "routes": [str(rule) for rule in app.url_map.iter_rules()]
        })

    return app


def setup_logging():
    """
    Loglama yapılandırması.
    """
    if not os.path.exists(LOGGING_PATH):
        os.makedirs(LOGGING_PATH)

    # Genel loglama
    log_file = os.path.join(LOGGING_PATH, 'app_process.log')
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10 MB dosya boyutu, 5 yedek
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    # Hata loglama
    error_handler = RotatingFileHandler(os.path.join(LOGGING_PATH, 'errors.log'), maxBytes=5 * 1024 * 1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logging.getLogger().addHandler(error_handler)

    # Stream loglama (Terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info("Loglama yapılandırması tamamlandı.")


def setup_error_handlers(app):
    """
    Flask uygulaması için gelişmiş hata yönetimi yapılandırması.
    """
    from datetime import datetime
    import traceback

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """
        Tüm HTTP hatalarını yakalar ve düzenler.
        """
        # Hata bilgilerini detaylı bir şekilde logla
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "HTTPException",
            "code": e.code,
            "name": e.name,
            "description": e.description,
            "method": request.method,
            "url": request.url,
            "remote_addr": request.remote_addr,
        }
        app.logger.error(f"HTTPException: {log_data}")

        # Kullanıcıya dönecek JSON yanıt
        response = jsonify({
            "code": e.code,
            "name": e.name,
            "description": e.description,
            "path": request.path,
            "method": request.method,
            "remote_addr": request.remote_addr,
        })
        response.status_code = e.code
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        """
        Tüm genel hataları yakalar ve loglar.
        """
        # Hata ID'si oluştur
        error_id = os.urandom(8).hex()

        # Hata bilgilerini detaylı bir şekilde logla
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "UnhandledException",
            "error_id": error_id,
            "exception_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "method": request.method,
            "url": request.url,
            "remote_addr": request.remote_addr,
        }
        app.logger.critical(f"UnhandledException: {log_data}")

        # Kullanıcıya dönecek JSON yanıt
        return jsonify({
            "error": "Beklenmeyen bir hata oluştu.",
            "details": "Lütfen destek ekibiyle iletişime geçin.",
            "error_id": error_id,
            "method": request.method,
            "url": request.url,
        }), 500


if __name__ == '__main__':
    try:
        # Varsayılan portu ve timeout değerlerini belirle
        PORT = int(os.getenv("PORT", 5000))
        TIMEOUT = int(os.getenv("TIMEOUT", 172800))  # Varsayılan timeout 48 saat

        # Kullanımda olmayan bir port bulunana kadar devam et
        initial_port = PORT
        while is_port_in_use(PORT):
            logging.warning(f"Port {PORT} kullanımda. {PORT + 1} portuna geçiliyor.")
            PORT += 1
            if PORT - initial_port > 50:  # Maksimum 50 port denemesi
                raise RuntimeError("50 farklı port denendi, kullanılabilir port bulunamadı!")

        # Flask uygulamasını oluştur ve yapılandır
        app = create_app()
        logging.info(f"Flask uygulaması {PORT} portunda başlatılıyor...")

        # Flask uygulamasını çalıştır
        app.run(host='0.0.0.0', port=PORT)

    except KeyboardInterrupt:
        logging.info("Uygulama manuel olarak durduruldu.")
        print("Uygulama kapatıldı.")
    except RuntimeError as e:
        logging.critical(f"Kritik bir hata oluştu: {str(e)}")
        print(f"Kritik hata: {e}")
    except Exception as e:
        logging.error(f"Flask uygulaması başlatılamadı: {str(e)}")
        traceback.print_exc()
        print(f"Beklenmeyen bir hata oluştu: {e}")
    finally:
        logging.info("Uygulama kapatma işlemleri tamamlandı.")
        print("Uygulama sonlandırıldı.")
