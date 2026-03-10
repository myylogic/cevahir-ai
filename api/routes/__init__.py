from flask import Blueprint
from routes.json_routes import json_routes
from routes.vocab_routes import vocab_routes
from routes.ml_routes import ml_routes
from routes.test_routes import test_routes
from routes.orchestrator_routes import orchestrator_routes
from routes.training_routes import training_routes
from routes.model_routes import model_routes

# Diğer blueprint'ler isteğe bağlı olarak eklenebilir
try:
    from routes.gameplay_routes import gameplay_routes
    GAMEPLAY_ENABLED = True
except ImportError:
    gameplay_routes = None
    GAMEPLAY_ENABLED = False


def register_blueprints(app):
    """
    Uygulamaya blueprint'leri ekler.
    """
    app.register_blueprint(json_routes, url_prefix='/json')
    app.register_blueprint(vocab_routes, url_prefix='/vocab')
    app.register_blueprint(ml_routes, url_prefix='/ml')
    app.register_blueprint(test_routes, url_prefix='/test')
    app.register_blueprint(orchestrator_routes, url_prefix='/orchestrator')
    app.register_blueprint(training_routes, url_prefix='/train')
    app.register_blueprint(model_routes, url_prefix='/model')  # Model routes kaydedildi

    if GAMEPLAY_ENABLED and gameplay_routes:
        app.register_blueprint(gameplay_routes, url_prefix='/gameplay')

    # Loglama
    print("Blueprint'ler başarıyla kaydedildi:")
    for rule in app.url_map.iter_rules():
        print(f"Route: {rule.endpoint} - {rule}")
