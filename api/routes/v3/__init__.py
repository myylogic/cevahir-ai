# -*- coding: utf-8 -*-
"""
API v3 Routes
=============

Modern RESTful API endpoints.
"""

from flask import Blueprint

# Create v3 blueprint
v3_bp = Blueprint('v3', __name__, url_prefix='/api/v3')

# Import routes (will be registered in app.py)
from api.routes.v3 import chat, sessions, users, health

def create_v3_blueprint(chat_service, session_service, user_service):
    blueprint = Blueprint('v3', __name__, url_prefix='/api/v3')
    chat.init_chat_routes(chat_service, blueprint=blueprint)
    sessions.init_session_routes(session_service, blueprint=blueprint)
    users.init_user_routes(user_service, blueprint=blueprint)
    blueprint.add_url_rule('/health', view_func=health.health_check, methods=['GET'])
    blueprint.add_url_rule('/health/detailed', view_func=health.detailed_health_check, methods=['GET'])
    return blueprint


__all__ = ['v3_bp', 'create_v3_blueprint']

