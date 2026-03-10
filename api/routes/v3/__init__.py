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

__all__ = ['v3_bp']

