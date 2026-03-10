# -*- coding: utf-8 -*-
"""
Security Middleware
===================

Security headers and CSRF protection middleware.
"""

from flask import Flask, request, g
from api.security.headers import add_security_headers
import logging

logger = logging.getLogger(__name__)


def register_security_middleware(app: Flask):
    """
    Register security middleware.
    
    Args:
        app: Flask application instance
    """
    # Security headers
    @app.after_request
    def security_headers(response):
        """Add security headers to all responses"""
        if app.config.get("ENABLE_SECURITY_HEADERS", True):
            return add_security_headers(response)
        return response
    
    # Request ID (for tracing)
    @app.before_request
    def add_request_id():
        """Add request ID for tracing"""
        if app.config.get("ENABLE_REQUEST_ID", True):
            request_id_header = app.config.get("REQUEST_ID_HEADER", "X-Request-ID")
            request_id = request.headers.get(request_id_header) or request.headers.get("X-Request-ID")
            if not request_id:
                import uuid
                request_id = str(uuid.uuid4())
            g.request_id = request_id
    
    # CORS preflight handling
    @app.before_request
    def handle_preflight():
        """Handle CORS preflight requests"""
        if request.method == "OPTIONS":
            response = app.make_default_options_response()
            headers = response.headers
            headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
            headers['Access-Control-Allow-Methods'] = ', '.join(app.config.get('CORS_METHODS', ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']))
            headers['Access-Control-Allow-Headers'] = ', '.join(app.config.get('CORS_HEADERS', ['Content-Type', 'Authorization']))
            headers['Access-Control-Max-Age'] = '3600'
            return response
    
    logger.info("Security middleware registered")

