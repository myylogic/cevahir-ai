# -*- coding: utf-8 -*-
"""
Request ID Middleware
=====================

Request ID tracking for distributed tracing.
Endüstri Standardı: Request ID for tracing and debugging
"""

import uuid
from flask import Flask, request, g, has_request_context
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_request_id() -> Optional[str]:
    """
    Get current request ID.
    
    Returns:
        Request ID string or None
    """
    if has_request_context():
        return getattr(g, 'request_id', None)
    return None


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for current request.
    
    Args:
        request_id: Optional request ID (generates new if None)
    
    Returns:
        Request ID string
    """
    if not has_request_context():
        return None
    
    if request_id is None:
        # Try to get from header
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            # Generate new UUID
            request_id = str(uuid.uuid4())
    
    g.request_id = request_id
    return request_id


def register_request_id_middleware(app: Flask):
    """
    Register request ID middleware.
    
    Args:
        app: Flask application instance
    """
    @app.before_request
    def add_request_id():
        """Add request ID to request context"""
        if app.config.get("ENABLE_REQUEST_ID", True):
            request_id = request.headers.get('X-Request-ID')
            if not request_id:
                request_id = str(uuid.uuid4())
            g.request_id = request_id
    
    @app.after_request
    def add_request_id_header(response):
        """Add request ID to response headers"""
        if app.config.get("ENABLE_REQUEST_ID", True) and hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        return response
    
    logger.info("Request ID middleware registered")

