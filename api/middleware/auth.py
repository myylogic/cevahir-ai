# -*- coding: utf-8 -*-
"""
Authentication Middleware
========================

JWT-based authentication.
Endüstri Standardı: JWT token-based authentication
"""

from functools import wraps
from flask import request, g, current_app
from api.utils.exceptions import AuthenticationError
from api.utils import error_response
from api.security.jwt import verify_token, get_user_from_token
import logging

logger = logging.getLogger(__name__)


def get_current_user() -> str:
    """
    Get current user ID from JWT token or header.
    
    Priority:
    1. JWT token (Authorization header)
    2. X-User-ID header (fallback for development)
    
    Returns:
        User ID string
    
    Raises:
        AuthenticationError: If user ID not found
    """
    # Try JWT token first
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            # Extract token (Bearer <token>)
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header
            
            # Verify token
            user_id = get_user_from_token(token)
            if user_id:
                logger.debug(f"User authenticated via JWT: {user_id}")
                return user_id
        except Exception as e:
            logger.warning(f"JWT token verification failed: {e}")
    
    # Fallback: X-User-ID header (development only)
    user_id = request.headers.get('X-User-ID')
    if user_id:
        # Check if we're in development mode
        if current_app.config.get('DEBUG', False):
            logger.warning("Using X-User-ID header (development mode only)")
            return user_id
        else:
            raise AuthenticationError("JWT token required in production. X-User-ID header is not allowed.")
    
    # No authentication found
    raise AuthenticationError(
        "Authentication required. Please provide Authorization header with Bearer token or X-User-ID header (development only)."
    )


def require_auth(f):
    """
    Decorator to require authentication for an endpoint.
    
    Usage:
        @require_auth
        def my_endpoint():
            user_id = g.current_user_id
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            user_id = get_current_user()
            g.current_user_id = user_id
            return f(*args, **kwargs)
        except AuthenticationError as e:
            return error_response(
                error_code=e.error_code,
                message=e.message,
                status_code=e.status_code
            )
    
    return decorated_function

