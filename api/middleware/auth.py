# -*- coding: utf-8 -*-
"""
Authentication Middleware
========================

JWT-based authentication.
Endüstri Standardı: JWT token-based authentication

Identity comes only from a verified ``Authorization: Bearer <JWT>`` header.
The ``X-User-ID`` header is ignored unless the configuration explicitly trusts
it (``TRUST_X_USER_ID``, set from ``CEVAHIR_TRUST_X_USER_ID=1``) in a
development or testing configuration; every such use is logged as a warning.
This module verifies tokens; it does not issue them (there is no login or
token endpoint in this API).
"""

from functools import wraps
from flask import request, g, current_app
from api.api_config import TRUSTED_USER_HEADER_ENVS
from api.utils.exceptions import AuthenticationError
from api.utils import error_response
from api.security.jwt import verify_token, get_user_from_token
import logging

logger = logging.getLogger(__name__)

TRUSTED_USER_HEADER = "X-User-ID"


def trusts_user_id_header(config) -> bool:
    """
    Whether the X-User-ID header may act as the caller's identity.

    Requires both the explicit opt-in (``TRUST_X_USER_ID is True``) and a
    development or testing configuration (``API_ENV``). DEBUG alone is not enough.
    """
    return config.get("TRUST_X_USER_ID") is True and config.get("API_ENV") in TRUSTED_USER_HEADER_ENVS


def get_current_user() -> str:
    """
    Get current user ID from JWT token or header.
    
    Priority:
    1. JWT token (Authorization header); an invalid or expired token is rejected
       even when other identity headers are present
    2. X-User-ID header, only when trusts_user_id_header() allows it
    
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
            scheme, _, credentials = auth_header.strip().partition(' ')
            token = credentials.strip() if scheme.lower() == 'bearer' else auth_header.strip()
            
            # Verify token
            user_id = get_user_from_token(token)
            if user_id:
                logger.debug(f"User authenticated via JWT: {user_id}")
                return user_id
        except Exception as e:
            logger.warning("JWT token verification failed")
        raise AuthenticationError("Invalid or expired authentication token")
    
    user_id = request.headers.get(TRUSTED_USER_HEADER)
    if user_id is not None:
        if not trusts_user_id_header(current_app.config):
            logger.info("Ignored X-User-ID header: identity is accepted only from a verified JWT")
        elif user_id.strip():
            logger.warning(
                "Identity %r taken from the unauthenticated X-User-ID header "
                "(CEVAHIR_TRUST_X_USER_ID, %s configuration); local development only",
                user_id, current_app.config.get("API_ENV"),
            )
            return user_id
    
    # No authentication found
    raise AuthenticationError(
        "Authentication required. Please provide an Authorization header with a Bearer token."
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

