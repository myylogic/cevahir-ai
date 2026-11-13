# -*- coding: utf-8 -*-
"""
Security Middleware
===================

Security headers, request IDs and the CORS policy.

CORS, including preflight (OPTIONS) requests, is answered by Flask-CORS with
the options from cors_options(); this module adds no CORS headers of its own.
Only origins listed in CORS_ORIGINS are allowed: literal entries must match the
Origin exactly, regular-expression entries must match all of it. A configured
'*' is sent literally instead of echoing the caller's Origin, and credentials
are never allowed (the API authenticates with bearer tokens, not cookies).
"""

import re
from flask import Flask, request, g
from api.api_config import DEFAULT_CORS_HEADERS, DEFAULT_CORS_METHODS, PLACEHOLDER_SECRETS
from api.middleware.auth import trusts_user_id_header
from api.security.headers import add_security_headers
import logging

logger = logging.getLogger(__name__)

# Characters that make Flask-CORS treat an origin string as a regular expression.
_REGEX_HINTS = frozenset("*\\]?$^[]()")


def _as_list(value):
    """Config lists may also arrive as one comma-separated string."""
    if isinstance(value, str):
        value = value.split(",")
    items = [item.strip() if isinstance(item, str) else item for item in value or []]
    return [item for item in items if item]


def _origin_pattern(origin):
    """Keep literal origins and '*'; compile regular expressions to match the whole Origin."""
    if not isinstance(origin, str) or origin == "*" or _REGEX_HINTS.isdisjoint(origin):
        return origin
    return re.compile(rf"(?:{origin})\Z", re.IGNORECASE)


def cors_options(config) -> dict:
    """
    Flask-CORS resource options for an application config.

    - only configured origins are answered; a non-matching Origin gets no CORS headers
    - '*' is sent literally (send_wildcard), never as a reflected Origin
    - credentials are never allowed, so they cannot be combined with '*'
    - requests without an Origin header get no CORS headers
    """
    origins = [_origin_pattern(origin) for origin in _as_list(config.get("CORS_ORIGINS"))]
    return {
        "origins": origins,
        "methods": _as_list(config.get("CORS_METHODS")) or list(DEFAULT_CORS_METHODS),
        "allow_headers": _as_list(config.get("CORS_HEADERS")) or list(DEFAULT_CORS_HEADERS),
        "expose_headers": ["X-Request-ID"],
        "max_age": 3600,
        "supports_credentials": False,
        "send_wildcard": "*" in origins,
        "always_send": False,
        "vary_header": True,
    }


def warn_about_risky_settings(config):
    """Log the security-relevant settings an operator should not miss."""
    if trusts_user_id_header(config):
        logger.warning(
            "X-User-ID is accepted as identity (CEVAHIR_TRUST_X_USER_ID=1, %s configuration): "
            "any caller can act as any user. Use this only on a local machine.",
            config.get("API_ENV"),
        )
    if "*" in _as_list(config.get("CORS_ORIGINS")):
        logger.warning("CORS_ORIGINS contains '*': any website may call this API from a browser")
    jwt_secret = config.get("JWT_SECRET_KEY") or config.get("SECRET_KEY")
    if not jwt_secret or jwt_secret in PLACEHOLDER_SECRETS:
        logger.warning(
            "JWT_SECRET_KEY is not configured: authenticated endpoints answer 401 until it is set"
        )


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
    
    warn_about_risky_settings(app.config)
    logger.info("Security middleware registered")

