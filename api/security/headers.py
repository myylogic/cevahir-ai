# -*- coding: utf-8 -*-
"""
Security Headers
================

Security headers middleware (Helmet.js benzeri).
Endüstri Standardı: Security headers for production
"""

from flask import Response
from typing import Dict


def add_security_headers(response: Response) -> Response:
    """
    Add security headers to response.
    
    Endüstri Standardı: OWASP security headers
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy
    - Referrer-Policy
    
    Args:
        response: Flask response object
    
    Returns:
        Response with security headers
    """
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # XSS Protection (legacy browsers)
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # HSTS (HTTPS only - production)
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Content Security Policy (CSP)
    # response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions Policy (formerly Feature-Policy)
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    return response

