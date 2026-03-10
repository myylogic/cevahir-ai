# -*- coding: utf-8 -*-
"""
Error Handler Middleware
========================

Centralized error handling for Flask app.
"""

from flask import Flask
from api.utils import error_response
from api.utils.exceptions import APIError
from chatting_management.exceptions import (
    UserNotFoundError,
    SessionNotFoundError,
    SessionAccessDeniedError,
    InvalidMessageError,
    CevahirIntegrationError,
)


def register_error_handlers(app: Flask):
    """
    Register error handlers for Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(APIError)
    def handle_api_error(e: APIError):
        """Handle custom API errors"""
        return error_response(
            error_code=e.error_code,
            message=e.message,
            status_code=e.status_code
        )
    
    @app.errorhandler(UserNotFoundError)
    def handle_user_not_found(e: UserNotFoundError):
        """Handle user not found errors"""
        return error_response(
            error_code="USER_NOT_FOUND",
            message=str(e),
            status_code=404
        )
    
    @app.errorhandler(SessionNotFoundError)
    def handle_session_not_found(e: SessionNotFoundError):
        """Handle session not found errors"""
        return error_response(
            error_code="SESSION_NOT_FOUND",
            message=str(e),
            status_code=404
        )
    
    @app.errorhandler(SessionAccessDeniedError)
    def handle_session_access_denied(e: SessionAccessDeniedError):
        """Handle session access denied errors"""
        return error_response(
            error_code="SESSION_ACCESS_DENIED",
            message=str(e),
            status_code=403
        )
    
    @app.errorhandler(InvalidMessageError)
    def handle_invalid_message(e: InvalidMessageError):
        """Handle invalid message errors"""
        return error_response(
            error_code="INVALID_MESSAGE",
            message=str(e),
            status_code=400
        )
    
    @app.errorhandler(CevahirIntegrationError)
    def handle_cevahir_error(e: CevahirIntegrationError):
        """Handle Cevahir integration errors"""
        return error_response(
            error_code="MODEL_ERROR",
            message=f"Model processing error: {str(e)}",
            status_code=500
        )
    
    @app.errorhandler(404)
    def handle_not_found(e):
        """Handle 404 errors"""
        return error_response(
            error_code="NOT_FOUND",
            message="Endpoint not found",
            status_code=404
        )
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        """Handle 500 errors"""
        app.logger.error(f"Internal server error: {e}", exc_info=True)
        return error_response(
            error_code="INTERNAL_ERROR",
            message="Internal server error",
            status_code=500
        )

