# -*- coding: utf-8 -*-
"""
Standard API Response Format
============================

Endüstri Standardı: Consistent response format across all endpoints
"""

from flask import jsonify
from datetime import datetime
from typing import Any, Dict, Optional


def success_response(
    data: Any = None,
    message: str = "Operation successful",
    status_code: int = 200,
    **kwargs
) -> tuple:
    """
    Standard success response format.
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
        **kwargs: Additional fields
    
    Returns:
        (response, status_code) tuple
    """
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    if data is not None:
        response["data"] = data
    
    # Add any additional fields
    response.update(kwargs)
    
    return jsonify(response), status_code


def error_response(
    error_code: str,
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple:
    """
    Standard error response format.
    
    Args:
        error_code: Error code (e.g., "VALIDATION_ERROR")
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
        **kwargs: Additional fields
    
    Returns:
        (response, status_code) tuple
    """
    response = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    if details:
        response["error"]["details"] = details
    
    # Add any additional fields
    response.update(kwargs)
    
    return jsonify(response), status_code

