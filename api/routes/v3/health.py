# -*- coding: utf-8 -*-
"""
Health Check API Routes (v3)
============================

Health check endpoints.
Endüstri Standardı: Comprehensive health checks
"""

from flask import current_app
from api.routes.v3 import v3_bp
from api.utils import success_response
from api.monitoring.health import get_health_status


@v3_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Response:
        {
            "success": true,
            "data": {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    """
    from datetime import datetime
    
    return success_response(
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        message="API is healthy"
    )


@v3_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Checks all system components:
        - Database connection
        - Cevahir status
        - ChattingManager status
    
    Response:
        {
            "success": true,
            "data": {
                "status": "healthy",
                "checks": {
                    "database": {"status": "ok", "message": "..."},
                    "cevahir": {"status": "ok", "message": "..."},
                    "chatting_manager": {"status": "ok", "message": "..."}
                }
            }
        }
    """
    health_status = get_health_status()
    
    return success_response(
        data=health_status,
        message="Detailed health check completed"
    )

