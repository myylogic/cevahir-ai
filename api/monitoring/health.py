# -*- coding: utf-8 -*-
"""
Health Checks
=============

Detailed health check system.
Endüstri Standardı: Comprehensive health checks
"""

from flask import Flask, current_app
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthCheck:
    """Health check manager"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_func: callable):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Check function (returns dict with 'status' and optional 'message')
        """
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Health status dictionary
        """
        results = {}
        overall_status = "healthy"
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result
                
                if result.get("status") != "ok":
                    overall_status = "degraded"
                    
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
                results[name] = {
                    "status": "error",
                    "message": str(e)
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global health check instance
_health_check = HealthCheck()


def register_health_checks(app: Flask):
    """
    Register health checks for Flask app.
    
    Args:
        app: Flask application instance
    """
    # Database health check
    def check_database():
        try:
            from database import UnitOfWork
            with UnitOfWork() as uow:
                # Simple query test
                pass
            return {"status": "ok", "message": "Database connection healthy"}
        except Exception as e:
            return {"status": "error", "message": f"Database error: {str(e)}"}
    
    _health_check.register_check("database", check_database)
    
    # Cevahir health check
    def check_cevahir():
        try:
            if hasattr(app, 'cevahir') and app.cevahir:
                return {"status": "ok", "message": "Cevahir initialized"}
            return {"status": "error", "message": "Cevahir not initialized"}
        except Exception as e:
            return {"status": "error", "message": f"Cevahir error: {str(e)}"}
    
    _health_check.register_check("cevahir", check_cevahir)
    
    # ChattingManager health check
    def check_chatting_manager():
        try:
            if hasattr(app, 'chatting_manager') and app.chatting_manager:
                return {"status": "ok", "message": "ChattingManager initialized"}
            return {"status": "error", "message": "ChattingManager not initialized"}
        except Exception as e:
            return {"status": "error", "message": f"ChattingManager error: {str(e)}"}
    
    _health_check.register_check("chatting_manager", check_chatting_manager)
    
    logger.info("Health checks registered")


def get_health_status() -> Dict[str, Any]:
    """
    Get current health status.
    
    Returns:
        Health status dictionary
    """
    return _health_check.run_checks()

