# -*- coding: utf-8 -*-
"""
Health Checks
=============

Detailed health check system.
Endüstri Standardı: Comprehensive health checks

The detailed endpoint is unauthenticated, so results carry only fixed, generic
messages; exception details are written to the server log, never to responses.

Checks are registered per application. get_health_status() runs the checks of
the given or current application and raises when there are none to run, rather
than reporting an empty, healthy result.
"""

from flask import Flask, current_app, has_app_context
from typing import Dict, Any, List, Optional
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
                    "message": "Health check failed (details in the server log)"
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


def register_health_checks(app: Flask):
    """
    Register health checks for Flask app.
    
    Each app gets its own registry (app.extensions["cevahir_health"]), so one
    app's status is never reported by another app in the same process.

    Args:
        app: Flask application instance
    """
    health_check = HealthCheck()
    app.extensions["cevahir_health"] = health_check

    # Database health check
    def check_database():
        try:
            from database import UnitOfWork
            with UnitOfWork() as uow:
                # Simple query test
                pass
            return {"status": "ok", "message": "Database connection healthy"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=True)
            return {"status": "error", "message": "Database unavailable"}
    
    health_check.register_check("database", check_database)
    
    # Cevahir health check
    def check_cevahir():
        try:
            if hasattr(app, 'cevahir') and app.cevahir:
                return {"status": "ok", "message": "Cevahir initialized"}
            return {"status": "error", "message": "Cevahir not initialized"}
        except Exception as e:
            logger.error(f"Cevahir health check failed: {e}", exc_info=True)
            return {"status": "error", "message": "Cevahir check failed"}
    
    health_check.register_check("cevahir", check_cevahir)
    
    # ChattingManager health check
    def check_chatting_manager():
        try:
            if hasattr(app, 'chatting_manager') and app.chatting_manager:
                return {"status": "ok", "message": "ChattingManager initialized"}
            return {"status": "error", "message": "ChattingManager not initialized"}
        except Exception as e:
            logger.error(f"ChattingManager health check failed: {e}", exc_info=True)
            return {"status": "error", "message": "ChattingManager check failed"}
    
    health_check.register_check("chatting_manager", check_chatting_manager)
    
    logger.info("Health checks registered")


def get_health_status(app: Optional[Flask] = None) -> Dict[str, Any]:
    """
    Run the health checks registered for an application.

    Args:
        app: Application to check; defaults to the current application.

    Returns:
        Health status dictionary

    Raises:
        RuntimeError: No application is given or active, or register_health_checks()
            was not called for it. There is no process-wide registry to fall back on.
    """
    if app is None:
        if not has_app_context():
            raise RuntimeError(
                "get_health_status() needs an application: pass app= or call it in an application context"
            )
        app = current_app
    health_check = app.extensions.get("cevahir_health")
    if health_check is None:
        raise RuntimeError(
            "No health checks are registered for this application; call register_health_checks(app) first"
        )
    return health_check.run_checks()

