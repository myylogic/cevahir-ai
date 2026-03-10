# -*- coding: utf-8 -*-
"""
Structured Logging
==================

Structured logging utilities for better observability.
Endüstri Standardı: Structured logging (JSON format)
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import request, g, has_request_context


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logs.
    
    Endüstri Standardı: JSON logging for log aggregation
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
        
        Returns:
            JSON string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request context if available
        if has_request_context():
            log_data["request_id"] = getattr(g, 'request_id', None)
            log_data["method"] = request.method
            log_data["path"] = request.path
            log_data["remote_addr"] = request.remote_addr
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_structured_logging(
    logger_name: str = None,
    level: str = "INFO",
    use_json: bool = False
) -> logging.Logger:
    """
    Setup structured logging.
    
    Args:
        logger_name: Logger name
        level: Log level
        use_json: Use JSON format (for production)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler()
    
    # Set formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **kwargs
):
    """
    Log HTTP request in structured format.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        **kwargs: Additional fields
    """
    log_data = {
        "type": "http_request",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }
    
    if has_request_context():
        log_data["request_id"] = getattr(g, 'request_id', None)
        log_data["remote_addr"] = request.remote_addr
        log_data["user_agent"] = request.headers.get('User-Agent')
    
    log_data.update(kwargs)
    
    if status_code >= 500:
        logger.error("HTTP Request", extra=log_data)
    elif status_code >= 400:
        logger.warning("HTTP Request", extra=log_data)
    else:
        logger.info("HTTP Request", extra=log_data)

