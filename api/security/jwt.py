# -*- coding: utf-8 -*-
"""
JWT Authentication
=================

JWT token creation and verification.
Endüstri Standardı: JWT-based authentication
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from flask import current_app
import logging

logger = logging.getLogger(__name__)


def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        user_id: User ID
        expires_delta: Optional expiration time delta
        additional_claims: Additional JWT claims
    
    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(
            minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        )
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": user_id,  # Subject (user ID)
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    if additional_claims:
        payload.update(additional_claims)
    
    secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY", "change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    
    logger.debug(f"Access token created for user: {user_id}")
    return token


def create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token.
    
    Args:
        user_id: User ID
    
    Returns:
        JWT refresh token string
    """
    expires_delta = timedelta(
        days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
    )
    
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + expires_delta,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY", "change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    
    logger.debug(f"Refresh token created for user: {user_id}")
    return token


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verify JWT token.
    
    Args:
        token: JWT token string
        token_type: Token type ("access" or "refresh")
    
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY", "change-me")
        algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        
        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
            return None
        
        # Verify expiration
        if datetime.utcnow() > datetime.fromtimestamp(payload.get("exp", 0)):
            logger.warning("Token expired")
            return None
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}", exc_info=True)
        return None


def get_user_from_token(token: str) -> Optional[str]:
    """
    Extract user ID from JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        User ID or None if invalid
    """
    payload = verify_token(token, token_type="access")
    if payload:
        return payload.get("sub")  # Subject (user ID)
    return None

