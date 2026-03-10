# -*- coding: utf-8 -*-
"""
Password Hashing
================

Password hashing and verification using bcrypt.
Endüstri Standardı: bcrypt for password hashing
"""

import bcrypt
import logging

logger = logging.getLogger(__name__)

# Try to import bcrypt, fallback to simple hashing if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available, using simple hashing (NOT SECURE FOR PRODUCTION)")


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password string
    """
    if BCRYPT_AVAILABLE:
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        # Fallback: simple hash (NOT SECURE - only for development)
        import hashlib
        logger.warning("Using simple hash (NOT SECURE) - install bcrypt for production")
        return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        password: Plain text password
        hashed: Hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    else:
        # Fallback: simple hash comparison
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed

