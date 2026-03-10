# -*- coding: utf-8 -*-
"""
User API Routes (v3)
=====================

RESTful user endpoints.
"""

from flask import request, g
from api.routes.v3 import v3_bp
from api.middleware.auth import require_auth
from api.utils import success_response, error_response
from api.services.user_service import UserService
import logging

logger = logging.getLogger(__name__)


def init_user_routes(user_service: UserService):
    """
    Initialize user routes with service.
    
    Args:
        user_service: UserService instance
    """
    
    @v3_bp.route('/users/me', methods=['GET'])
    @require_auth
    def get_current_user():
        """
        Get current user endpoint.
        
        Response:
            {
                "success": true,
                "data": {
                    "user_id": "user-uuid",
                    "email": "user@example.com",
                    "name": "User Name",
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        """
        try:
            user_id = g.current_user_id
            
            # Get user
            user = user_service.get_user(user_id)
            
            return success_response(
                data=user,
                message="User retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in get_current_user: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to get user",
                status_code=500
            )
    
    @v3_bp.route('/users/me/preferences', methods=['GET'])
    @require_auth
    def get_user_preferences():
        """
        Get user preferences endpoint.
        
        Response:
            {
                "success": true,
                "data": {
                    "preferences": {...}
                }
            }
        """
        try:
            user_id = g.current_user_id
            
            # Get user (preferences are in metadata)
            user = user_service.get_user(user_id)
            preferences = user.get('metadata', {})
            
            return success_response(
                data={"preferences": preferences},
                message="Preferences retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in get_user_preferences: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to get preferences",
                status_code=500
            )
    
    @v3_bp.route('/users/me/preferences', methods=['PATCH'])
    @require_auth
    def update_user_preferences():
        """
        Update user preferences endpoint.
        
        Request:
            {
                "preferences": {...}
            }
        
        Response:
            {
                "success": true,
                "data": {
                    "user_id": "user-uuid",
                    "preferences": {...}
                }
            }
        """
        try:
            user_id = g.current_user_id
            data = request.get_json() or {}
            
            preferences = data.get('preferences', {})
            if not isinstance(preferences, dict):
                return error_response(
                    error_code="VALIDATION_ERROR",
                    message="preferences must be a dictionary",
                    status_code=400
                )
            
            # Update preferences
            result = user_service.update_user_preferences(
                user_id=user_id,
                preferences=preferences
            )
            
            return success_response(
                data=result,
                message="Preferences updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in update_user_preferences: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to update preferences",
                status_code=500
            )

