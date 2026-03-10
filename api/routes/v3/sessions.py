# -*- coding: utf-8 -*-
"""
Session API Routes (v3)
========================

RESTful session endpoints.
"""

from flask import request, g
from api.routes.v3 import v3_bp
from api.middleware.auth import require_auth
from api.middleware.validator import validate_request
from api.utils import success_response, error_response
from api.services.session_service import SessionService
from api.utils.exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)


def init_session_routes(session_service: SessionService):
    """
    Initialize session routes with service.
    
    Args:
        session_service: SessionService instance
    """
    
    @v3_bp.route('/sessions', methods=['POST'])
    @require_auth
    def create_session():
        """
        Create new session endpoint.
        
        Request:
            {
                "title": "Session title" (optional)
            }
        
        Response:
            {
                "success": true,
                "data": {
                    "session_id": "session-uuid",
                    "user_id": "user-uuid",
                    "title": "Session title",
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        """
        try:
            user_id = g.current_user_id
            data = request.get_json() or {}
            
            # Validate request (title is optional)
            validated = validate_request(
                data,
                required_fields=[],
                optional_fields=['title']
            )
            
            title = validated.get('title')
            
            # Create session
            session = session_service.create_session(
                user_id=user_id,
                title=title
            )
            
            return success_response(
                data=session,
                message="Session created successfully",
                status_code=201
            )
            
        except ValidationError as e:
            return error_response(
                error_code="VALIDATION_ERROR",
                message=str(e),
                status_code=400
            )
        except Exception as e:
            logger.error(f"Error in create_session: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to create session",
                status_code=500
            )
    
    @v3_bp.route('/sessions', methods=['GET'])
    @require_auth
    def list_sessions():
        """
        List user sessions endpoint.
        
        Query params:
            - limit: Maximum number of sessions (optional)
        
        Response:
            {
                "success": true,
                "data": {
                    "sessions": [...]
                }
            }
        """
        try:
            user_id = g.current_user_id
            limit = request.args.get('limit', type=int)
            
            # List sessions
            sessions = session_service.list_sessions(
                user_id=user_id,
                limit=limit
            )
            
            return success_response(
                data={"sessions": sessions},
                message="Sessions retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in list_sessions: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to list sessions",
                status_code=500
            )

