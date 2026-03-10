# -*- coding: utf-8 -*-
"""
Chat API Routes (v3)
====================

RESTful chat endpoints.
"""

from flask import request, g
from api.routes.v3 import v3_bp
from api.middleware.auth import require_auth
from api.middleware.validator import validate_request
from api.utils import success_response, error_response
from api.services.chat_service import ChatService
from api.utils.exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)


def init_chat_routes(chat_service: ChatService):
    """
    Initialize chat routes with service.
    
    Args:
        chat_service: ChatService instance
    """
    
    @v3_bp.route('/chat/messages', methods=['POST'])
    @require_auth
    def send_message():
        """
        Send message endpoint.
        
        Request:
            {
                "session_id": "session-uuid",
                "message": "User message text"
            }
        
        Response:
            {
                "success": true,
                "data": {
                    "response": "Assistant response",
                    "session_id": "session-uuid",
                    "message_id": "message-uuid",
                    "metadata": {...}
                }
            }
        """
        try:
            user_id = g.current_user_id
            data = request.get_json() or {}
            
            # Validate request
            validated = validate_request(
                data,
                required_fields=['session_id', 'message']
            )
            
            session_id = validated['session_id']
            message = validated['message']
            
            # Send message
            result = chat_service.send_message(
                user_id=user_id,
                session_id=session_id,
                message=message
            )
            
            return success_response(
                data=result,
                message="Message sent successfully"
            )
            
        except ValidationError as e:
            return error_response(
                error_code="VALIDATION_ERROR",
                message=str(e),
                status_code=400,
                details=getattr(e, 'details', {})
            )
        except Exception as e:
            logger.error(f"Error in send_message: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to send message",
                status_code=500
            )
    
    @v3_bp.route('/chat/messages', methods=['GET'])
    @require_auth
    def get_messages():
        """
        Get conversation history endpoint.
        
        Query params:
            - session_id: Session ID (required)
            - limit: Maximum number of messages (optional)
        
        Response:
            {
                "success": true,
                "data": {
                    "messages": [...]
                }
            }
        """
        try:
            user_id = g.current_user_id
            session_id = request.args.get('session_id')
            limit = request.args.get('limit', type=int)
            
            if not session_id:
                return error_response(
                    error_code="VALIDATION_ERROR",
                    message="session_id is required",
                    status_code=400
                )
            
            # Get conversation history
            messages = chat_service.get_conversation_history(
                user_id=user_id,
                session_id=session_id,
                limit=limit
            )
            
            return success_response(
                data={"messages": messages},
                message="Conversation history retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in get_messages: {e}", exc_info=True)
            return error_response(
                error_code="INTERNAL_ERROR",
                message="Failed to get conversation history",
                status_code=500
            )

