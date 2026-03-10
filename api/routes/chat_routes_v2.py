# -*- coding: utf-8 -*-
"""
Chat Routes V2
==============

Flask routes for ChattingManagement module.
Endüstri Standartları: RESTful API, error handling, logging
"""

from flask import Blueprint, request, jsonify
import logging
from typing import Dict, Any

# ChattingManagement import
from chatting_management import ChattingManager, ChattingConfig
from chatting_management.exceptions import (
    UserNotFoundError,
    SessionNotFoundError,
    SessionAccessDeniedError,
    InvalidMessageError,
    CevahirIntegrationError,
)

logger = logging.getLogger(__name__)

# Blueprint oluştur
chat_v2_bp = Blueprint('chat_v2', __name__)

# ChattingManager instance (Flask app factory'de initialize edilecek)
chatting_manager: Optional[ChattingManager] = None


def init_chatting_manager(cevahir_instance, config: Optional[ChattingConfig] = None):
    """
    Initialize ChattingManager (Flask app factory'de çağrılacak).
    
    Args:
        cevahir_instance: Cevahir instance
        config: ChattingConfig (optional, default config kullanılır)
    """
    global chatting_manager
    if config is None:
        config = ChattingConfig()
    chatting_manager = ChattingManager(config=config, cevahir=cevahir_instance)
    logger.info("ChattingManager initialized for Flask routes")


@chat_v2_bp.route('/api/v2/chat/send', methods=['POST'])
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
            "status": "success",
            "response": "Assistant response",
            "session_id": "session-uuid",
            "message_id": "message-uuid"
        }
    """
    try:
        if chatting_manager is None:
            return jsonify({
                "status": "error",
                "message": "ChattingManager not initialized"
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "JSON data required"
            }), 400
        
        # Get user_id from JWT (TODO: implement JWT authentication)
        user_id = request.headers.get('X-User-ID') or data.get('user_id')
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "User ID required"
            }), 401
        
        session_id = data.get('session_id')
        message = data.get('message')
        
        if not session_id or not message:
            return jsonify({
                "status": "error",
                "message": "session_id and message are required"
            }), 400
        
        # Send message
        result = chatting_manager.send_message(
            user_id=user_id,
            session_id=session_id,
            message=message
        )
        
        return jsonify({
            "status": "success",
            **result
        }), 200
        
    except (SessionNotFoundError, SessionAccessDeniedError) as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except InvalidMessageError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except CevahirIntegrationError as e:
        return jsonify({
            "status": "error",
            "message": f"Model processing error: {str(e)}"
        }), 500
    except Exception as e:
        logger.error(f"Error in send_message endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@chat_v2_bp.route('/api/v2/chat/history', methods=['GET'])
def get_history():
    """
    Get conversation history endpoint.
    
    Query params:
        - session_id: Session ID
        - limit: Maximum number of messages (optional)
    
    Response:
        {
            "status": "success",
            "messages": [...]
        }
    """
    try:
        if chatting_manager is None:
            return jsonify({
                "status": "error",
                "message": "ChattingManager not initialized"
            }), 500
        
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "User ID required"
            }), 401
        
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required"
            }), 400
        
        limit = request.args.get('limit', type=int)
        
        messages = chatting_manager.get_conversation_history(
            session_id=session_id,
            user_id=user_id,
            limit=limit
        )
        
        return jsonify({
            "status": "success",
            "messages": messages
        }), 200
        
    except (SessionNotFoundError, SessionAccessDeniedError) as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error in get_history endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@chat_v2_bp.route('/api/v2/sessions', methods=['POST'])
def create_session():
    """
    Create new session endpoint.
    
    Request:
        {
            "title": "Session title" (optional)
        }
    
    Response:
        {
            "status": "success",
            "session_id": "session-uuid",
            "title": "Session title"
        }
    """
    try:
        if chatting_manager is None:
            return jsonify({
                "status": "error",
                "message": "ChattingManager not initialized"
            }), 500
        
        user_id = request.headers.get('X-User-ID') or request.json.get('user_id')
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "User ID required"
            }), 401
        
        data = request.get_json() or {}
        title = data.get('title')
        
        session = chatting_manager.create_session(
            user_id=user_id,
            title=title
        )
        
        return jsonify({
            "status": "success",
            **session
        }), 201
        
    except UserNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error in create_session endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@chat_v2_bp.route('/api/v2/sessions', methods=['GET'])
def list_sessions():
    """
    List user sessions endpoint.
    
    Query params:
        - limit: Maximum number of sessions (optional)
    
    Response:
        {
            "status": "success",
            "sessions": [...]
        }
    """
    try:
        if chatting_manager is None:
            return jsonify({
                "status": "error",
                "message": "ChattingManager not initialized"
            }), 500
        
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "User ID required"
            }), 401
        
        limit = request.args.get('limit', type=int)
        
        sessions = chatting_manager.list_sessions(
            user_id=user_id,
            limit=limit
        )
        
        return jsonify({
            "status": "success",
            "sessions": sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Error in list_sessions endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

