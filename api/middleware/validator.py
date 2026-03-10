# -*- coding: utf-8 -*-
"""
Request Validator
================

Request validation utilities.
"""

from typing import Dict, Any, Optional
from api.utils.exceptions import ValidationError


def validate_request(
    data: Dict[str, Any],
    required_fields: list,
    optional_fields: Optional[list] = None
) -> Dict[str, Any]:
    """
    Validate request data.
    
    Args:
        data: Request data dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names
    
    Returns:
        Validated data dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Request data must be a JSON object")
    
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )
    
    # Validate field types and values
    validated = {}
    for field in required_fields:
        value = data.get(field)
        if value is None:
            continue
        validated[field] = value
    
    # Add optional fields if present
    if optional_fields:
        for field in optional_fields:
            if field in data:
                validated[field] = data[field]
    
    return validated

