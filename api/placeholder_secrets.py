# -*- coding: utf-8 -*-
"""
Placeholder Secrets
===================

Example secret values that ship with this repository. They are never accepted
as real secrets: api.api_config refuses them in production, and
api.security.jwt refuses to sign or verify tokens with them.

This module has no imports and reads no environment variables, so importing
api.security never loads or validates the API configuration.
"""

PLACEHOLDER_SECRETS = frozenset({
    "change-me",
    "change-me-in-production",
    "dev-secret-key-change-in-production",
    "test-secret-key",
})
