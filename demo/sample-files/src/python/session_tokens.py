"""Synthetic authentication helper for the ScoutX demo corpus."""

from __future__ import annotations

import hashlib
import hmac


def sign_session_token(payload: bytes, signing_key: bytes) -> str:
    """Return a hexadecimal HMAC-SHA256 signature for a session payload."""
    return hmac.new(signing_key, payload, hashlib.sha256).hexdigest()


def verify_session_token(payload: bytes, signature: str, signing_key: bytes) -> bool:
    """Compare a supplied signature without a timing-sensitive string check."""
    expected = sign_session_token(payload, signing_key)
    return hmac.compare_digest(expected, signature)
