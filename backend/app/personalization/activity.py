"""Best-effort event recording that stores interaction metadata, never content."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.database.engine import SessionLocal
from app.database.repositories import ActivityRepository

logger = logging.getLogger(__name__)


def record_activity_background(
    user_id: uuid.UUID,
    event_type: str,
    *,
    file_id: uuid.UUID | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist non-critical activity after the HTTP response has been sent."""
    try:
        with SessionLocal() as session:
            ActivityRepository(session).record(
                user_id, event_type, file_id=file_id, details=details
            )
            session.commit()
    except (OSError, SQLAlchemyError, ValueError):
        # Tracking is optional and must never change the success of user actions.
        logger.info("Best-effort activity write was unavailable")
