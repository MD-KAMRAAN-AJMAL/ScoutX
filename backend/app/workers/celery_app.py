"""Celery configuration using Redis for local task brokering and results."""

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_init

from app.config import get_settings
from app.security.permissions import ensure_not_root

settings = get_settings()

celery_app = Celery(
    "scoutx",
    broker=str(settings.redis_url),
    backend=str(settings.redis_url),
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_default_queue="filesystem",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_transport_options={"visibility_timeout": 3600},
    beat_schedule={
        "reconcile-enabled-roots-every-six-hours": {
            "task": "scoutx.reconcile_enabled_roots",
            "schedule": crontab(minute=15, hour="*/6"),
        }
    },
)


@worker_init.connect
def reject_root_worker(**_: object) -> None:
    """Prevent a Celery worker from running with privileges that bypass file ACLs."""
    ensure_not_root()

