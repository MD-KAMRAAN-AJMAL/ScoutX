from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.api import recommendations as recommendations_api
from app.database.repositories import ActivityRepository, RecommendationFeedbackRepository
from app.database.schemas import RecommendationFeedbackCreate
from app.personalization.evaluation import (
    category_diversity,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.personalization.profile import (
    UserProfile,
    _decay,
    build_user_profile,
    cached_user_profile,
    invalidate_profile_cache,
)
from app.personalization.recommendations import RecommendationService
from app.personalization.scoring import personalize_ranked_candidates, score_file
from app.vectorstore.qdrant_client import QdrantStore

USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
FILE_ID = uuid.UUID("00000000-0000-4000-8000-000000000002")
ROOT_ID = uuid.UUID("00000000-0000-4000-8000-000000000003")


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class ProfileSession:
    def __init__(self, events=(), feedback=(), preferences=None):
        self.events = list(events)
        self.feedback = list(feedback)
        self.preferences = preferences
        self.execute_calls = 0

    def get(self, _model, _user_id):
        return self.preferences

    def execute(self, _statement):
        self.execute_calls += 1
        return FakeRows(self.events if self.execute_calls == 1 else self.feedback)

    def scalars(self, _statement):
        return FakeRows([])


def test_profile_derives_recent_file_types_and_feedback_without_query_text():
    now = datetime.now(UTC)
    event = SimpleNamespace(
        file_id=FILE_ID,
        event_type="opened",
        created_at=now - timedelta(days=2),
        details={"query": "must not be returned"},
    )
    feedback = SimpleNamespace(
        file_id=FILE_ID,
        action="useful",
        rating=5,
        created_at=now - timedelta(days=1),
    )
    profile = build_user_profile(
        ProfileSession(events=[(event, ".py", ROOT_ID)], feedback=[(feedback, ".py", ROOT_ID)]),
        USER_ID,
        now=now,
    )
    assert profile.recent_file_ids == (str(FILE_ID),)
    assert profile.preferred_extensions == (".py",)
    assert profile.file_feedback[str(FILE_ID)] > 0.9
    assert profile.extension_feedback[".py"] > 0.9
    assert "query" not in str(profile)


def test_activity_time_decay_has_configured_half_life_and_old_events_fade():
    now = datetime.now(UTC)
    assert _decay(now - timedelta(days=30), now, 30) == pytest.approx(0.5, abs=0.002)
    assert _decay(now - timedelta(days=180), now, 30) < 0.02


def test_profile_cache_reuses_summary_and_supports_explicit_invalidation(monkeypatch):
    import app.personalization.profile as profile_module

    calls = []
    profile = UserProfile(profile_status="cold_start")
    monkeypatch.setattr(
        profile_module, "build_user_profile", lambda *_args, **_kwargs: calls.append(1) or profile
    )
    invalidate_profile_cache(USER_ID)
    first = cached_user_profile(object(), USER_ID)
    second = cached_user_profile(object(), USER_ID)
    assert first is second
    assert len(calls) == 1
    invalidate_profile_cache(USER_ID)
    cached_user_profile(object(), USER_ID)
    assert len(calls) == 2
    invalidate_profile_cache(USER_ID)


def test_personalization_adjustment_is_bounded_and_preserves_base_signal():
    profile = UserProfile(
        preferences={"personalization_enabled": True},
        preferred_extensions=(".py",),
        frequent_root_ids=(str(ROOT_ID),),
        file_access_weights={str(FILE_ID): 1.0},
        file_feedback={str(FILE_ID): 1.0},
    )
    result = score_file(
        base_score=0.7,
        file_id=str(FILE_ID),
        extension=".py",
        root_id=str(ROOT_ID),
        profile=profile,
    )
    assert result.score == pytest.approx(0.78)
    assert result.adjustment == 0.08
    assert len(result.reasons) >= 3


def test_negative_feedback_and_disabled_mode_do_not_replace_retrieval_score():
    profile = UserProfile(
        preferences={"personalization_enabled": True},
        file_feedback={str(FILE_ID): -1.0},
    )
    negative = score_file(
        base_score=0.7, file_id=str(FILE_ID), extension=".txt", root_id=None, profile=profile
    )
    disabled = score_file(
        base_score=0.7,
        file_id=str(FILE_ID),
        extension=".txt",
        root_id=None,
        profile=profile,
        personalization_enabled=False,
    )
    assert negative.score == pytest.approx(0.64)
    assert disabled.score == 0.7
    assert disabled.adjustment == 0


def test_negative_feedback_softly_reduces_similar_file_categories():
    profile = UserProfile(
        preferences={"personalization_enabled": True},
        extension_feedback={".tmp": -1.0},
        root_feedback={str(ROOT_ID): -1.0},
    )
    result = score_file(
        base_score=0.7,
        file_id=str(FILE_ID),
        extension=".tmp",
        root_id=str(ROOT_ID),
        profile=profile,
    )
    assert result.score == pytest.approx(0.67)
    assert len(result.reasons) == 2


def test_preference_exclusions_and_candidate_deduplication():
    profile = UserProfile(
        preferences={
            "personalization_enabled": True,
            "excluded_extensions": [".tmp"],
            "excluded_root_ids": [str(ROOT_ID)],
        }
    )
    assert score_file(
        base_score=0.8,
        file_id=str(FILE_ID),
        extension=".tmp",
        root_id=None,
        profile=profile,
    ).excluded
    candidates = [
        {"final_score": 0.8, "candidate": {"file_id": str(FILE_ID), "chunk_id": "c1"}},
        {"final_score": 0.7, "candidate": {"file_id": str(FILE_ID), "chunk_id": "c1"}},
        {"final_score": 0.6, "candidate": {"file_id": str(FILE_ID), "chunk_id": "c2"}},
    ]
    ranked = personalize_ranked_candidates(candidates, UserProfile(), enabled=False)
    assert len(ranked) == 2


def test_activity_repository_rejects_unrecognized_events():
    with pytest.raises(ValueError, match="Unsupported activity"):
        ActivityRepository(ProfileSession()).record(USER_ID, "raw_query_saved")


def test_latest_duplicate_explicit_feedback_is_updated_not_multiplied():
    existing = SimpleNamespace(action="relevant", rating=5, agent_run_id=None, feedback_text="old")

    class FeedbackSession:
        calls = 0

        def scalar(self, _statement):
            self.calls += 1
            return FILE_ID if self.calls == 1 else existing

        def add(self, _record):
            raise AssertionError("a rapid duplicate should update the existing signal")

        def flush(self):
            pass

    result = RecommendationFeedbackRepository(FeedbackSession()).create(
        USER_ID,
        RecommendationFeedbackCreate(file_id=FILE_ID, rating=1, action="not_relevant"),
    )
    assert result is existing
    assert existing.rating == 1
    assert existing.action == "not_relevant"
    assert existing.feedback_text is None


def test_recommendations_cold_start_returns_safe_recently_indexed_items(tmp_path):
    root_path = tmp_path / "authorized" / "notes"
    root_path.mkdir(parents=True)
    file_path = root_path / "intro.pdf"
    file_path.write_text("indexed fixture", encoding="utf-8")
    file = SimpleNamespace(
        id=FILE_ID,
        user_id=USER_ID,
        root_id=ROOT_ID,
        canonical_path=str(file_path),
        filename="intro.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        modified_at=datetime.now(UTC),
        last_indexed_at=datetime.now(UTC),
        indexing_status="indexed",
    )
    root = SimpleNamespace(
        id=ROOT_ID, user_id=USER_ID, canonical_path=str(root_path), is_enabled=True
    )

    class RecommendationSession(ProfileSession):
        def execute(self, _statement):
            return FakeRows([(file, root)])

    profile = UserProfile(preferences={"personalization_enabled": True})
    service = RecommendationService(
        RecommendationSession(),
        USER_ID,
        profile=profile,
        readable_file=lambda path: Path(path),
        readable_root=lambda path: Path(path),
    )
    items = service.recommend(category="recommended_for_you", limit=5)
    assert len(items) == 1
    assert items[0].file_id == FILE_ID
    assert items[0].path == "intro.pdf"
    assert items[0].category == "recommended_for_you"
    assert "place to start" in items[0].reason
    assert str(tmp_path) not in items[0].path


def test_recommendations_api_serializes_stable_ids(monkeypatch):
    item = SimpleNamespace(
        file_id=FILE_ID,
        filename="intro.pdf",
        path="intro.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        modified_at=None,
        score=0.5,
        category="recently_indexed",
        reason="Recently indexed",
    )

    class FakeRecommendations:
        def __init__(self, *_args):
            pass

        def recommend(self, **_kwargs):
            return [item]

    monkeypatch.setattr(recommendations_api, "RecommendationService", FakeRecommendations)
    response = recommendations_api.get_recommendations(
        limit=12, extension=None, session=object(), user_id=USER_ID
    )
    assert response.items[0].id == FILE_ID
    assert response.items[0].path == "intro.pdf"


def test_user_preferences_default_to_enabled_and_store_explicit_exclusions(monkeypatch):
    from app.api import personalization as personalization_api

    class FakePreferenceRepository:
        saved = None

        def __init__(self, _session):
            pass

        def get(self, _user_id):
            return SimpleNamespace(preferences=self.saved) if self.saved is not None else None

        def upsert(self, _user_id, value):
            self.saved = value

    class FakeRootRepository:
        def __init__(self, _session):
            pass

        def get(self, _user_id, root_id):
            return SimpleNamespace(id=root_id)

    class FakeSession:
        def commit(self):
            pass

    monkeypatch.setattr(personalization_api, "PreferenceRepository", FakePreferenceRepository)
    monkeypatch.setattr(personalization_api, "IndexRootRepository", FakeRootRepository)
    session = FakeSession()
    default = personalization_api.get_preferences(session=session, user_id=USER_ID)
    assert default.personalization_enabled is True

    request = personalization_api.PreferencesUpdate(
        personalization_enabled=False,
        preferred_extensions=["PY", ".py"],
        excluded_extensions=["tmp"],
        preferred_root_ids=[ROOT_ID],
        excluded_root_ids=[],
    )
    saved = personalization_api.update_preferences(request, session=session, user_id=USER_ID)
    assert saved.personalization_enabled is False
    assert saved.preferred_extensions == [".py"]
    assert saved.excluded_extensions == [".tmp"]


def test_profile_reset_records_a_cutoff_without_deleting_activity(monkeypatch):
    from app.api import personalization as personalization_api

    stored = {}
    cleared = []

    class FakePreferenceRepository:
        def __init__(self, _session):
            pass

        def get(self, _user_id):
            return SimpleNamespace(preferences=stored) if stored else None

        def upsert(self, _user_id, value):
            stored.update(value)

    class FakeActivityRepository:
        def __init__(self, _session):
            pass

        def clear_for_user(self, _user_id):
            cleared.append("activity")
            return 1

    class FakeFeedbackRepository:
        def __init__(self, _session):
            pass

        def clear_for_user(self, _user_id):
            cleared.append("feedback")
            return 1

    class FakeSession:
        def commit(self):
            pass

    monkeypatch.setattr(personalization_api, "PreferenceRepository", FakePreferenceRepository)
    monkeypatch.setattr(personalization_api, "ActivityRepository", FakeActivityRepository)
    monkeypatch.setattr(
        personalization_api, "RecommendationFeedbackRepository", FakeFeedbackRepository
    )
    response = personalization_api.reset_personalization(
        personalization_api.ResetRequest(clear_activity_history=False),
        session=FakeSession(),
        user_id=USER_ID,
    )
    assert response.reset and not response.activity_history_cleared
    assert "profile_reset_at" in stored
    assert cleared == []


def test_qdrant_related_search_uses_existing_points_and_user_filter():
    class FakeClient:
        def retrieve(self, **kwargs):
            return [
                SimpleNamespace(id=point_id, payload={"user_id": str(USER_ID), "is_active": True})
                for point_id in kwargs["ids"]
            ]

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(points=[])

    client = FakeClient()
    store = QdrantStore(client=client)
    store._ready = True
    point_id = uuid.uuid4()
    assert store.related_point_search([point_id], user_id=USER_ID, limit=5) == []
    query = client.kwargs["query"]
    assert isinstance(query, models.RecommendQuery)
    assert query.recommend.positive == [str(point_id)]
    matches = {
        condition.key: condition.match.value for condition in client.kwargs["query_filter"].must
    }
    assert matches["user_id"] == str(USER_ID)
    assert matches["is_active"] is True


def test_offline_ranking_metrics_for_controlled_fixture():
    relevant = {"a", "c"}
    baseline = ["b", "a", "c"]
    personalized = ["a", "c", "b"]
    assert precision_at_k(personalized, relevant, 2) == 1.0
    assert recall_at_k(personalized, relevant, 2) == 1.0
    assert mean_reciprocal_rank(personalized, relevant) == 1.0
    assert ndcg_at_k(personalized, {"a": 2.0, "c": 1.0}, 3) >= ndcg_at_k(
        baseline, {"a": 2.0, "c": 1.0}, 3
    )
    assert category_diversity(["code", "docs", "docs"], 3) == pytest.approx(2 / 3)
