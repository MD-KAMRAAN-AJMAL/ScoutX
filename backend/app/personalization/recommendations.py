"""Permission-aware recommendations built from current files and stored vectors."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File, FileActivity, IndexRoot
from app.database.repositories import DocumentChunkRepository
from app.personalization.profile import UserProfile, _aware, cached_user_profile
from app.personalization.scoring import score_file
from app.security.permissions import is_within, require_readable_directory, require_readable_file
from app.vectorstore.qdrant_client import QdrantStore

logger = logging.getLogger(__name__)
RecommendationCategory = Literal[
    "recommended_for_you",
    "recently_accessed",
    "continue_working",
    "frequently_used",
    "related",
    "recently_indexed",
]


@dataclass(frozen=True)
class Recommendation:
    file_id: uuid.UUID
    filename: str
    path: str
    extension: str | None
    mime_type: str | None
    modified_at: datetime | None
    score: float
    category: RecommendationCategory
    reason: str


class RecommendationService:
    """Generate bounded, explainable recommendations from local ScoutX data."""

    def __init__(
        self,
        session: Session,
        user_id: uuid.UUID,
        *,
        profile: UserProfile | None = None,
        vector_store: QdrantStore | None = None,
        readable_file=require_readable_file,
        readable_root=require_readable_directory,
        now: datetime | None = None,
    ) -> None:
        self.session = session
        self.user_id = user_id
        self.now = now or datetime.now(UTC)
        self.profile = profile or cached_user_profile(session, user_id, now=self.now)
        self.vector_store = vector_store
        self.readable_file = readable_file
        self.readable_root = readable_root
        self._accessible: dict[uuid.UUID, bool] = {}

    def recommend(
        self,
        *,
        category: RecommendationCategory = "recommended_for_you",
        limit: int = 12,
        extension: str | None = None,
    ) -> list[Recommendation]:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        if not self.profile.enabled and category != "recently_indexed":
            if category == "recommended_for_you":
                return self._recently_indexed(
                    self._available_files(extension), limit, cold_start=True
                )
            return []
        files = self._available_files(extension)
        if category == "recently_accessed":
            return self._recently_accessed(files, limit)
        if category == "frequently_used":
            return self._frequently_used(files, limit)
        if category == "recently_indexed":
            return self._recently_indexed(files, limit)
        if category == "continue_working":
            return self._continue_working(files, limit)
        related = self._related(files, min(50, limit * 3))
        if category == "related":
            return related[:limit]
        if related:
            return related[:limit]
        continued = self._continue_working(files, limit)
        if continued:
            return continued
        return self._recently_indexed(files, limit, cold_start=True)

    def _available_files(self, extension: str | None) -> dict[uuid.UUID, tuple[File, IndexRoot]]:
        statement = (
            select(File, IndexRoot)
            .join(IndexRoot, (IndexRoot.id == File.root_id) & (IndexRoot.user_id == File.user_id))
            .where(
                File.user_id == self.user_id,
                File.indexing_status == "indexed",
                IndexRoot.is_enabled.is_(True),
            )
            .order_by(File.last_indexed_at.desc().nullslast(), File.modified_at.desc().nullslast())
            .limit(1000)
        )
        if extension:
            normalized = extension.strip().lower()
            normalized = normalized if normalized.startswith(".") else f".{normalized}"
            statement = statement.where(File.extension == normalized)
        rows = self.session.execute(statement).all()
        excluded_roots = {
            str(value) for value in self.profile.preferences.get("excluded_root_ids", [])
        }
        excluded_extensions = {
            str(value).lower() for value in self.profile.preferences.get("excluded_extensions", [])
        }
        result = {}
        for file, root in rows:
            if (
                str(root.id) in excluded_roots
                or (file.extension or "").lower() in excluded_extensions
            ):
                continue
            if self._can_access(file, root):
                result[file.id] = (file, root)
        return result

    def _can_access(self, file: File, root: IndexRoot) -> bool:
        if file.id in self._accessible:
            return self._accessible[file.id]
        try:
            root_path = Path(self.readable_root(root.canonical_path)).resolve(strict=True)
            file_path = Path(self.readable_file(file.canonical_path)).resolve(strict=True)
            accessible = is_within(file_path, root_path)
        except OSError:
            accessible = False
        self._accessible[file.id] = accessible
        return accessible

    @staticmethod
    def _display_path(file: File, root: IndexRoot) -> str:
        try:
            return Path(file.canonical_path).relative_to(Path(root.canonical_path)).as_posix()
        except ValueError:
            return file.filename

    def _item(
        self,
        file: File,
        root: IndexRoot,
        *,
        category: RecommendationCategory,
        base_score: float,
        reason: str,
    ) -> Recommendation | None:
        scored = score_file(
            base_score=base_score,
            file_id=str(file.id),
            extension=file.extension,
            root_id=str(file.root_id),
            profile=self.profile,
            personalization_enabled=self.profile.enabled,
        )
        if scored.excluded:
            return None
        reasons = [reason, *scored.reasons]
        return Recommendation(
            file_id=file.id,
            filename=file.filename,
            path=self._display_path(file, root),
            extension=file.extension,
            mime_type=file.mime_type,
            modified_at=file.modified_at,
            score=scored.score,
            category=category,
            reason=" · ".join(dict.fromkeys(reasons)),
        )

    def _activity(self, limit: int = 1000):
        return self.session.execute(
            select(FileActivity.file_id, FileActivity.event_type, FileActivity.created_at)
            .where(
                FileActivity.user_id == self.user_id,
                FileActivity.file_id.is_not(None),
                FileActivity.created_at >= self.now - timedelta(days=180),
                FileActivity.event_type.in_(
                    ("opened", "selected", "file_opened", "recommendation_clicked")
                ),
            )
            .order_by(FileActivity.created_at.desc())
            .limit(limit)
        ).all()

    def _recently_accessed(self, files, limit: int) -> list[Recommendation]:
        items = []
        seen = set()
        from app.personalization.profile import _decay

        for file_id, _event_type, created_at in self._activity():
            if file_id in seen or file_id not in files:
                continue
            seen.add(file_id)
            weight = _decay(created_at, self.now, 30.0)
            item = self._item(
                *files[file_id],
                category="recently_accessed",
                base_score=0.50 + 0.20 * weight,
                reason="Recently opened",
            )
            if item:
                items.append(item)
            if len(items) >= limit:
                break
        return items

    def _frequently_used(self, files, limit: int) -> list[Recommendation]:
        from app.personalization.profile import _decay

        weights: defaultdict[uuid.UUID, float] = defaultdict(float)
        for file_id, _event_type, created_at in self._activity():
            if file_id in files:
                weights[file_id] += _decay(created_at, self.now, 60.0)
        ranked = sorted(weights.items(), key=lambda pair: (-pair[1], str(pair[0])))
        return [
            item
            for file_id, weight in ranked[:limit]
            if (
                item := self._item(
                    *files[file_id],
                    category="frequently_used",
                    base_score=min(0.75, 0.45 + 0.05 * weight),
                    reason="You access this file often",
                )
            )
        ]

    def _recently_indexed(
        self, files, limit: int, cold_start: bool = False
    ) -> list[Recommendation]:
        result = []
        for file, root in files.values():
            if not cold_start:
                if file.last_indexed_at is None:
                    continue
                indexed_at = _aware(file.last_indexed_at, self.now)
                if (self.now - indexed_at).total_seconds() > 30 * 86400:
                    continue
            item = self._item(
                file,
                root,
                category="recommended_for_you" if cold_start else "recently_indexed",
                base_score=0.50,
                reason=(
                    "A place to start in your library; recommendations improve as you use ScoutX"
                    if cold_start
                    else "Recently indexed"
                ),
            )
            if item:
                result.append(item)
            if len(result) >= limit:
                break
        return result

    def _continue_working(self, files, limit: int) -> list[Recommendation]:
        latest_root = next(
            (
                files[uuid.UUID(file_id)][1].id
                for file_id in self.profile.recent_file_ids
                if _uuid_file_in(file_id, files)
            ),
            None,
        )
        if latest_root is None:
            return []
        recent_ids = {uuid.UUID(file_id) for file_id in self.profile.recent_file_ids[:10]}
        candidates = sorted(
            (
                (file, root)
                for file, root in files.values()
                if root.id == latest_root and file.id not in recent_ids
            ),
            key=lambda pair: pair[0].modified_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        result = []
        for file, root in candidates:
            item = self._item(
                file,
                root,
                category="continue_working",
                base_score=0.55,
                reason="More files from a folder you recently used",
            )
            if item:
                result.append(item)
            if len(result) >= limit:
                break
        return result

    def _related(self, files, limit: int) -> list[Recommendation]:
        positive_sources = [
            uuid.UUID(file_id)
            for file_id, value in self.profile.file_feedback.items()
            if value > 0 and _uuid_file_in(file_id, files)
        ]
        negative_sources = [
            uuid.UUID(file_id)
            for file_id, value in self.profile.file_feedback.items()
            if value < 0 and _uuid_file_in(file_id, files)
        ]
        source_file_ids = list(
            dict.fromkeys(
                positive_sources
                + [
                    uuid.UUID(file_id)
                    for file_id in self.profile.recent_file_ids
                    if _uuid_file_in(file_id, files)
                ]
            )
        )[:5]
        if not source_file_ids:
            return []
        file_ids_for_vectors = list(dict.fromkeys(source_file_ids + negative_sources))
        point_ids_by_file: defaultdict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
        for file_id, point_id in self.session.execute(
            select(File.id, DocumentChunk.qdrant_point_id)
            .join(File, File.id == DocumentChunk.file_id)
            .where(
                File.user_id == self.user_id,
                File.id.in_(file_ids_for_vectors),
                File.indexing_status == "indexed",
                DocumentChunk.content_version == File.version,
                DocumentChunk.qdrant_point_id.is_not(None),
            )
            .order_by(DocumentChunk.chunk_index)
            .limit(30)
        ).all():
            if point_id is not None:
                point_ids_by_file[file_id].append(point_id)
        point_ids = [point for file_id in source_file_ids for point in point_ids_by_file[file_id]][
            :20
        ]
        negative_point_ids = [
            point for file_id in negative_sources for point in point_ids_by_file[file_id]
        ][:10]
        if not point_ids:
            return []
        try:
            store = self.vector_store or QdrantStore()
            hits = store.related_point_search(
                list(point_ids),
                user_id=self.user_id,
                limit=limit * 4,
                negative_point_ids=negative_point_ids,
            )
        except (OSError, RuntimeError, ValueError):
            logger.info("Related-file vector recommendations are unavailable")
            return []
        rows = DocumentChunkRepository(self.session).get_active_by_point_ids(
            self.user_id, [hit.point_id for hit in hits]
        )
        scores = {hit.point_id: hit.score for hit in hits}
        by_file: dict[uuid.UUID, float] = {}
        for source in rows:
            file_id = source.file.id
            if file_id in source_file_ids or file_id not in files:
                continue
            score = scores.get(source.chunk.qdrant_point_id, 0.0)
            by_file[file_id] = max(by_file.get(file_id, 0.0), score)
        result = []
        for file_id, similarity in sorted(by_file.items(), key=lambda pair: pair[1], reverse=True):
            file, root = files[file_id]
            item = self._item(
                file,
                root,
                category="related",
                base_score=min(0.9, max(0.25, similarity)),
                reason="Similar to a file you recently opened or marked useful",
            )
            if item:
                result.append(item)
            if len(result) >= limit:
                break
        return result


def _uuid_file_in(file_id: str, files) -> bool:
    try:
        return uuid.UUID(file_id) in files
    except ValueError:
        return False
