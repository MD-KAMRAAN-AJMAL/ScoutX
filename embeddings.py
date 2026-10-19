"""Bounded embedding provider abstraction and stdlib implementations for ScoutX."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from scoutx.models import FileDocument
from scoutx.text import SEMANTIC_EXPANSIONS, tokenize

ALLOWED_EMBEDDING_PROVIDERS: frozenset[str] = frozenset(
    {"local_hash", "ollama", "openai_compatible", "gemini"}
)

DEFAULT_EMBEDDING_BASE_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434/v1",
    "openai_compatible": "http://127.0.0.1:8000/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}

DEFAULT_EMBEDDING_API_KEY_ENVS: dict[str, str | None] = {
    "local_hash": None,
    "ollama": None,
    "openai_compatible": "SCOUTX_OPENAI_API_KEY",
    "gemini": "SCOUTX_GEMINI_API_KEY",
}

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NORM_EPSILON = 1e-12


class EmbeddingError(RuntimeError):
    """Base exception for all embedding provider and vector validation failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """Raised when an embedding endpoint is unreachable, misconfigured, or returns HTTP errors."""


class EmbeddingTimeoutError(EmbeddingUnavailableError):
    """Raised when an embedding request exceeds its configured timeout."""


class EmbeddingResponseFormatError(EmbeddingError):
    """Raised when an embedding response is malformed, non-finite, or zero-norm."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions or model identities do not match expectations."""


def _redact_secret(text: str, secret: str | None) -> str:
    if not text:
        return ""
    redacted = text
    if secret and len(secret) >= 4:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for a ScoutX embedding provider.

    ``max_input_chars`` is a strict per-input-text character limit applied to every
    individual string before it is embedded or serialized into a provider request.
    """

    provider: str = "local_hash"
    model: str = "scoutx-feature-hash-v1"
    dimensions: int | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 8.0
    max_input_chars: int = 4000
    batch_size: int = 16
    normalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in ALLOWED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"provider must be one of {sorted(ALLOWED_EMBEDDING_PROVIDERS)}, got {self.provider!r}"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        object.__setattr__(self, "model", self.model.strip())

        if self.provider == "local_hash" and self.dimensions is None:
            object.__setattr__(self, "dimensions", 64)

        if self.dimensions is not None:
            if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
                raise ValueError(f"dimensions must be an integer, got {self.dimensions!r}")
            if self.provider == "local_hash":
                if not (8 <= self.dimensions <= 4096):
                    raise ValueError(
                        f"local_hash dimensions must be in [8, 4096], got {self.dimensions}"
                    )
            elif self.dimensions < 1:
                raise ValueError(f"dimensions must be >= 1, got {self.dimensions}")

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError(
                f"timeout_seconds must be a finite number > 0, got {self.timeout_seconds!r}"
            )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

        if (
            isinstance(self.max_input_chars, bool)
            or not isinstance(self.max_input_chars, int)
            or self.max_input_chars < 32
        ):
            raise ValueError(
                f"max_input_chars must be an integer >= 32, got {self.max_input_chars!r}"
            )

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError(f"batch_size must be an integer >= 1, got {self.batch_size!r}")

        if not isinstance(self.normalize, bool):
            raise ValueError(f"normalize must be a bool, got {self.normalize!r}")

        if self.base_url is None and self.provider in DEFAULT_EMBEDDING_BASE_URLS:
            object.__setattr__(self, "base_url", DEFAULT_EMBEDDING_BASE_URLS[self.provider])
        elif self.base_url is not None:
            if not isinstance(self.base_url, str) or not self.base_url.strip():
                raise ValueError("base_url must be a non-empty URL string when provided")
            parsed = urllib.parse.urlparse(self.base_url.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"base_url must be a valid http(s) URL, got {self.base_url!r}")
            object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/"))

        if self.api_key_env is None:
            object.__setattr__(self, "api_key_env", DEFAULT_EMBEDDING_API_KEY_ENVS[self.provider])
        else:
            if not isinstance(self.api_key_env, str) or not _ENV_VAR_NAME_RE.match(
                self.api_key_env
            ):
                raise ValueError(
                    "api_key_env must be a valid environment variable name (e.g. 'SCOUTX_OPENAI_API_KEY'), never a raw secret"
                )

    def resolve_api_key(self, *, required: bool = False) -> str | None:
        if not self.api_key_env:
            if required:
                raise EmbeddingUnavailableError(
                    f"Provider '{self.provider}' requires an API key environment variable, but none is configured."
                )
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            if required:
                raise EmbeddingUnavailableError(
                    f"Required API key environment variable '{self.api_key_env}' is not set or is empty."
                )
            return None
        return value


def bound_embedding_input(text: str, max_input_chars: int) -> str:
    """Validate and truncate an individual embedding input string to max_input_chars."""
    if not isinstance(text, str):
        raise EmbeddingResponseFormatError(
            f"Embedding input must be a str, got {type(text).__name__}"
        )
    if (
        isinstance(max_input_chars, bool)
        or not isinstance(max_input_chars, int)
        or max_input_chars < 1
    ):
        raise ValueError(f"max_input_chars must be a positive integer, got {max_input_chars!r}")
    bounded = text[:max_input_chars].strip()
    if not bounded:
        raise EmbeddingResponseFormatError("Embedding input text must not be empty")
    return bounded


def build_document_embedding_text(document: FileDocument, max_input_chars: int = 4000) -> str:
    """Build a bounded, privacy-safe embedding text for a FileDocument.

    Never includes host filesystem paths or parent directory paths.
    Always returns a string of length <= max_input_chars.
    """
    if (
        isinstance(max_input_chars, bool)
        or not isinstance(max_input_chars, int)
        or max_input_chars < 32
    ):
        raise ValueError(f"max_input_chars must be an integer >= 32, got {max_input_chars!r}")

    title = (document.title or "").strip() or "untitled"
    ext = (document.extension or "").strip()
    header = f"{title} ({ext})" if ext else title
    body = (document.content or "").strip()
    combined = f"{header}\n{body}" if body else header
    return combined[:max_input_chars]


def validate_and_normalize_vector(
    vector: Any,
    *,
    expected_dimensions: int | None = None,
    normalize: bool = True,
) -> tuple[float, ...]:
    """Validate that a vector contains finite non-zero floats of expected dimensionality."""
    if isinstance(vector, (str, bytes, bytearray, Mapping)) or not isinstance(vector, Sequence):
        raise EmbeddingResponseFormatError(
            f"Embedding vector must be a sequence of numbers, got {type(vector).__name__}"
        )
    length = len(vector)
    if length == 0:
        raise EmbeddingResponseFormatError("Embedding vector must not be empty")
    if expected_dimensions is not None and length != expected_dimensions:
        raise EmbeddingDimensionMismatchError(
            f"Expected embedding vector of dimension {expected_dimensions}, got {length}"
        )

    coords: list[float] = []
    sq_sum = 0.0
    for idx, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingResponseFormatError(
                f"Vector coordinate at index {idx} must be a finite number, got {value!r}"
            )
        fval = float(value)
        if not math.isfinite(fval):
            raise EmbeddingResponseFormatError(
                f"Vector coordinate at index {idx} is not finite: {value!r}"
            )
        coords.append(fval)
        sq_sum += fval * fval

    norm = math.sqrt(sq_sum)
    if not math.isfinite(norm) or norm <= _NORM_EPSILON:
        raise EmbeddingResponseFormatError("Embedding vector has zero or non-finite L2 norm")

    if not normalize:
        return tuple(coords)
    return tuple(coord / norm for coord in coords)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense/feature vector providers in ScoutX."""

    @property
    def model_id(self) -> str:
        """Canonical model identifier '<provider>:<model>' (including dimension tag when applicable)."""
        ...

    @property
    def dimensions(self) -> int | None:
        """Expected vector dimensionality if fixed by configuration or provider."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed a sequence of non-empty texts into validated vectors of uniform dimension."""
        ...

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a single query text into a validated vector."""
        ...


class LocalHashEmbeddingProvider:
    """Deterministic offline feature-hashing embedding provider.

    IMPORTANT: This provider computes a signed count-sketch projection over:
      1. Primary alphanumeric tokens (weight 1.0),
      2. Single-hop curated domain synonym expansions from SEMANTIC_EXPANSIONS (weight 0.55),
      3. Character 3-grams and 4-grams within tokens of length >= 4 (weight 0.25).

    It is NOT a trained neural language model:
      - It cannot capture arbitrary open-domain semantic paraphrases absent from
        lexical/morphological overlap or SEMANTIC_EXPANSIONS.
      - It is a bag-of-features projection that ignores word order and negation.
      - Finite bucket counts introduce bounded hash-collision noise O(1/sqrt(dimensions)).
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig(provider="local_hash")
        if self.config.provider != "local_hash":
            raise ValueError(
                f"LocalHashEmbeddingProvider requires provider='local_hash', got {self.config.provider!r}"
            )
        assert self.config.dimensions is not None
        self._dimensions: int = self.config.dimensions

    @property
    def model_id(self) -> str:
        return f"local_hash:{self.config.model}:d{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_texts((text,))[0]

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingResponseFormatError("texts must be a sequence of strings")
        if not texts:
            return ()

        results: list[tuple[float, ...]] = []
        for raw_text in texts:
            bounded = bound_embedding_input(raw_text, self.config.max_input_chars)
            results.append(self._project_text(bounded))
        return tuple(results)

    def _project_text(self, bounded_text: str) -> tuple[float, ...]:
        primary_tokens = tokenize(bounded_text)
        if not primary_tokens:
            raise EmbeddingResponseFormatError(
                "Input text contains no indexable alphanumeric tokens"
            )

        token_counts = Counter(primary_tokens)
        features: dict[str, float] = {}

        # 1. Primary lexical tokens (weight 1.0 * sublinear tf)
        for token, count in token_counts.items():
            features[f"tok:{token}"] = 1.0 * (1.0 + math.log(count))

        # 2. Single-hop curated synonym expansions (weight 0.55)
        for token in token_counts:
            for expanded in SEMANTIC_EXPANSIONS.get(token, ()):
                key = f"tok:{expanded}"
                if key not in features:
                    features[key] = 0.55

        # 3. Character 3-grams and 4-grams for morphological/stem overlap (weight 0.25)
        ngram_counts: Counter[str] = Counter()
        for token in primary_tokens:
            if len(token) >= 4:
                for n in (3, 4):
                    if len(token) >= n:
                        for i in range(len(token) - n + 1):
                            ngram_counts[token[i : i + n]] += 1
        for ngram, count in ngram_counts.items():
            features[f"ng:{ngram}"] = 0.25 * (1.0 + 0.5 * math.log(count))

        buckets = [0.0] * self._dimensions
        for feature_key, weight in sorted(features.items()):
            digest = hashlib.blake2b(feature_key.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            buckets[idx] += sign * weight

        # Ensure non-zero vector even in the degenerate case of exact cancellation
        if all( abs(val) <= _NORM_EPSILON for val in buckets ):
            first_digest = hashlib.blake2b(
                primary_tokens[0].encode("utf-8"), digest_size=4
            ).digest()
            buckets[int.from_bytes(first_digest, "little") % self._dimensions] = 1.0

        return validate_and_normalize_vector(
            buckets,
            expected_dimensions=self._dimensions,
            normalize=self.config.normalize,
        )


class _BaseHTTPEmbeddingProvider:
    """Shared HTTP transport and batching utilities for stdlib embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._observed_dimensions: int | None = config.dimensions

    @property
    def model_id(self) -> str:
        return f"{self.config.provider}:{self.config.model}"

    @property
    def dimensions(self) -> int | None:
        return self._observed_dimensions

    def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = self.embed_texts((text,))
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingResponseFormatError("texts must be a sequence of strings")
        if not texts:
            return ()

        bounded_texts = [
            bound_embedding_input(item, self.config.max_input_chars) for item in texts
        ]

        all_vectors: list[tuple[float, ...]] = []
        expected_dim: int | None = self.config.dimensions
        batch_size = self.config.batch_size

        for start in range(0, len(bounded_texts), batch_size):
            batch = bounded_texts[start : start + batch_size]
            raw_vectors = self._request_batch_embeddings(batch)
            if not isinstance(raw_vectors, Sequence) or len(raw_vectors) != len(batch):
                raise EmbeddingResponseFormatError(
                    f"Provider '{self.model_id}' returned {len(raw_vectors) if isinstance(raw_vectors, Sequence) else 'non-sequence'} embeddings for batch of size {len(batch)}"
                )
            for raw_vec in raw_vectors:
                validated = validate_and_normalize_vector(
                    raw_vec,
                    expected_dimensions=expected_dim,
                    normalize=self.config.normalize,
                )
                if expected_dim is None:
                    expected_dim = len(validated)
                all_vectors.append(validated)

        if expected_dim is not None and self._observed_dimensions is None:
            self._observed_dimensions = expected_dim
        return tuple(all_vectors)

    def _request_batch_embeddings(self, batch_texts: list[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError

    def _http_post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        secret_to_redact: str | None = None,
    ) -> dict[str, Any]:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                error_body = ""
            safe_msg = _redact_secret(
                f"HTTP {exc.code} from {self.config.provider} embedding endpoint: {error_body or exc.reason}",
                secret_to_redact,
            )
            raise EmbeddingUnavailableError(safe_msg) from None
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise EmbeddingTimeoutError(
                    f"Embedding request to {self.config.provider} timed out after {self.config.timeout_seconds}s"
                ) from None
            safe_msg = _redact_secret(
                f"Failed to reach {self.config.provider} embedding endpoint: {reason}",
                secret_to_redact,
            )
            raise EmbeddingUnavailableError(safe_msg) from None
        except (TimeoutError, socket.timeout):
            raise EmbeddingTimeoutError(
                f"Embedding request to {self.config.provider} timed out after {self.config.timeout_seconds}s"
            ) from None
        except OSError as exc:
            safe_msg = _redact_secret(
                f"Transport error communicating with {self.config.provider} embedding endpoint: {exc}",
                secret_to_redact,
            )
            raise EmbeddingUnavailableError(safe_msg) from None

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise EmbeddingResponseFormatError(
                f"Provider '{self.config.provider}' returned non-JSON embedding HTTP body: {exc}"
            ) from None
        if not isinstance(parsed, dict):
            raise EmbeddingResponseFormatError(
                f"Provider '{self.config.provider}' returned top-level {type(parsed).__name__}, expected JSON object"
            )
        return parsed


class OpenAICompatibleEmbeddingProvider(_BaseHTTPEmbeddingProvider):
    """Stdlib HTTP embedding provider for OpenAI-compatible and Ollama /v1/embeddings endpoints."""

    def __init__(self, config: EmbeddingConfig):
        if config.provider not in {"ollama", "openai_compatible"}:
            raise ValueError(
                f"OpenAICompatibleEmbeddingProvider only supports 'ollama' or 'openai_compatible', got {config.provider!r}"
            )
        super().__init__(config)

    def _request_batch_embeddings(self, batch_texts: list[str]) -> Sequence[Sequence[float]]:
        api_key = self.config.resolve_api_key(required=False)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self.config.base_url}/embeddings"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": list(batch_texts),
        }
        if self.config.dimensions is not None and self.config.provider == "openai_compatible":
            payload["dimensions"] = self.config.dimensions

        response_json = self._http_post_json(url, payload, headers, secret_to_redact=api_key)

        if "data" in response_json:
            data = response_json.get("data")
            if not isinstance(data, list):
                raise EmbeddingResponseFormatError(
                    "OpenAI-compatible embedding response missing 'data' list"
                )
            indexed_items: list[tuple[int, Any]] = []
            for pos, item in enumerate(data):
                if not isinstance(item, Mapping) or "embedding" not in item:
                    raise EmbeddingResponseFormatError(
                        "OpenAI-compatible embedding entry must be an object containing 'embedding'"
                    )
                idx = item.get("index", pos)
                if isinstance(idx, bool) or not isinstance(idx, int):
                    idx = pos
                indexed_items.append((idx, item["embedding"]))
            indexed_items.sort(key=lambda pair: pair[0])
            return [vec for _, vec in indexed_items]

        if "embeddings" in response_json:
            embeddings = response_json.get("embeddings")
            if not isinstance(embeddings, list):
                raise EmbeddingResponseFormatError(
                    "Ollama embedding response 'embeddings' field must be a list"
                )
            return embeddings

        raise EmbeddingResponseFormatError(
            "Embedding response must contain 'data' or 'embeddings' array"
        )


class GeminiEmbeddingProvider(_BaseHTTPEmbeddingProvider):
    """Stdlib HTTP embedding provider for the native Google Gemini batchEmbedContents API."""

    def __init__(self, config: EmbeddingConfig):
        if config.provider != "gemini":
            raise ValueError(
                f"GeminiEmbeddingProvider requires provider='gemini', got {config.provider!r}"
            )
        super().__init__(config)

    def _request_batch_embeddings(self, batch_texts: list[str]) -> Sequence[Sequence[float]]:
        api_key = self.config.resolve_api_key(required=True)
        assert api_key is not None
        headers = {"x-goog-api-key": api_key}

        model_path = (
            self.config.model
            if self.config.model.startswith("models/")
            else f"models/{self.config.model}"
        )
        encoded_model = urllib.parse.quote(self.config.model, safe="")
        url = f"{self.config.base_url}/models/{encoded_model}:batchEmbedContents"

        requests_payload: list[dict[str, Any]] = []
        for text in batch_texts:
            req_item: dict[str, Any] = {
                "model": model_path,
                "content": {"parts": [{"text": text}]},
            }
            if self.config.dimensions is not None:
                req_item["outputDimensionality"] = self.config.dimensions
            requests_payload.append(req_item)

        payload = {"requests": requests_payload}
        response_json = self._http_post_json(url, payload, headers, secret_to_redact=api_key)

        embeddings = response_json.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseFormatError(
                "Gemini batchEmbedContents response missing 'embeddings' list"
            )
        vectors: list[Sequence[float]] = []
        for item in embeddings:
            if not isinstance(item, Mapping) or "values" not in item:
                raise EmbeddingResponseFormatError(
                    "Gemini embedding entry must be an object containing 'values'"
                )
            vectors.append(item["values"])
        return vectors


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the configured EmbeddingProvider."""
    if not isinstance(config, EmbeddingConfig):
        raise TypeError(f"Expected EmbeddingConfig, got {type(config).__name__}")
    if config.provider == "local_hash":
        return LocalHashEmbeddingProvider(config)
    if config.provider in {"ollama", "openai_compatible"}:
        return OpenAICompatibleEmbeddingProvider(config)
    if config.provider == "gemini":
        return GeminiEmbeddingProvider(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider!r}")
