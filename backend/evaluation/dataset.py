"""Load and validate versioned, local evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.schema import EvaluationDataset

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "scoutx-small-v1" / "dataset.json"


def load_dataset(path: str | Path = DEFAULT_DATASET) -> EvaluationDataset:
    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix.lower() != ".json":
        raise ValueError("evaluation dataset must be a JSON file")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return EvaluationDataset.model_validate(payload)
