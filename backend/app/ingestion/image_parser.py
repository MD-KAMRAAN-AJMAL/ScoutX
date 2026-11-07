"""OpenCV preprocessing and PaddleOCR text extraction for raster images."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

from app.ingestion.normalized import ExtractedDocument, ExtractionStatus, make_block
from app.security.parser_sandbox import ParserInputError, validate_input

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class ImageParser:
    name = "opencv-paddleocr"

    def __init__(self, ocr_factory=None):
        self._ocr_factory = ocr_factory
        self._engine = None

    def parse(self, path: str | Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        path = Path(path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return ExtractedDocument(file_id=file_id, content_version=content_version,
                                     status=ExtractionStatus.UNSUPPORTED, blocks=[], parser_name=self.name,
                                     parser_version="unknown", warnings=["Unsupported image extension"])
        try:
            path = validate_input(path)
            import cv2
            import numpy as np

            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError("OpenCV could not decode image")
            # OCR benefits from a grayscale, mildly denoised input; retain original dimensions.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            preprocessed = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
            engine = self._get_engine()
            lines = self._recognize(engine, preprocessed)
            paddle_version = version("paddleocr")
            blocks = []
            for text, score, polygon in lines:
                if not text.strip():
                    continue
                blocks.append(make_block(text.strip(), file_id=file_id, content_version=content_version,
                                         parser_name=self.name, parser_version=paddle_version,
                                         content_type="image_text", metadata={"confidence": score, "polygon": polygon,
                                                                               "image_width": int(image.shape[1]),
                                                                               "image_height": int(image.shape[0])}))
            warnings = [] if blocks else ["OCR completed without recognizing text"]
            return ExtractedDocument(file_id=file_id, content_version=content_version,
                                     status=ExtractionStatus.SUCCESS if blocks else ExtractionStatus.PARTIAL,
                                     blocks=blocks, parser_name=self.name, parser_version=paddle_version, warnings=warnings)
        except Exception as exc:
            return ExtractedDocument(file_id=file_id, content_version=content_version,
                                     status=ExtractionStatus.FAILED, blocks=[], parser_name=self.name,
                                     parser_version="unknown", error=f"{type(exc).__name__}: {exc}")

    def _get_engine(self):
        if self._engine is None:
            if self._ocr_factory is not None:
                self._engine = self._ocr_factory()
            else:
                from paddleocr import PaddleOCR

                # PaddleOCR 3.x uses predict(); this is adapted below for 2.x OCR packages too.
                self._engine = PaddleOCR(lang="en")
        return self._engine

    @staticmethod
    def _recognize(engine, image):
        if hasattr(engine, "predict"):
            result = engine.predict(input=image)
            normalized = []
            for page in result or []:
                payload = page.json if hasattr(page, "json") else page
                if callable(payload):
                    payload = payload()
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                if isinstance(payload, dict) and "res" in payload:
                    payload = payload["res"]
                if isinstance(payload, dict):
                    texts = payload.get("rec_texts", [])
                    scores = payload.get("rec_scores", [])
                    polygons = payload.get("dt_polys", [])
                    normalized.extend((str(text), float(scores[i]) if i < len(scores) else None,
                                       _points(polygons[i]) if i < len(polygons) else None)
                                      for i, text in enumerate(texts))
            return normalized
        result = engine.ocr(image, cls=True)
        normalized = []
        for page in result or []:
            for line in page or []:
                box, recognition = line
                normalized.append((str(recognition[0]), float(recognition[1]), _points(box)))
        return normalized


def _points(value):
    try:
        return [[float(point[0]), float(point[1])] for point in value]
    except (TypeError, ValueError, IndexError):
        return None
