"""Timestamp-preserving speech transcription for supported audio and video."""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

from app.ingestion.normalized import ExtractedDocument, ExtractionStatus, make_block
from app.security.parser_sandbox import DEFAULT_LIMITS, run_tool, validate_input

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg"}


class MediaParser:
    name = "ffmpeg-faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        model_factory=None,
    ):
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self._model_factory = model_factory
        self._model = None

    def parse(self, path: str | Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        path = Path(path)
        ext = path.suffix.lower()
        if ext not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.UNSUPPORTED,
                blocks=[],
                parser_name=self.name,
                parser_version="unknown",
                warnings=["Unsupported media extension"],
            )
        try:
            path = validate_input(path)
            metadata = self._probe(path)
            duration = float((metadata.get("format") or {}).get("duration") or 0)
            if duration > DEFAULT_LIMITS.max_media_seconds:
                raise ValueError(f"Media exceeds {DEFAULT_LIMITS.max_media_seconds} second limit")
            if duration <= 0:
                raise ValueError("Media has no measurable duration")
            with tempfile.TemporaryDirectory(prefix="scoutx-media-") as temp_dir:
                wav_path = Path(temp_dir) / "audio.wav"
                run_tool(
                    "ffmpeg",
                    [
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-f",
                        "wav",
                        str(wav_path),
                    ],
                    timeout=DEFAULT_LIMITS.subprocess_timeout_seconds,
                )
                if not wav_path.is_file() or wav_path.stat().st_size <= 44:
                    raise ValueError("FFmpeg produced no audio track")
                segments, info = self._transcribe(wav_path)
            whisper_version = version("faster-whisper")
            blocks = [
                make_block(
                    text.strip(),
                    file_id=file_id,
                    content_version=content_version,
                    parser_name=self.name,
                    parser_version=whisper_version,
                    content_type="transcript",
                    language=getattr(info, "language", None),
                    start_time_ms=max(0, round(start * 1000)),
                    end_time_ms=max(0, round(end * 1000)),
                    section_title=f"{start:.2f}s–{end:.2f}s",
                )
                for start, end, text in segments
                if text.strip()
            ]
            warnings = [] if blocks else ["Transcription completed without speech segments"]
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.SUCCESS if blocks else ExtractionStatus.PARTIAL,
                blocks=blocks,
                parser_name=self.name,
                parser_version=whisper_version,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 - media/model errors are per-file failures
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.FAILED,
                blocks=[],
                parser_name=self.name,
                parser_version="unknown",
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _probe(path: Path) -> dict:
        result = run_tool(
            "ffprobe",
            [
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
        return json.loads(result.stdout or "{}")

    def _transcribe(self, wav_path: Path):
        if self._model is None:
            if self._model_factory:
                self._model = self._model_factory()
            else:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
        segment_iter, info = self._model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
        segments = [
            (float(segment.start), float(segment.end), str(segment.text))
            for segment in segment_iter
        ]
        return segments, info
