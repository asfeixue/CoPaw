# -*- coding: utf-8 -*-
"""API endpoints for TTS (Text-to-Speech) functionality."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

# Directory for storing generated audio files
AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "copaw_tts_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""

    text: str = Field(..., description="Text to synthesize into speech")
    model_id: str = Field(..., description="TTS model ID (from local models)")
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5 - 2.0)",
    )
    pitch: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Pitch multiplier (0.5 - 2.0)",
    )
    engine_type: Optional[str] = Field(
        default=None,
        description="TTS engine type (sambert, cosyvoice, melotts). Auto-detect if not specified.",
    )


class TTSResponse(BaseModel):
    """Response model for TTS synthesis."""

    audio_url: str = Field(..., description="URL to download the generated audio")
    duration: float = Field(default=0.0, description="Estimated audio duration in seconds")
    format: str = Field(default="wav", description="Audio format")


class TTSModelInfo(BaseModel):
    """Information about an available TTS model."""

    id: str
    name: str
    engine_type: str
    language: str


@router.post(
    "/synthesize",
    response_model=TTSResponse,
    summary="Synthesize text to speech",
)
async def synthesize(request: TTSRequest) -> TTSResponse:
    """Synthesize text into speech using a local TTS model.

    Returns a URL to download the generated audio file.
    """
    try:
        from ...local_models import get_local_model
        from ...local_models.tts_engine import synthesize_speech
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="TTS dependencies not installed. Install with: pip install modelscope",
        ) from exc

    # Get model info
    model_info = get_local_model(request.model_id)
    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"TTS model '{request.model_id}' not found. "
            "Please download the model first.",
        )

    # Validate model is a TTS model
    if model_info.backend.value != "tts":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_id}' is not a TTS model.",
        )

    try:
        # Synthesize speech
        audio_bytes = synthesize_speech(
            text=request.text,
            model_path=model_info.local_path,
            engine_type=request.engine_type,
            speed=request.speed,
            pitch=request.pitch,
        )

        # Save to cache file
        audio_id = str(uuid.uuid4())
        audio_file = AUDIO_CACHE_DIR / f"{audio_id}.wav"
        audio_file.write_bytes(audio_bytes)

        # Estimate duration (rough approximation: 16000 samples/sec, 2 bytes/sample)
        duration = len(audio_bytes) / (16000 * 2)

        return TTSResponse(
            audio_url=f"/tts/audio/{audio_id}.wav",
            duration=duration,
            format="wav",
        )

    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {exc}",
        ) from exc


@router.get(
    "/audio/{audio_id}.wav",
    summary="Get generated audio file",
)
async def get_audio(audio_id: str) -> bytes:
    """Retrieve a generated audio file by ID."""
    audio_file = AUDIO_CACHE_DIR / f"{audio_id}.wav"

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return audio_file.read_bytes()


@router.get(
    "/models",
    response_model=list[TTSModelInfo],
    summary="List available TTS models",
)
async def list_tts_models() -> list[TTSModelInfo]:
    """List all downloaded TTS models."""
    try:
        from ...local_models import list_local_models
        from ...local_models.schema import BackendType
        from ...local_models.tts_engine import TTSEngineFactory
    except ImportError:
        return []

    models = []
    for model in list_local_models(backend=BackendType.TTS):
        # Auto-detect engine type from model path
        engine_type = TTSEngineFactory.detect_engine_type(model.local_path)

        # Determine language from model name/path
        language = "zh"
        if "en" in model.display_name.lower() or "us" in model.display_name.lower():
            language = "en"
        elif "multilingual" in model.display_name.lower():
            language = "multilingual"

        models.append(
            TTSModelInfo(
                id=model.id,
                name=model.display_name,
                engine_type=engine_type,
                language=language,
            )
        )

    return models


@router.delete(
    "/cache",
    summary="Clear TTS audio cache",
)
async def clear_cache() -> dict:
    """Clear all cached TTS audio files."""
    try:
        count = 0
        for audio_file in AUDIO_CACHE_DIR.glob("*.wav"):
            audio_file.unlink()
            count += 1

        return {"status": "success", "cleared_files": count}
    except Exception as exc:
        logger.exception("Failed to clear TTS cache: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {exc}",
        ) from exc
