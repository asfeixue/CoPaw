# -*- coding: utf-8 -*-
"""TTS (Text-to-Speech) engine for local model inference."""

from __future__ import annotations

import io
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        """Initialize TTS engine with model path.

        Args:
            model_path: Path to the TTS model directory or file.
            **kwargs: Engine-specific parameters.
        """
        self.model_path = Path(model_path)
        self.kwargs = kwargs
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the TTS model into memory."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bytes:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            speed: Speech speed multiplier (0.5 - 2.0).
            pitch: Pitch multiplier (0.5 - 2.0).

        Returns:
            Audio data as WAV bytes.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and release resources."""

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._is_loaded


class SambertEngine(TTSEngine):
    """TTS engine for Alibaba Sambert-HiFiGAN models from ModelScope."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self.model = None
        self.vocoder = None

    def load(self) -> None:
        """Load Sambert model."""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as e:
            raise ImportError(
                "modelscope is required for Sambert TTS. "
                "Install with: pip install modelscope"
            ) from e

        logger.info("Loading Sambert TTS model from %s", self.model_path)

        # Sambert models are directory-based with config.json
        self.model = pipeline(
            task=Tasks.text_to_speech,
            model=str(self.model_path),
        )
        self._is_loaded = True
        logger.info("Sambert TTS model loaded successfully")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bytes:
        """Synthesize text using Sambert."""
        if not self._is_loaded:
            self.load()

        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required for audio processing. "
                "Install with: pip install soundfile"
            ) from e

        result = self.model(text)
        audio_data = result["output_wav"]

        # Apply speed adjustment if needed
        if speed != 1.0:
            audio_data = self._adjust_speed(audio_data, speed)

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, samplerate=16000, format="WAV")
        wav_buffer.seek(0)
        return wav_buffer.read()

    def _adjust_speed(self, audio_data: Any, speed: float) -> Any:
        """Adjust audio speed using resampling."""
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed, skipping speed adjustment")
            return audio_data

        return librosa.effects.time_stretch(audio_data, rate=speed)

    def unload(self) -> None:
        """Unload Sambert model."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        logger.info("Sambert TTS model unloaded")


class CosyVoiceEngine(TTSEngine):
    """TTS engine for CosyVoice models from ModelScope."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self.model = None
        self.sample_rate = 22050

    def load(self) -> None:
        """Load CosyVoice model."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError as e:
            raise ImportError(
                "cosyvoice is required for CosyVoice TTS. "
                "Install with: pip install cosyvoice"
            ) from e

        logger.info("Loading CosyVoice model from %s", self.model_path)
        self.model = CosyVoice(str(self.model_path))
        self._is_loaded = True
        logger.info("CosyVoice model loaded successfully")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bytes:
        """Synthesize text using CosyVoice."""
        if not self._is_loaded:
            self.load()

        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required for CosyVoice. "
                "Install with: pip install soundfile"
            ) from e

        # Generate speech
        output = self.model.inference_sft(text, "中文女")
        audio_data = output["tts_speech"].numpy().squeeze()

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, samplerate=self.sample_rate, format="WAV")
        wav_buffer.seek(0)
        return wav_buffer.read()

    def unload(self) -> None:
        """Unload CosyVoice model."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        logger.info("CosyVoice model unloaded")


class MeloTTSEngine(TTSEngine):
    """TTS engine for MeloTTS models."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self.model = None
        self.sample_rate = 44100

    def load(self) -> None:
        """Load MeloTTS model."""
        try:
            from melo.api import TTS
        except ImportError as e:
            raise ImportError(
                "melotts is required for MeloTTS. "
                "Install with: pip install melotts"
            ) from e

        logger.info("Loading MeloTTS model from %s", self.model_path)

        # Detect language from model path or kwargs
        language = self.kwargs.get("language", "ZH")
        self.model = TTS(language=language)
        self._is_loaded = True
        logger.info("MeloTTS model loaded successfully")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bytes:
        """Synthesize text using MeloTTS."""
        if not self._is_loaded:
            self.load()

        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required for audio processing."
            ) from e

        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate speech
            speaker_id = self.kwargs.get("speaker_id", 0)
            self.model.tts_to_file(text, speaker_id, tmp_path, speed=speed)

            # Read the generated audio
            audio_data, sr = sf.read(tmp_path)
            self.sample_rate = sr

            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, samplerate=sr, format="WAV")
            wav_buffer.seek(0)
            return wav_buffer.read()
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def unload(self) -> None:
        """Unload MeloTTS model."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        logger.info("MeloTTS model unloaded")


class TTSEngineFactory:
    """Factory for creating TTS engine instances."""

    _engines: dict[str, type[TTSEngine]] = {
        "sambert": SambertEngine,
        "cosyvoice": CosyVoiceEngine,
        "melotts": MeloTTSEngine,
    }

    @classmethod
    def create_engine(
        cls,
        model_path: str,
        engine_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TTSEngine:
        """Create a TTS engine instance.

        Args:
            model_path: Path to the TTS model.
            engine_type: Type of engine (sambert, cosyvoice, melotts).
                If None, auto-detect from model path.
            **kwargs: Additional engine parameters.

        Returns:
            TTSEngine instance.
        """
        if engine_type is None:
            engine_type = cls.detect_engine_type(model_path)

        engine_class = cls._engines.get(engine_type.lower())
        if engine_class is None:
            raise ValueError(f"Unknown TTS engine type: {engine_type}")

        return engine_class(model_path, **kwargs)

    @classmethod
    def detect_engine_type(cls, model_path: str) -> str:
        """Auto-detect engine type from model path/name."""
        path_lower = model_path.lower()

        if "sambert" in path_lower or "hifigan" in path_lower:
            return "sambert"
        elif "cosyvoice" in path_lower:
            return "cosyvoice"
        elif "melo" in path_lower:
            return "melotts"
        else:
            # Default to sambert for ModelScope models
            return "sambert"

    @classmethod
    def register_engine(cls, name: str, engine_class: type[TTSEngine]) -> None:
        """Register a custom TTS engine."""
        cls._engines[name.lower()] = engine_class


# Global engine cache
_engine_cache: dict[str, TTSEngine] = {}


def get_tts_engine(model_path: str, engine_type: Optional[str] = None) -> TTSEngine:
    """Get or create a cached TTS engine.

    Args:
        model_path: Path to the TTS model.
        engine_type: Type of engine (auto-detect if None).

    Returns:
        TTSEngine instance.
    """
    cache_key = f"{model_path}:{engine_type or 'auto'}"

    if cache_key not in _engine_cache:
        engine = TTSEngineFactory.create_engine(model_path, engine_type)
        engine.load()
        _engine_cache[cache_key] = engine

    return _engine_cache[cache_key]


def unload_tts_engine(model_path: str, engine_type: Optional[str] = None) -> None:
    """Unload a cached TTS engine.

    Args:
        model_path: Path to the TTS model.
        engine_type: Type of engine.
    """
    cache_key = f"{model_path}:{engine_type or 'auto'}"

    if cache_key in _engine_cache:
        _engine_cache[cache_key].unload()
        del _engine_cache[cache_key]


def synthesize_speech(
    text: str,
    model_path: str,
    engine_type: Optional[str] = None,
    speed: float = 1.0,
    pitch: float = 1.0,
) -> bytes:
    """Convenience function to synthesize speech.

    Args:
        text: Text to synthesize.
        model_path: Path to the TTS model.
        engine_type: Type of engine (auto-detect if None).
        speed: Speech speed multiplier.
        pitch: Pitch multiplier.

    Returns:
        Audio data as WAV bytes.
    """
    engine = get_tts_engine(model_path, engine_type)
    return engine.synthesize(text, speed=speed, pitch=pitch)
