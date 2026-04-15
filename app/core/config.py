"""Runtime configuration for The Empathy Engine."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path

load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from defaults and environment variables."""

    app_name: str
    app_version: str
    static_dir: Path
    audio_dir: Path
    templates_dir: Path
    default_voice_rate: int
    default_voice_volume: float
    preferred_voice_gender: str
    audio_base_url: str
    emotion_backend: str
    hf_emotion_model: str
    hf_device: int
    enable_hf_tts: bool
    hf_tts_model: str
    hf_tts_backend: str
    hf_tts_speaker: str
    hf_tts_device: str
    hf_tts_provider: str | None
    hf_token: str | None
    hf_tts_timeout_seconds: int
    kokoro_voice: str
    kokoro_lang_code: str


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""
    static_dir = Path(os.getenv("EMPATHY_STATIC_DIR", APP_DIR / "static"))
    return Settings(
        app_name=os.getenv("EMPATHY_APP_NAME", "The Empathy Engine"),
        app_version=os.getenv("EMPATHY_APP_VERSION", "1.0.0"),
        static_dir=static_dir,
        audio_dir=Path(os.getenv("EMPATHY_AUDIO_DIR", static_dir / "audio")),
        templates_dir=Path(os.getenv("EMPATHY_TEMPLATES_DIR", APP_DIR / "templates")),
        default_voice_rate=int(os.getenv("EMPATHY_DEFAULT_VOICE_RATE", "180")),
        default_voice_volume=float(os.getenv("EMPATHY_DEFAULT_VOICE_VOLUME", "0.9")),
        preferred_voice_gender=os.getenv("EMPATHY_PREFERRED_VOICE_GENDER", "female").lower(),
        audio_base_url=os.getenv("EMPATHY_AUDIO_BASE_URL", "/static/audio"),
        emotion_backend=os.getenv("EMPATHY_EMOTION_BACKEND", "vader").lower(),
        hf_emotion_model=os.getenv(
            "EMPATHY_HF_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        ),
        hf_device=int(os.getenv("EMPATHY_HF_DEVICE", "-1")),
        enable_hf_tts=os.getenv("EMPATHY_ENABLE_HF_TTS", "false").lower() in {"1", "true", "yes", "on"},
        hf_tts_model=os.getenv("EMPATHY_HF_TTS_MODEL", "parler-tts/parler-tts-mini-expresso"),
        hf_tts_backend=os.getenv("EMPATHY_HF_TTS_BACKEND", "auto").lower(),
        hf_tts_speaker=os.getenv("EMPATHY_HF_TTS_SPEAKER", "Elisabeth"),
        hf_tts_device=os.getenv("EMPATHY_HF_TTS_DEVICE", "auto"),
        hf_tts_provider=os.getenv("EMPATHY_HF_TTS_PROVIDER"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        hf_tts_timeout_seconds=int(os.getenv("EMPATHY_HF_TTS_TIMEOUT_SECONDS", "45")),
        kokoro_voice=os.getenv("EMPATHY_KOKORO_VOICE", "af_heart"),
        kokoro_lang_code=os.getenv("EMPATHY_KOKORO_LANG_CODE", "a"),
    )


settings = get_settings()
settings.audio_dir.mkdir(parents=True, exist_ok=True)
