"""Request and response schemas for the speech generation API."""

from pydantic import BaseModel, Field, field_validator


class GenerateSpeechRequest(BaseModel):
    """Input payload for speech generation."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject whitespace-only text."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text must not be empty.")
        return cleaned


class VoiceParameters(BaseModel):
    """Voice settings derived from emotion and intensity."""

    rate: int
    pitch: float
    volume: float


class ProsodyPlan(BaseModel):
    """Human-readable and machine-readable voice delivery plan."""

    valence: str
    arousal: str
    rate_change_percent: int
    pitch_change_percent: int
    volume_change_percent: int
    pause_ms: int
    emphasis: str
    delivery: str
    neural_style_prompt: str


class SentenceEmotion(BaseModel):
    """Emotion analysis for a single sentence."""

    sentence: str
    sentiment: str
    emotion: str
    compound: float
    intensity: float
    confidence: float
    description: str
    voice_parameters: VoiceParameters
    prosody_plan: ProsodyPlan


class GenerateSpeechResponse(BaseModel):
    """Speech generation result returned to API clients."""

    emotion: str
    sentiment: str
    intensity: float
    confidence: float
    emotion_description: str
    audio_url: str
    normal_audio_url: str
    emotional_audio_url: str
    huggingface_audio_url: str | None = None
    huggingface_tts_enabled: bool = False
    huggingface_tts_error: str | None = None
    voice_parameters: VoiceParameters
    prosody_plan: ProsodyPlan
    sentences: list[SentenceEmotion]
