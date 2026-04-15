"""Application orchestration for emotion-aware speech generation."""

import asyncio
from functools import lru_cache
from uuid import uuid4

from app.core.config import settings
from app.models.schemas import (
    GenerateSpeechRequest,
    GenerateSpeechResponse,
    ProsodyPlan,
    SentenceEmotion,
    VoiceParameters,
)
from app.services.emotion_service import EmotionAnalyzer, HuggingFaceEmotionAnalyzer, VaderEmotionAnalyzer
from app.services.tts_service import (
    SpeechSegment,
    TTSProvider,
    VoiceMapper,
    build_prosody_plan,
    get_huggingface_tts_provider,
    get_tts_provider,
    get_voice_mapper,
)


class EmpathyPipeline:
    """Coordinate text analysis, voice mapping, and speech synthesis."""

    def __init__(
        self,
        emotion_analyzer: EmotionAnalyzer,
        voice_mapper: VoiceMapper,
        tts_provider: TTSProvider,
        huggingface_tts_provider: TTSProvider | None = None,
    ) -> None:
        self._emotion_analyzer = emotion_analyzer
        self._voice_mapper = voice_mapper
        self._tts_provider = tts_provider
        self._huggingface_tts_provider = huggingface_tts_provider

    async def generate(self, request: GenerateSpeechRequest) -> GenerateSpeechResponse:
        """Generate comparison-mode speech audio and metadata."""
        document_emotion = self._emotion_analyzer.analyze(request.text)
        emotional_voice = self._voice_mapper.map(document_emotion.emotion, document_emotion.intensity)
        document_prosody = build_prosody_plan(document_emotion.emotion, document_emotion.intensity, emotional_voice)
        neutral_voice = self._voice_mapper.neutral()
        output_id = uuid4().hex
        normal_path = settings.audio_dir / f"{output_id}_normal.wav"
        emotional_path = settings.audio_dir / f"{output_id}_emotional.wav"
        huggingface_path = settings.audio_dir / f"{output_id}_huggingface.wav"

        normal_segments = [
            SpeechSegment(
                text=request.text,
                voice=neutral_voice,
                emotion="neutral",
                intensity=0.0,
            )
        ]
        sentence_voice_configs = [
            self._voice_mapper.map(sentence_result.emotion, sentence_result.intensity)
            for sentence_result in document_emotion.sentences
        ]
        sentence_prosody_plans = [
            build_prosody_plan(sentence_result.emotion, sentence_result.intensity, voice_config)
            for sentence_result, voice_config in zip(document_emotion.sentences, sentence_voice_configs)
        ]
        emotional_segments = [
            SpeechSegment(
                text=sentence_result.sentence,
                voice=voice_config,
                emotion=sentence_result.emotion,
                intensity=sentence_result.intensity,
            )
            for sentence_result, voice_config in zip(document_emotion.sentences, sentence_voice_configs)
        ]

        await self._tts_provider.synthesize(normal_segments, normal_path)
        await self._tts_provider.synthesize(emotional_segments, emotional_path)
        huggingface_url = None
        huggingface_error = None
        if self._huggingface_tts_provider is not None:
            try:
                await asyncio.wait_for(
                    self._huggingface_tts_provider.synthesize(emotional_segments, huggingface_path),
                    timeout=settings.hf_tts_timeout_seconds,
                )
                huggingface_url = f"{settings.audio_base_url}/{huggingface_path.name}"
            except TimeoutError:
                huggingface_path.unlink(missing_ok=True)
                huggingface_error = (
                    "Hugging Face neural TTS timed out before audio was returned. "
                    "The app still generated the local voices."
                )
            except Exception as exc:
                huggingface_path.unlink(missing_ok=True)
                huggingface_error = str(exc)

        normal_url = f"{settings.audio_base_url}/{normal_path.name}"
        emotional_url = f"{settings.audio_base_url}/{emotional_path.name}"

        return GenerateSpeechResponse(
            emotion=document_emotion.emotion,
            sentiment=document_emotion.sentiment,
            intensity=document_emotion.intensity,
            confidence=document_emotion.confidence,
            emotion_description=document_emotion.description,
            audio_url=emotional_url,
            normal_audio_url=normal_url,
            emotional_audio_url=emotional_url,
            huggingface_audio_url=huggingface_url,
            huggingface_tts_enabled=self._huggingface_tts_provider is not None,
            huggingface_tts_error=huggingface_error,
            voice_parameters=VoiceParameters(
                rate=emotional_voice.rate,
                pitch=emotional_voice.pitch,
                volume=emotional_voice.volume,
            ),
            prosody_plan=ProsodyPlan(
                valence=document_prosody.valence,
                arousal=document_prosody.arousal,
                rate_change_percent=document_prosody.rate_change_percent,
                pitch_change_percent=document_prosody.pitch_change_percent,
                volume_change_percent=document_prosody.volume_change_percent,
                pause_ms=document_prosody.pause_ms,
                emphasis=document_prosody.emphasis,
                delivery=document_prosody.delivery,
                neural_style_prompt=document_prosody.neural_style_prompt,
            ),
            sentences=[
                SentenceEmotion(
                    sentence=result.sentence,
                    sentiment=result.sentiment,
                    emotion=result.emotion,
                    compound=result.compound,
                    intensity=result.intensity,
                    confidence=result.confidence,
                    description=result.description,
                    voice_parameters=VoiceParameters(
                        rate=voice_config.rate,
                        pitch=voice_config.pitch,
                        volume=voice_config.volume,
                    ),
                    prosody_plan=ProsodyPlan(
                        valence=prosody_plan.valence,
                        arousal=prosody_plan.arousal,
                        rate_change_percent=prosody_plan.rate_change_percent,
                        pitch_change_percent=prosody_plan.pitch_change_percent,
                        volume_change_percent=prosody_plan.volume_change_percent,
                        pause_ms=prosody_plan.pause_ms,
                        emphasis=prosody_plan.emphasis,
                        delivery=prosody_plan.delivery,
                        neural_style_prompt=prosody_plan.neural_style_prompt,
                    ),
                )
                for result, voice_config, prosody_plan in zip(
                    document_emotion.sentences,
                    sentence_voice_configs,
                    sentence_prosody_plans,
                )
            ],
        )

@lru_cache
def get_emotion_analyzer() -> EmotionAnalyzer:
    """Dependency provider for emotion analysis."""
    if settings.emotion_backend in {"hf", "huggingface", "transformer"}:
        return HuggingFaceEmotionAnalyzer(
            model_name=settings.hf_emotion_model,
            device=settings.hf_device,
        )
    return VaderEmotionAnalyzer()


@lru_cache
def get_pipeline() -> EmpathyPipeline:
    """Dependency provider for the application pipeline."""
    return EmpathyPipeline(
        emotion_analyzer=get_emotion_analyzer(),
        voice_mapper=get_voice_mapper(),
        tts_provider=get_tts_provider(),
        huggingface_tts_provider=get_huggingface_tts_provider() if settings.enable_hf_tts else None,
    )


async def warmup_pipeline_dependencies() -> None:
    """Preload reusable model-backed dependencies during app startup."""
    pipeline = get_pipeline()
    if pipeline._huggingface_tts_provider is not None:
        await pipeline._huggingface_tts_provider.warmup()
