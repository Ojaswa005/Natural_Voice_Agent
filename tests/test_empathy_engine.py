"""Focused tests for core Empathy Engine behavior."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.schemas import GenerateSpeechRequest
from app.services.emotion_service import VaderEmotionAnalyzer
from app.services.pipeline import EmpathyPipeline
from app.services.tts_service import KokoroTTSProvider, SpeechSegment, TTSProvider, VoiceConfig, VoiceMapper


class FakeTTSProvider(TTSProvider):
    """Test double that records synthesis requests without invoking OS voices."""

    def __init__(self) -> None:
        self.calls: list[list[SpeechSegment]] = []

    async def synthesize(self, segments, output_path: Path) -> Path:
        segment_list = list(segments)
        self.calls.append(segment_list)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-wav")
        return output_path


class SlowTTSProvider(TTSProvider):
    """Test double that exceeds the pipeline timeout."""

    async def synthesize(self, segments, output_path: Path) -> Path:
        await asyncio.sleep(0.05)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"slow-fake-wav")
        return output_path


class WarmableTTSProvider(TTSProvider):
    """Test double that records warmup calls."""

    def __init__(self) -> None:
        self.warmup_calls = 0

    async def synthesize(self, segments, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"warm-fake-wav")
        return output_path

    async def warmup(self) -> None:
        self.warmup_calls += 1


def test_voice_mapper_scales_parameters_by_emotion_intensity() -> None:
    mapper = VoiceMapper(base_rate=180, base_volume=0.9)

    happy = mapper.map("happy", 1.0)
    sad = mapper.map("sad", 1.0)
    surprised = mapper.map("surprised", 0.8)

    assert happy.rate > 180
    assert happy.pitch > 1.0
    assert sad.rate < 180
    assert sad.volume < 0.9
    assert surprised.pitch > happy.pitch


def test_vader_analyzer_detects_granular_sentence_emotions() -> None:
    try:
        analyzer = VaderEmotionAnalyzer()
    except LookupError:
        pytest.skip("NLTK vader_lexicon is not installed.")

    result = analyzer.analyze("Wow, this is incredible! I am worried about the delay. Why now?")
    sentence_emotions = [sentence.emotion for sentence in result.sentences]

    assert "surprised" in sentence_emotions
    assert "concerned" in sentence_emotions
    assert "inquisitive" in sentence_emotions
    assert result.intensity > 0


def test_pipeline_generates_comparison_audio_and_sentence_metadata() -> None:
    try:
        analyzer = VaderEmotionAnalyzer()
    except LookupError:
        pytest.skip("NLTK vader_lexicon is not installed.")

    fake_tts = FakeTTSProvider()
    pipeline = EmpathyPipeline(
        emotion_analyzer=analyzer,
        voice_mapper=VoiceMapper(base_rate=180, base_volume=0.9),
        tts_provider=fake_tts,
    )

    response = asyncio.run(
        pipeline.generate(GenerateSpeechRequest(text="This is wonderful! I am worried about tomorrow."))
    )

    assert response.audio_url == response.emotional_audio_url
    assert response.normal_audio_url.endswith("_normal.wav")
    assert response.emotional_audio_url.endswith("_emotional.wav")
    assert response.huggingface_audio_url is None
    assert response.prosody_plan.delivery
    assert response.prosody_plan.neural_style_prompt
    assert len(fake_tts.calls) == 2
    assert len(response.sentences) == 2
    assert all(sentence.voice_parameters.rate > 0 for sentence in response.sentences)
    assert all(sentence.description for sentence in response.sentences)
    assert all(sentence.prosody_plan.emphasis for sentence in response.sentences)


def test_pipeline_times_out_huggingface_tts_and_returns_local_audio(monkeypatch) -> None:
    try:
        analyzer = VaderEmotionAnalyzer()
    except LookupError:
        pytest.skip("NLTK vader_lexicon is not installed.")

    fake_tts = FakeTTSProvider()
    pipeline = EmpathyPipeline(
        emotion_analyzer=analyzer,
        voice_mapper=VoiceMapper(base_rate=180, base_volume=0.9),
        tts_provider=fake_tts,
        huggingface_tts_provider=SlowTTSProvider(),
    )
    monkeypatch.setattr(
        "app.services.pipeline.settings",
        SimpleNamespace(audio_dir=Path("app/static/audio"), audio_base_url="/static/audio", hf_tts_timeout_seconds=0.01),
    )

    response = asyncio.run(
        pipeline.generate(GenerateSpeechRequest(text="This is wonderful! I am worried about tomorrow."))
    )

    assert response.audio_url == response.emotional_audio_url
    assert response.huggingface_audio_url is None
    assert response.huggingface_tts_enabled is True
    assert "timed out" in (response.huggingface_tts_error or "")


def test_pipeline_dependencies_are_cached_and_warmed_once(monkeypatch) -> None:
    from app.services import pipeline as pipeline_module

    warmable_provider = WarmableTTSProvider()
    pipeline_module.get_pipeline.cache_clear()
    pipeline_module.get_emotion_analyzer.cache_clear()

    monkeypatch.setattr(pipeline_module, "get_emotion_analyzer", lambda: VaderEmotionAnalyzer())
    monkeypatch.setattr(pipeline_module, "get_voice_mapper", lambda: VoiceMapper(base_rate=180, base_volume=0.9))
    monkeypatch.setattr(pipeline_module, "get_tts_provider", lambda: FakeTTSProvider())
    monkeypatch.setattr(pipeline_module, "get_huggingface_tts_provider", lambda: warmable_provider)
    monkeypatch.setattr(pipeline_module, "settings", SimpleNamespace(enable_hf_tts=True))

    first = pipeline_module.get_pipeline()
    second = pipeline_module.get_pipeline()
    asyncio.run(pipeline_module.warmup_pipeline_dependencies())

    assert first is second
    assert warmable_provider.warmup_calls == 1


def test_kokoro_backend_selection(monkeypatch) -> None:
    from app.services import tts_service as tts_module

    tts_module.get_huggingface_tts_provider.cache_clear()
    monkeypatch.setattr(
        tts_module,
        "settings",
        SimpleNamespace(
            kokoro_voice="af_heart",
            kokoro_lang_code="a",
            hf_tts_model="parler-tts/parler-tts-mini-expresso",
            hf_tts_speaker="Elisabeth",
            hf_tts_device="auto",
            hf_token=None,
            hf_tts_provider=None,
            hf_tts_timeout_seconds=45,
            hf_tts_backend="kokoro",
        ),
    )

    provider = tts_module.get_huggingface_tts_provider()

    assert isinstance(provider, KokoroTTSProvider)
