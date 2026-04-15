"""Text-to-speech generation services."""

from __future__ import annotations

import asyncio
import math
import sys
import subprocess
import tempfile
import wave
from array import array
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pyttsx3

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from app.core.config import settings


@dataclass(frozen=True)
class VoiceConfig:
    """Voice parameters used by the TTS engine."""

    rate: int
    pitch: float
    volume: float


@dataclass(frozen=True)
class SpeechSegment:
    """A synthesis unit with emotional context and mapped voice settings."""

    text: str
    voice: VoiceConfig
    emotion: str = "neutral"
    intensity: float = 0.0


@dataclass(frozen=True)
class ProsodyPlan:
    """Detailed delivery plan derived from an emotion and voice configuration."""

    valence: str
    arousal: str
    rate_change_percent: int
    pitch_change_percent: int
    volume_change_percent: int
    pause_ms: int
    emphasis: str
    delivery: str
    neural_style_prompt: str


VOICE_MAPPING: dict[str, dict[str, float]] = {
    "happy": {"rate_delta": 32, "pitch_delta": 0.12, "volume_delta": 0.05},
    "sad": {"rate_delta": -38, "pitch_delta": -0.10, "volume_delta": -0.14},
    "neutral": {"rate_delta": 0, "pitch_delta": 0.0, "volume_delta": 0.0},
    "angry": {"rate_delta": 46, "pitch_delta": 0.08, "volume_delta": 0.1},
    "concerned": {"rate_delta": -18, "pitch_delta": -0.05, "volume_delta": -0.06},
    "surprised": {"rate_delta": 42, "pitch_delta": 0.16, "volume_delta": 0.08},
    "inquisitive": {"rate_delta": 12, "pitch_delta": 0.10, "volume_delta": 0.0},
}

EMOTION_PERFORMANCE: dict[str, dict[str, str | int]] = {
    "happy": {
        "valence": "positive",
        "arousal": "medium-high",
        "pause_ms": 180,
        "emphasis": "lift key positive words",
        "delivery": "warm, smiling, confident, and genuinely enthusiastic",
    },
    "sad": {
        "valence": "negative",
        "arousal": "low",
        "pause_ms": 360,
        "emphasis": "soften endings and avoid sharp attacks",
        "delivery": "soft, slower, compassionate, and slightly subdued",
    },
    "neutral": {
        "valence": "neutral",
        "arousal": "low-medium",
        "pause_ms": 240,
        "emphasis": "balanced emphasis with clean articulation",
        "delivery": "clear, natural, professional, and balanced",
    },
    "angry": {
        "valence": "negative",
        "arousal": "high",
        "pause_ms": 160,
        "emphasis": "firmly stress frustration words without shouting",
        "delivery": "firm, controlled, tense, serious, and direct",
    },
    "concerned": {
        "valence": "negative-cautious",
        "arousal": "medium",
        "pause_ms": 330,
        "emphasis": "slow down around risk or uncertainty words",
        "delivery": "warm, careful, patient, reassuring, and attentive",
    },
    "surprised": {
        "valence": "positive-alert",
        "arousal": "high",
        "pause_ms": 140,
        "emphasis": "brightly emphasize surprise and discovery words",
        "delivery": "lively, amazed, energetic, expressive, and bright",
    },
    "inquisitive": {
        "valence": "neutral-curious",
        "arousal": "medium",
        "pause_ms": 260,
        "emphasis": "use a rising question-like contour",
        "delivery": "curious, attentive, lightly rising, and engaged",
    },
}


def build_prosody_plan(emotion: str, intensity: float, voice: VoiceConfig) -> ProsodyPlan:
    """Build a detailed delivery plan for UI, API, and neural TTS prompting."""
    bounded_intensity = min(max(intensity, 0.0), 1.0)
    performance = EMOTION_PERFORMANCE.get(emotion, EMOTION_PERFORMANCE["neutral"])
    pause_base = int(performance["pause_ms"])
    pause_ms = max(90, int(pause_base * (1.15 - bounded_intensity * 0.3)))
    rate_change = round((voice.rate - settings.default_voice_rate) / settings.default_voice_rate * 100)
    pitch_change = round((voice.pitch - 1.0) * 100)
    volume_change = round((voice.volume - settings.default_voice_volume) / settings.default_voice_volume * 100)
    energy = "strong" if bounded_intensity >= 0.7 else "moderate" if bounded_intensity >= 0.35 else "subtle"
    delivery = str(performance["delivery"])
    neural_style_prompt = (
        f"{settings.hf_tts_speaker} speaks in a realistic female voice. "
        f"The tone is {delivery}. Emotion intensity is {energy}. "
        f"Use {performance['emphasis']}, {pause_ms} ms natural pauses, "
        f"a {rate_change:+d}% rate shift, {pitch_change:+d}% pitch shift, "
        f"and {volume_change:+d}% volume shift. Keep the audio clean, human, and conversational."
    )
    return ProsodyPlan(
        valence=str(performance["valence"]),
        arousal=str(performance["arousal"]),
        rate_change_percent=rate_change,
        pitch_change_percent=pitch_change,
        volume_change_percent=volume_change,
        pause_ms=pause_ms,
        emphasis=str(performance["emphasis"]),
        delivery=delivery,
        neural_style_prompt=neural_style_prompt,
    )


class VoiceMapper:
    """Map emotion labels and intensity to voice parameters."""

    def __init__(self, base_rate: int, base_volume: float) -> None:
        self._base_rate = base_rate
        self._base_volume = base_volume

    def map(self, emotion: str, intensity: float) -> VoiceConfig:
        """Create voice configuration scaled by emotional intensity."""
        bounded_intensity = min(max(intensity, 0.0), 1.0)
        mapping = VOICE_MAPPING.get(emotion, VOICE_MAPPING["neutral"])
        rate = int(self._base_rate + mapping["rate_delta"] * bounded_intensity)
        pitch = round(1.0 + mapping["pitch_delta"] * bounded_intensity, 3)
        volume = min(max(self._base_volume + mapping["volume_delta"] * bounded_intensity, 0.0), 1.0)
        return VoiceConfig(rate=rate, pitch=pitch, volume=round(volume, 3))

    def neutral(self) -> VoiceConfig:
        """Return an unmodified neutral voice configuration."""
        return VoiceConfig(rate=self._base_rate, pitch=1.0, volume=self._base_volume)


class TTSProvider(ABC):
    """Interface for swappable TTS engines."""

    @abstractmethod
    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        """Generate speech audio for the supplied text segments."""

    async def warmup(self) -> None:
        """Optionally pre-load provider dependencies before the first request."""


class Pyttsx3TTSProvider(TTSProvider):
    """Offline pyttsx3 TTS provider that writes WAV files."""

    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        """Run pyttsx3 synthesis in a worker thread to keep FastAPI responsive."""
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("At least one text segment is required for synthesis.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            return self._synthesize_sync(segment_list, output_path)
        return await asyncio.to_thread(self._synthesize_sync, segment_list, output_path)

    def _synthesize_sync(self, segments: list[SpeechSegment], output_path: Path) -> Path:
        """Synchronously generate WAV audio using pyttsx3."""
        if len(segments) == 1:
            self._save_segment(segments[0].text, segments[0].voice, output_path)
            return output_path

        temp_paths: list[Path] = []
        try:
            for index, segment in enumerate(segments):
                temp_path = output_path.with_name(f"{output_path.stem}_part_{index}{output_path.suffix}")
                self._save_segment(segment.text, segment.voice, temp_path)
                temp_paths.append(temp_path)
            self._merge_wav_files(temp_paths, output_path)
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        return output_path

    def _save_segment(self, text: str, config: VoiceConfig, output_path: Path) -> None:
        try:
            if pyttsx3 is None:
                raise RuntimeError("pyttsx3 is not installed.")
            engine = pyttsx3.init()
            self._select_preferred_voice(engine)
            engine.setProperty("rate", config.rate)
            engine.setProperty("volume", config.volume)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            engine.stop()
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("pyttsx3 did not produce an audio file.")
        except Exception as exc:
            if not sys.platform.startswith("win"):
                raise RuntimeError("pyttsx3 failed to synthesize speech.") from exc
            try:
                self._save_segment_with_windows_speech(text, config, output_path)
            except Exception:
                self._save_prosody_fallback(text, config, output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Speech synthesis completed without producing audio.")
        self._apply_pitch_shift(output_path, config.pitch)

    def _save_segment_with_windows_speech(self, text: str, config: VoiceConfig, output_path: Path) -> None:
        """Fallback to Windows System.Speech when pyttsx3 SAPI initialization is blocked."""
        speech_rate = max(-10, min(10, round((config.rate - settings.default_voice_rate) / 12)))
        speech_volume = max(0, min(100, round(config.volume * 100)))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as text_file:
            text_file.write(text)
            text_path = Path(text_file.name)

        escaped_text_path = str(text_path).replace("'", "''")
        escaped_output_path = str(output_path).replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            f"$text = Get-Content -LiteralPath '{escaped_text_path}' -Raw; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice = $speaker.GetInstalledVoices() | "
            "Where-Object { $_.VoiceInfo.Gender.ToString().ToLower() -eq 'female' } | "
            "Select-Object -First 1; "
            "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name); } "
            f"$speaker.Rate = {speech_rate}; "
            f"$speaker.Volume = {speech_volume}; "
            f"$speaker.SetOutputToWaveFile('{escaped_output_path}'); "
            "$speaker.Speak($text); "
            "$speaker.Dispose();"
        )

        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=90,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "Windows speech synthesis failed.")
            if not output_path.exists() or output_path.stat().st_size == 0:
                details = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(details or "Windows speech synthesis produced no audio file.")
        finally:
            text_path.unlink(missing_ok=True)

    def _save_prosody_fallback(self, text: str, config: VoiceConfig, output_path: Path) -> None:
        """Generate a playable emotion-modulated WAV if no OS speech voice is available."""
        sample_rate = 22050
        base_voice_frequency = 220.0 if settings.preferred_voice_gender == "female" else 185.0
        base_frequency = base_voice_frequency * config.pitch
        word_duration = max(0.08, min(0.22, 24.0 / max(config.rate, 1)))
        pause_duration = 0.035
        amplitude = int(26000 * config.volume)
        words = [word for word in text.split() if word.strip()] or ["..."]
        samples: array[int] = array("h")

        for word_index, word in enumerate(words):
            duration = word_duration + min(len(word), 12) * 0.006
            frame_total = int(sample_rate * duration)
            word_lift = 1.0 + (word_index % 4) * 0.035
            if word.endswith("?"):
                word_lift += 0.18
            if word.endswith("!"):
                word_lift += 0.12

            for frame_index in range(frame_total):
                progress = frame_index / max(frame_total - 1, 1)
                envelope = math.sin(math.pi * progress)
                vibrato = 1.0 + 0.015 * math.sin(2 * math.pi * 5 * progress)
                frequency = base_frequency * word_lift * vibrato
                sample = int(amplitude * envelope * math.sin(2 * math.pi * frequency * frame_index / sample_rate))
                samples.append(sample)

            samples.extend([0] * int(sample_rate * pause_duration))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

    def _select_preferred_voice(self, engine: "pyttsx3.Engine") -> None:
        """Prefer an installed female voice when one is available."""
        if settings.preferred_voice_gender != "female":
            return

        preferred_terms = ("female", "zira", "hazel", "susan", "heera", "elena", "aria", "jenny")
        try:
            voices = engine.getProperty("voices") or []
        except Exception:
            return

        for voice in voices:
            voice_blob = " ".join(
                str(value).lower()
                for value in (
                    getattr(voice, "id", ""),
                    getattr(voice, "name", ""),
                    getattr(voice, "gender", ""),
                )
            )
            if any(term in voice_blob for term in preferred_terms):
                engine.setProperty("voice", voice.id)
                return

    def _merge_wav_files(self, input_paths: list[Path], output_path: Path) -> None:
        params = None
        frames: list[bytes] = []

        for path in input_paths:
            with wave.open(str(path), "rb") as wav_file:
                current_params = wav_file.getparams()
                if params is None:
                    params = current_params
                elif current_params[:3] != params[:3]:
                    raise RuntimeError("Generated WAV segments have incompatible audio formats.")
                frames.append(wav_file.readframes(wav_file.getnframes()))

        if params is None:
            raise RuntimeError("No WAV data was generated.")

        with wave.open(str(output_path), "wb") as output_file:
            output_file.setparams(params)
            for frame_data in frames:
                output_file.writeframes(frame_data)

    def _apply_pitch_shift(self, path: Path, pitch_factor: float) -> None:
        """Pitch-shift WAV audio by resampling frames after synthetic sample-rate change."""
        if abs(pitch_factor - 1.0) < 0.01:
            return

        with wave.open(str(path), "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(wav_file.getnframes())

        if params.sampwidth not in (1, 2, 4) or params.nframes == 0:
            return

        typecode_by_width = {1: "b", 2: "h", 4: "i"}
        samples = array(typecode_by_width[params.sampwidth])
        samples.frombytes(frames)

        if not samples:
            return

        channel_count = params.nchannels
        frame_count = len(samples) // channel_count
        target_frame_count = max(1, int(frame_count / pitch_factor))
        resampled = array(samples.typecode)

        for output_frame_index in range(target_frame_count):
            source_frame_index = min(int(output_frame_index * pitch_factor), frame_count - 1)
            source_start = source_frame_index * channel_count
            for channel_index in range(channel_count):
                resampled.append(samples[source_start + channel_index])

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setparams(params)
            wav_file.writeframes(resampled.tobytes())


class HuggingFaceParlerTTSProvider(TTSProvider):
    """Prompt-controlled neural TTS provider backed by Parler-TTS."""

    def __init__(
        self,
        model_name: str,
        speaker_name: str,
        device: str,
        token: str | None,
    ) -> None:
        self._model_name = model_name
        self._speaker_name = speaker_name
        self._device_setting = device
        self._token = token
        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None
        self._device = None

    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        """Generate neural emotional speech in a worker thread."""
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("At least one text segment is required for synthesis.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._synthesize_sync, segment_list, output_path)

    async def warmup(self) -> None:
        """Load the model weights once so requests can reuse the in-memory model."""
        await asyncio.to_thread(self._load_model)

    def _synthesize_sync(self, segments: list[SpeechSegment], output_path: Path) -> Path:
        self._load_model()
        temp_paths: list[Path] = []
        try:
            for index, segment in enumerate(segments):
                temp_path = output_path.with_name(f"{output_path.stem}_hf_part_{index}{output_path.suffix}")
                self._generate_segment(segment, temp_path)
                temp_paths.append(temp_path)
            Pyttsx3TTSProvider()._merge_wav_files(temp_paths, output_path)
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
        return output_path

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face neural TTS requires optional dependencies. "
                "Install them with: pip install -r requirements-ml.txt"
            ) from exc

        if self._device_setting == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_setting

        kwargs = {"token": self._token} if self._token else {}
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(self._model_name, **kwargs).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, **kwargs)
        self._description_tokenizer = self._tokenizer

    def _generate_segment(self, segment: SpeechSegment, output_path: Path) -> None:
        import soundfile as sf
        import torch

        description = self._build_description(segment)
        prompt_inputs = self._tokenizer(segment.text, return_tensors="pt").to(self._device)
        description_inputs = self._description_tokenizer(description, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )

        audio = generation.cpu().numpy().squeeze()
        sampling_rate = self._model.config.sampling_rate
        sf.write(str(output_path), audio, sampling_rate)

    def _build_description(self, segment: SpeechSegment) -> str:
        return build_prosody_plan(segment.emotion, segment.intensity, segment.voice).neural_style_prompt


class KokoroTTSProvider(TTSProvider):
    """Fast local Kokoro neural TTS provider."""

    def __init__(self, voice_name: str, lang_code: str) -> None:
        self._voice_name = voice_name
        self._lang_code = lang_code
        self._pipeline = None
        self._sample_rate = 24000
        self._language = "en-us"

    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("At least one text segment is required for synthesis.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._synthesize_sync, segment_list, output_path)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load_pipeline)

    def _synthesize_sync(self, segments: list[SpeechSegment], output_path: Path) -> Path:
        self._load_pipeline()
        temp_paths: list[Path] = []
        try:
            for index, segment in enumerate(segments):
                temp_path = output_path.with_name(f"{output_path.stem}_kokoro_part_{index}{output_path.suffix}")
                self._generate_segment(segment, temp_path)
                temp_paths.append(temp_path)
            Pyttsx3TTSProvider()._merge_wav_files(temp_paths, output_path)
            if segments:
                average_pitch = sum(segment.voice.pitch for segment in segments) / len(segments)
                Pyttsx3TTSProvider()._apply_pitch_shift(output_path, average_pitch)
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
        return output_path

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from pykokoro import KokoroPipeline, PipelineConfig
            from pykokoro.generation_config import GenerationConfig
            from pykokoro.pipeline_config import TokenizerConfig
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS requires optional dependencies. "
                "Install them with: pip install pykokoro soundfile"
            ) from exc

        self._language = {
            "a": "en-us",
            "b": "en-gb",
            "e": "es",
            "f": "fr-fr",
            "h": "hi",
            "i": "it",
            "p": "pt-br",
        }.get(self._lang_code, "en-us")
        config = PipelineConfig(
            voice=self._voice_name,
            model_source="huggingface",
            model_quality="q8",
            generation=GenerationConfig(lang=self._language, speed=1.0),
            tokenizer_config=TokenizerConfig(
                use_espeak_fallback=False,
                use_spacy=True,
                spacy_model="en_core_web_sm" if self._language.startswith("en") else "auto",
                spacy_model_size="sm",
            ),
            cache_dir=str(settings.audio_dir.parent / "pykokoro_cache"),
        )
        self._pipeline = KokoroPipeline(config)

    def _generate_segment(self, segment: SpeechSegment, output_path: Path) -> None:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS requires optional audio dependencies. "
                "Install them with: pip install soundfile numpy"
            ) from exc

        from pykokoro.generation_config import GenerationConfig

        speed = min(max(segment.voice.rate / max(settings.default_voice_rate, 1), 0.85), 1.25)
        result = self._pipeline.run(
            segment.text,
            generation=GenerationConfig(
                lang=self._language,
                speed=speed,
            ),
        )
        audio = np.asarray(result.audio)
        sample_rate = getattr(result, "sample_rate", self._sample_rate) or self._sample_rate
        if audio.size == 0:
            raise RuntimeError("Kokoro TTS returned no audio.")
        audio = np.clip(audio * max(segment.voice.volume, 0.0), -1.0, 1.0)
        sf.write(str(output_path), audio, sample_rate)


class HuggingFaceInferenceTTSProvider(TTSProvider):
    """Token-based Hugging Face Inference API TTS provider."""

    def __init__(
        self,
        model_name: str,
        speaker_name: str,
        token: str | None,
        provider: str | None = None,
        timeout_seconds: int = 45,
    ) -> None:
        self._model_name = model_name
        self._speaker_name = speaker_name
        self._token = token
        self._provider = provider
        self._timeout_seconds = timeout_seconds

    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        """Generate speech through Hugging Face hosted inference."""
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("At least one text segment is required for synthesis.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._synthesize_sync, segment_list, output_path)

    def _synthesize_sync(self, segments: list[SpeechSegment], output_path: Path) -> Path:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face hosted TTS requires huggingface_hub. "
                "Install optional dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        kwargs = {"token": self._token}
        if self._provider:
            kwargs["provider"] = self._provider
        kwargs["timeout"] = self._timeout_seconds
        client = InferenceClient(model=self._model_name, **kwargs)
        styled_text = self._build_hosted_prompt(segments)
        audio = client.text_to_speech(styled_text, model=self._model_name)
        audio_bytes = audio if isinstance(audio, bytes) else getattr(audio, "blob", None)
        if not audio_bytes:
            raise RuntimeError("Hugging Face hosted TTS returned no audio bytes.")
        output_path.write_bytes(audio_bytes)
        return output_path

    def _build_hosted_prompt(self, segments: list[SpeechSegment]) -> str:
        prompt_parts = []
        for segment in segments:
            plan = build_prosody_plan(segment.emotion, segment.intensity, segment.voice)
            prompt_parts.append(f"{plan.neural_style_prompt}\nText: {segment.text}")
        return "\n\n".join(prompt_parts)


class HybridHuggingFaceTTSProvider(TTSProvider):
    """Try local Parler-TTS first, then hosted Hugging Face inference."""

    def __init__(self, primary_provider: TTSProvider, fallback_provider: TTSProvider) -> None:
        self._primary_provider = primary_provider
        self._fallback_provider = fallback_provider

    async def synthesize(self, segments: Iterable[SpeechSegment], output_path: Path) -> Path:
        segment_list = list(segments)
        try:
            return await self._primary_provider.synthesize(segment_list, output_path)
        except Exception:
            return await self._fallback_provider.synthesize(segment_list, output_path)

    async def warmup(self) -> None:
        """Warm the primary provider and only fall back if that setup fails."""
        try:
            await self._primary_provider.warmup()
        except Exception:
            await self._fallback_provider.warmup()

@lru_cache
def get_voice_mapper() -> VoiceMapper:
    """Dependency provider for voice mapping."""
    return VoiceMapper(
        base_rate=settings.default_voice_rate,
        base_volume=settings.default_voice_volume,
    )

@lru_cache
def get_tts_provider() -> TTSProvider:
    """Dependency provider for TTS synthesis."""
    return Pyttsx3TTSProvider()

@lru_cache
def get_huggingface_tts_provider() -> TTSProvider:
    """Dependency provider for optional neural TTS synthesis."""
    kokoro_provider = KokoroTTSProvider(
        voice_name=settings.kokoro_voice,
        lang_code=settings.kokoro_lang_code,
    )
    local_provider = HuggingFaceParlerTTSProvider(
        model_name=settings.hf_tts_model,
        speaker_name=settings.hf_tts_speaker,
        device=settings.hf_tts_device,
        token=settings.hf_token,
    )
    hosted_provider = HuggingFaceInferenceTTSProvider(
        model_name=settings.hf_tts_model,
        speaker_name=settings.hf_tts_speaker,
        token=settings.hf_token,
        provider=settings.hf_tts_provider,
        timeout_seconds=settings.hf_tts_timeout_seconds,
    )
    if settings.hf_tts_backend == "kokoro":
        return kokoro_provider
    if settings.hf_tts_backend == "local":
        return local_provider
    if settings.hf_tts_backend in {"api", "hosted", "inference"}:
        return hosted_provider
    if settings.hf_tts_backend == "auto":
        return kokoro_provider
    if settings.hf_token or settings.hf_tts_provider:
        return HybridHuggingFaceTTSProvider(primary_provider=hosted_provider, fallback_provider=local_provider)
    return HybridHuggingFaceTTSProvider(primary_provider=local_provider, fallback_provider=hosted_provider)
