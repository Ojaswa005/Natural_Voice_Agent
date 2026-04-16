# Natural Voice Agent

An emotion-aware text-to-speech system that compares normal, emotional, and neural voice generation from the same input text.

## Demo

Watch the project walkthrough here: [demo1.mp4](docs/media/demo1.mp4), due to large file size if does not open download the raw file to see on desktop.

<img width="858" height="476" alt="image" src="https://github.com/user-attachments/assets/1f93680c-0514-4c3f-a0f5-eb71aa7b518b" />


For the best GitHub README experience, keep a short preview GIF or thumbnail in `docs/media/` and link it to the full MP4.

## Overview

The Empathy Engine is a production-minded FastAPI project that turns plain text into emotionally expressive audio. It detects sentiment and intensity with NLTK VADER, derives granular emotions with transparent rules, maps emotion into voice parameters, and generates WAV output through an offline `pyttsx3` text-to-speech provider.

## Problem Statement

Most text-to-speech demos read content in a flat voice. Real applications, including accessibility tools, tutoring systems, mental health companions, and customer-support assistants, need speech that responds to emotional context. This project demonstrates a clean, extensible backend that detects emotional tone and applies controlled voice modulation while keeping the system offline-friendly.

## Features

- REST API endpoint: `POST /generate`
- Browser UI served with Jinja2 templates
- VADER sentiment classification: positive, negative, neutral
- Derived emotion labels: happy, sad, neutral, angry, concerned, surprised, inquisitive
- Compound-score-based intensity scaling with punctuation and emphasis boosts
- Dedicated emotion-to-voice mapping layer
- Multi-sentence emotion handling
- Comparison mode with normal, offline emotional, and optional Hugging Face neural WAV output
- Emotion, confidence, sentiment, and sentence-level voice metadata in the API response
- Detailed prosody plans with valence, arousal, pause timing, emphasis strategy, and neural voice prompts
- Audio-level pitch shifting for WAV output after TTS generation
- Graceful fallback audio rendering if the local operating-system TTS voice is unavailable
- Optional Hugging Face transformer emotion backend and Parler-TTS neural voice backend
- Female voice preference for offline pyttsx3/System.Speech providers
- Swappable emotion analyzer and TTS provider interfaces
- Static audio serving through FastAPI
- Focused tests for analyzer, voice mapping, and pipeline orchestration

## Architecture

The project follows clean architecture principles by separating API transport, orchestration, domain services, schemas, and utilities.

- `app/api/routes.py` owns HTTP routes and request/response handling.
- `app/services/pipeline.py` coordinates the use case: analyze text, map voice settings, synthesize audio.
- `app/services/emotion_service.py` contains emotion detection abstractions and the VADER implementation.
- `app/services/tts_service.py` contains voice mapping, TTS abstractions, and the `pyttsx3` provider.
- `app/models/schemas.py` defines typed Pydantic contracts.
- `app/core/config.py` centralizes runtime configuration.
- `app/utils/text_utils.py` handles reusable text processing.

This makes the implementation easy to extend. For example, a HuggingFace emotion model can replace `VaderEmotionAnalyzer`, and ElevenLabs or another provider can replace `Pyttsx3TTSProvider` without changing the API route.

The default TTS provider attempts `pyttsx3` first and prefers an installed female voice when available. On Windows, if local SAPI voices are blocked or unavailable, it attempts `System.Speech`, then falls back to a generated prosody WAV so the service still returns a playable audio artifact during demos. In production, the fallback would normally be replaced by a cloud or neural TTS provider.

For stronger emotion understanding, the project also includes an optional Hugging Face emotion analyzer. For a premium comparison output, it can also generate a third neural TTS file with Parler-TTS. Install `requirements-ml.txt`, set the relevant environment variables, and see `docs/MODEL_GUIDE.md` for recommended free models.

## Folder Structure

```text
empathy-engine/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── services/
│   │   ├── emotion_service.py
│   │   ├── tts_service.py
│   │   └── pipeline.py
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   │   └── text_utils.py
│   ├── static/
│   │   └── audio/
│   └── templates/
│       └── index.html
├── requirements.txt
├── README.md
├── requirements-ml.txt
├── run.py
├── docs/
│   └── MODEL_GUIDE.md
└── tests/
    └── test_empathy_engine.py
```

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the VADER lexicon once:

```bash
python -m nltk.downloader vader_lexicon
```

Run with Uvicorn:

```bash
uvicorn app.main:app --reload

```

Or run with the included script:

```bash
python run.py
```

Open the UI:

```text
http://127.0.0.1:8000
```

Run tests:

```bash
pytest
```

## Configuration

Environment variables can be set in your shell or in a `.env` file in the project root. Copy `.env.example` to `.env` and fill in your values. The application will automatically load variables from `.env` if it exists.

Enable Hugging Face emotion detection:

```bash
pip install -r requirements-ml.txt
```

On Windows PowerShell:

```powershell
$env:EMPATHY_EMOTION_BACKEND="huggingface"
$env:EMPATHY_HF_EMOTION_MODEL="j-hartmann/emotion-english-distilroberta-base"
uvicorn app.main:app --reload
```

The default Hugging Face model can be changed without code edits. For example:

```powershell
$env:EMPATHY_HF_EMOTION_MODEL="SamLowe/roberta-base-go_emotions"
```

Enable the third neural TTS comparison output with fast local Kokoro:

```powershell
$env:EMPATHY_ENABLE_HF_TTS="true"
$env:EMPATHY_HF_TTS_BACKEND="kokoro"
$env:EMPATHY_KOKORO_VOICE="af_heart"
$env:EMPATHY_KOKORO_LANG_CODE="a"
uvicorn app.main:app --reload
```

Use local Parler-TTS instead when you want heavier prompt-controlled emotion:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
$env:EMPATHY_ENABLE_HF_TTS="true"
$env:EMPATHY_HF_TTS_BACKEND="local"
$env:EMPATHY_HF_TTS_MODEL="parler-tts/parler-tts-mini-expresso"
$env:EMPATHY_HF_TTS_SPEAKER="Elisabeth"
uvicorn app.main:app --reload
```

Use token-based Hugging Face hosted inference instead of local model loading:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
$env:EMPATHY_ENABLE_HF_TTS="true"
$env:EMPATHY_HF_TTS_BACKEND="api"
$env:EMPATHY_HF_TTS_MODEL="espnet/kan-bayashi_ljspeech_vits"
uvicorn app.main:app --reload
```

Use `auto` to select the project default fast neural backend:

```powershell
$env:EMPATHY_HF_TTS_BACKEND="auto"
```

If your machine has a CUDA GPU, use:

```powershell
$env:EMPATHY_HF_TTS_DEVICE="cuda:0"
```

Otherwise the project uses CPU by default. First generation may take time because model weights must download, but Kokoro is much lighter than Parler-TTS and is the recommended fast local option.

## API Documentation

FastAPI automatically exposes interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

### Generate Speech

`POST /generate`

Request:

```json
{
  "text": "I am really happy today!"
}
```

Response:

```json
{
  "emotion": "happy",
  "sentiment": "positive",
  "intensity": 0.82,
  "confidence": 0.91,
  "emotion_description": "positive, upbeat, and warm",
  "audio_url": "/static/audio/example_emotional.wav",
  "normal_audio_url": "/static/audio/example_normal.wav",
  "emotional_audio_url": "/static/audio/example_emotional.wav",
  "huggingface_audio_url": "/static/audio/example_huggingface.wav",
  "huggingface_tts_enabled": true,
  "huggingface_tts_error": null,
  "voice_parameters": {
    "rate": 208,
    "pitch": 1.148,
    "volume": 0.941
  },
  "prosody_plan": {
    "valence": "positive",
    "arousal": "medium-high",
    "rate_change_percent": 16,
    "pitch_change_percent": 15,
    "volume_change_percent": 5,
    "pause_ms": 162,
    "emphasis": "lift key positive words",
    "delivery": "warm, smiling, confident, and genuinely enthusiastic",
    "neural_style_prompt": "Elisabeth speaks in a realistic female voice..."
  },
  "sentences": [
    {
      "sentence": "I am really happy today!",
      "sentiment": "positive",
      "emotion": "happy",
      "compound": 0.6468,
      "intensity": 0.6468,
      "confidence": 0.8264,
      "description": "positive, upbeat, and warm",
      "voice_parameters": {
        "rate": 202,
        "pitch": 1.078,
        "volume": 0.932
      },
      "prosody_plan": {
        "valence": "positive",
        "arousal": "medium-high",
        "rate_change_percent": 12,
        "pitch_change_percent": 8,
        "volume_change_percent": 4,
        "pause_ms": 172,
        "emphasis": "lift key positive words",
        "delivery": "warm, smiling, confident, and genuinely enthusiastic",
        "neural_style_prompt": "Elisabeth speaks in a realistic female voice..."
      }
    }
  ]
}
```

## Emotion Detection Logic

The default analyzer uses NLTK VADER's `compound` score for each sentence.

- `compound >= 0.05` maps to positive sentiment and usually `happy`.
- `compound <= -0.05` maps to negative sentiment and usually `sad`.
- Anger terms such as `furious`, `hate`, `annoyed`, and `frustrated` map to `angry`.
- Concern terms such as `worried`, `anxious`, `risk`, and `delay` map to `concerned`.
- Exclamation-heavy positive or surprise terms map to `surprised`.
- Question-mark-driven low-polarity text maps to `inquisitive`.
- Scores between `-0.05` and `0.05` map to `neutral` when no granular cue is present.

For multi-sentence input, each sentence is analyzed independently. The document-level emotion is selected from the dominant sentence emotions, while intensity is averaged across sentence intensities.

## Intensity Scaling

VADER's compound score ranges from `-1.0` to `1.0`. The system uses `abs(compound)` as the base emotional intensity, then adds bounded boosts for exclamation marks, question marks, and all-caps emphasis. A stronger positive or negative sentence therefore creates stronger modulation, while neutral text stays close to the baseline voice.

Example:

- `compound = 0.80` produces a high happy intensity.
- `compound = -0.75` produces a high sad or angry intensity.
- `compound = 0.02` produces a near-neutral intensity.

## Voice Mapping Design

Voice mapping lives in `app/services/tts_service.py` and is intentionally separate from emotion analysis. Each emotion has configurable deltas:

- Happy: faster rate, slightly higher pitch, slightly louder volume
- Sad: slower rate, lower pitch, softer volume
- Angry: faster rate, stronger volume, moderate pitch increase
- Concerned: slightly slower rate, lower pitch, softer volume
- Surprised: faster rate, stronger pitch lift, louder volume
- Inquisitive: mild speed increase and upward pitch lift
- Neutral: baseline rate, pitch, and volume

The `VoiceMapper` scales those deltas by intensity, so subtle emotion creates subtle modulation and intense emotion creates stronger modulation. `pyttsx3` supports rate and volume directly. The project then applies an audio-level WAV pitch transform so the generated file audibly reflects the mapped pitch value.

The `ProsodyPlan` makes the mapping explainable. It converts emotion and intensity into valence, arousal, pause timing, emphasis strategy, percentage rate/pitch/volume changes, and a neural style prompt. This is what the Hugging Face TTS backend uses to push the voice toward a more human emotional delivery.

## Extending the System

To add a new emotion model:

1. Implement `EmotionAnalyzer`.
2. Return a `DocumentEmotion`.
3. Replace the dependency provider in `app/services/pipeline.py`.

To add a new TTS engine:

1. Implement `TTSProvider`.
2. Accept `SpeechSegment` values containing text, emotion, intensity, and `VoiceConfig`.
3. Return the generated audio path.
4. Replace `get_tts_provider()`.

## Future Improvements

- Add SSML output for engines that support richer prosody control.
- Integrate Parler-TTS, Bark, ElevenLabs, or Azure Neural TTS for higher-quality emotional voices.
- Add real-time audio streaming for conversational agents.
- Add cleanup jobs for old generated audio files.
- Add authentication and request quotas for public deployment.
- Add automated tests for emotion mapping, pipeline orchestration, and API behavior.
