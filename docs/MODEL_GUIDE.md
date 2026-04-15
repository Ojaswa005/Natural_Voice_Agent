# Hugging Face Model Upgrade Guide

This project runs offline with VADER and `pyttsx3` by default. For a stronger demo, enable optional Hugging Face models for richer emotion detection and a third neural voice comparison output.

## Recommended Emotion Models

| Model | Best use | Notes |
| --- | --- | --- |
| `j-hartmann/emotion-english-distilroberta-base` | Strong default for English emotion classification | Detects emotion labels such as anger, disgust, fear, joy, neutral, sadness, and surprise. Good balance of quality and speed. |
| `SamLowe/roberta-base-go_emotions` | More granular emotion categories | Based on GoEmotions-style labels. Useful when you want labels like curiosity, annoyance, approval, disappointment, and nervousness. |
| `cardiffnlp/twitter-roberta-base-emotion` | Short social/customer messages | Useful for concise, informal text similar to chats, comments, and social posts. |
| `bhadresh-savani/distilbert-base-uncased-emotion` | Lightweight baseline transformer | Fast DistilBERT-style option for simple emotion classification. |

## Enabling Hugging Face Emotion Detection

Install optional ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Run the app with the Hugging Face backend:

```bash
$env:EMPATHY_EMOTION_BACKEND="huggingface"
$env:EMPATHY_HF_EMOTION_MODEL="j-hartmann/emotion-english-distilroberta-base"
uvicorn app.main:app --reload
```

Use CPU:

```bash
$env:EMPATHY_HF_DEVICE="-1"
```

Use the first CUDA GPU:

```bash
$env:EMPATHY_HF_DEVICE="0"
```

## Recommended Free / Open TTS Models

| Model | Best use | Integration idea |
| --- | --- | --- |
| `parler-tts/parler-tts-mini-expresso` | Best fit for this project | Prompt-controlled expressive speech with named voices and emotion/style descriptions. This is the recommended free neural TTS option for the comparison mode. |
| `parler-tts/parler-tts-mini-v1` | General expressive prompt-controlled speech | Good if you want broader style prompting with natural voice descriptions. |
| `espnet/kan-bayashi_ljspeech_vits` | Hosted Hugging Face fallback | Less emotionally expressive, but useful for token-based hosted inference when you do not want local model loading. |
| `suno/bark` | Highly expressive generative speech | Good for demos with laughter, hesitation, music-like prosody, and nonverbal cues. Heavier and less predictable than Parler-TTS. |
| `coqui/XTTS-v2` | Voice cloning and multilingual speech | Useful when you want consistent speaker identity plus expressive generation. Check license requirements before commercial use. |
| `microsoft/speecht5_tts` | Research-friendly neural TTS | Good educational baseline for neural TTS, less emotionally expressive than Bark or Parler. |

## Enabling Hugging Face Neural TTS

Install optional ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Set your Hugging Face token. Public models may not require a token, but using one helps with rate limits and private/gated models:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

Enable the third comparison voice:

```powershell
$env:EMPATHY_ENABLE_HF_TTS="true"
$env:EMPATHY_HF_TTS_BACKEND="local"
$env:EMPATHY_HF_TTS_MODEL="parler-tts/parler-tts-mini-expresso"
$env:EMPATHY_HF_TTS_SPEAKER="Elisabeth"
uvicorn app.main:app --reload
```

Use hosted Hugging Face inference instead:

```powershell
$env:EMPATHY_ENABLE_HF_TTS="true"
$env:EMPATHY_HF_TTS_BACKEND="api"
$env:EMPATHY_HF_TTS_MODEL="espnet/kan-bayashi_ljspeech_vits"
uvicorn app.main:app --reload
```

Use `auto` mode to try local Parler-TTS first and then hosted inference:

```powershell
$env:EMPATHY_HF_TTS_BACKEND="auto"
```

The UI will then show:

- Normal voice
- Offline emotional voice
- Hugging Face neural voice

Recommended speakers for the neural voice are `Elisabeth` or another female speaker supported by the selected Parler model. The application builds prompts from the detected sentence emotion, intensity, speaking rate, pitch, volume, valence, arousal, pause strategy, and emphasis plan.

## Why Parler-TTS Is the Best Free Fit

The project needs emotion-controlled prosody, not just a clearer voice. Parler-TTS is a better fit than many generic TTS models because it accepts descriptive prompts such as "warm, careful, patient, reassuring, and attentive." That lets the system convert emotion metadata into a voice-performance instruction. Bark is expressive too, but it is heavier and less predictable. SpeechT5 and many VITS models are useful baselines, but they are less directly controllable for emotion.

## Suggested Industry Upgrade Path

1. Keep VADER as the transparent fallback.
2. Use `j-hartmann/emotion-english-distilroberta-base` as the default transformer backend.
3. Enable the built-in Parler-TTS provider for style-prompted emotional delivery.
4. Store each generation request with metadata: text, emotion, intensity, model name, voice parameters, and output path.
5. Add a model selection dropdown in the UI for demo mode.

## Why Not Make Neural TTS the Default?

Neural TTS models are heavier, may require GPU acceleration, and often download large model weights on first run. The current architecture keeps the internship submission easy to run while clearly showing how production-grade models plug into the same pipeline.
