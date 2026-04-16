"""Pre-download and warm Kokoro assets before serving requests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.tts_service import KokoroTTSProvider, SpeechSegment, VoiceConfig


async def main() -> None:
    output_path = Path("app/static/audio/kokoro_warmup.wav")
    provider = KokoroTTSProvider("af_bella", "a")
    await provider.synthesize(
        [
            SpeechSegment(
                text="Hello. This is a Kokoro warmup run for the Natural Voice Agent.",
                voice=VoiceConfig(rate=180, pitch=1.0, volume=0.9),
                emotion="neutral",
                intensity=0.0,
            )
        ],
        output_path,
    )
    print(f"Warmed Kokoro successfully: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
