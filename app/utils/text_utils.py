"""Text processing helpers."""

import re


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split user text into clean sentence segments."""
    stripped = text.strip()
    if not stripped:
        return []

    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(stripped) if sentence.strip()]
    return sentences or [stripped]
