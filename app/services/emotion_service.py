"""Emotion detection services."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean

from nltk.sentiment import SentimentIntensityAnalyzer

from app.utils.text_utils import split_sentences


@dataclass(frozen=True)
class EmotionResult:
    """Emotion analysis for one sentence."""

    sentence: str
    sentiment: str
    emotion: str
    compound: float
    intensity: float
    confidence: float
    description: str


@dataclass(frozen=True)
class DocumentEmotion:
    """Aggregated emotion analysis for a full input document."""

    emotion: str
    sentiment: str
    compound: float
    intensity: float
    confidence: float
    description: str
    sentences: list[EmotionResult]


class EmotionAnalyzer(ABC):
    """Interface for pluggable emotion detection engines."""

    @abstractmethod
    def analyze(self, text: str) -> DocumentEmotion:
        """Analyze text and return document-level emotion metadata."""


class VaderEmotionAnalyzer(EmotionAnalyzer):
    """NLTK VADER-backed emotion analyzer with rule-based granular emotion cues."""

    NEGATIVE_THRESHOLD = -0.05
    POSITIVE_THRESHOLD = 0.05
    KEYWORD_GROUPS: dict[str, frozenset[str]] = {
        "angry": frozenset(
            {
                "angry",
                "anger",
                "furious",
                "rage",
                "outraged",
                "mad",
                "hate",
                "hated",
                "annoyed",
                "irritated",
                "frustrated",
                "unacceptable",
                "ridiculous",
            }
        ),
        "concerned": frozenset(
            {
                "concern",
                "concerned",
                "worried",
                "worry",
                "anxious",
                "afraid",
                "nervous",
                "risk",
                "issue",
                "problem",
                "delay",
            }
        ),
        "surprised": frozenset(
            {
                "wow",
                "amazing",
                "unexpected",
                "surprised",
                "surprise",
                "unbelievable",
                "incredible",
                "suddenly",
            }
        ),
    }
    WORD_PATTERN = re.compile(r"[a-zA-Z']+")
    EMOTION_DESCRIPTIONS = {
        "happy": "positive, upbeat, and warm",
        "sad": "low-energy, disappointed, or discouraged",
        "neutral": "balanced and factual",
        "angry": "frustrated, tense, or strongly dissatisfied",
        "concerned": "worried, cautious, and seeking reassurance",
        "surprised": "excited, amazed, or caught off guard",
        "inquisitive": "curious, questioning, and attentive",
    }

    def __init__(self) -> None:
        """Initialize the VADER sentiment analyzer."""
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> DocumentEmotion:
        """Analyze full text by scoring each sentence and aggregating results."""
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("Text must contain at least one sentence.")

        sentence_results = [self._analyze_sentence(sentence) for sentence in sentences]
        average_compound = mean(result.compound for result in sentence_results)
        average_intensity = mean(result.intensity for result in sentence_results)
        dominant = self._dominant_emotion(sentence_results, average_compound)

        return DocumentEmotion(
            emotion=dominant,
            sentiment=self._sentiment_from_compound(average_compound),
            compound=round(average_compound, 4),
            intensity=round(average_intensity, 4),
            confidence=round(mean(result.confidence for result in sentence_results), 4),
            description=self.EMOTION_DESCRIPTIONS.get(dominant, "emotionally balanced"),
            sentences=sentence_results,
        )

    def _analyze_sentence(self, sentence: str) -> EmotionResult:
        scores = self._analyzer.polarity_scores(sentence)
        compound = float(scores["compound"])
        intensity = self._scaled_intensity(sentence, compound)
        sentiment = self._sentiment_from_compound(compound)
        emotion = self._emotion_from_sentence(sentence, compound)
        confidence = self._confidence(sentence, compound, emotion)

        return EmotionResult(
            sentence=sentence,
            sentiment=sentiment,
            emotion=emotion,
            compound=round(compound, 4),
            intensity=round(intensity, 4),
            confidence=round(confidence, 4),
            description=self.EMOTION_DESCRIPTIONS.get(emotion, "emotionally balanced"),
        )

    def _sentiment_from_compound(self, compound: float) -> str:
        if compound >= self.POSITIVE_THRESHOLD:
            return "positive"
        if compound <= self.NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    def _emotion_from_sentence(self, sentence: str, compound: float) -> str:
        words = self._words(sentence)
        lowered = sentence.lower()

        if "?" in sentence and abs(compound) < 0.45:
            return "inquisitive"
        if "!" in sentence and (
            words.intersection(self.KEYWORD_GROUPS["surprised"])
            or compound >= self.POSITIVE_THRESHOLD
        ):
            return "surprised"
        if words.intersection(self.KEYWORD_GROUPS["angry"]):
            return "angry"
        if words.intersection(self.KEYWORD_GROUPS["concerned"]) or "not sure" in lowered:
            return "concerned"
        if compound <= self.NEGATIVE_THRESHOLD:
            return "sad"
        if compound >= self.POSITIVE_THRESHOLD:
            return "happy"
        return "neutral"

    def _dominant_emotion(self, results: list[EmotionResult], average_compound: float) -> str:
        emotion_scores: dict[str, float] = {}
        for result in results:
            emotion_scores[result.emotion] = emotion_scores.get(result.emotion, 0.0) + result.intensity + 0.1

        if emotion_scores:
            highest_score = max(emotion_scores.values())
            candidates = {emotion for emotion, score in emotion_scores.items() if score == highest_score}
            for priority in ("angry", "concerned", "sad", "surprised", "happy", "inquisitive", "neutral"):
                if priority in candidates:
                    return priority

        if average_compound > 0:
            return "happy"
        if average_compound < 0:
            return "sad"
        return "neutral"

    def _scaled_intensity(self, sentence: str, compound: float) -> float:
        punctuation_boost = min(sentence.count("!") * 0.08 + sentence.count("?") * 0.03, 0.2)
        all_caps_words = [word for word in sentence.split() if len(word) > 2 and word.isupper()]
        caps_boost = min(len(all_caps_words) * 0.05, 0.15)
        return min(abs(compound) + punctuation_boost + caps_boost, 1.0)

    def _confidence(self, sentence: str, compound: float, emotion: str) -> float:
        keyword_bonus = 0.15 if any(self._words(sentence).intersection(group) for group in self.KEYWORD_GROUPS.values()) else 0.0
        punctuation_bonus = 0.05 if any(mark in sentence for mark in "!?") else 0.0
        neutral_penalty = -0.1 if emotion == "neutral" else 0.0
        return min(max(0.55 + abs(compound) * 0.35 + keyword_bonus + punctuation_bonus + neutral_penalty, 0.0), 0.99)

    def _words(self, sentence: str) -> set[str]:
        return {match.group(0).lower().strip("'") for match in self.WORD_PATTERN.finditer(sentence)}


class HuggingFaceEmotionAnalyzer(EmotionAnalyzer):
    """Hugging Face transformer-backed emotion analyzer."""

    LABEL_MAP = {
        "joy": "happy",
        "love": "happy",
        "optimism": "happy",
        "happy": "happy",
        "happiness": "happy",
        "sadness": "sad",
        "sad": "sad",
        "grief": "sad",
        "anger": "angry",
        "annoyance": "angry",
        "disapproval": "angry",
        "fear": "concerned",
        "nervousness": "concerned",
        "confusion": "concerned",
        "remorse": "concerned",
        "surprise": "surprised",
        "realization": "surprised",
        "curiosity": "inquisitive",
        "neutral": "neutral",
    }

    def __init__(self, model_name: str, device: int = -1) -> None:
        """Load a Hugging Face text-classification pipeline lazily at startup."""
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face emotion backend requires optional dependencies. "
                "Install them with: pip install -r requirements-ml.txt"
            ) from exc

        self._classifier = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
            device=device,
        )
        self._vader = VaderEmotionAnalyzer()

    def analyze(self, text: str) -> DocumentEmotion:
        """Analyze sentences with a transformer model and aggregate the result."""
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("Text must contain at least one sentence.")

        sentence_results = [self._analyze_sentence(sentence) for sentence in sentences]
        average_compound = mean(result.compound for result in sentence_results)
        average_intensity = mean(result.intensity for result in sentence_results)
        dominant = self._dominant_emotion(sentence_results)

        return DocumentEmotion(
            emotion=dominant,
            sentiment=self._vader._sentiment_from_compound(average_compound),
            compound=round(average_compound, 4),
            intensity=round(average_intensity, 4),
            confidence=round(mean(result.confidence for result in sentence_results), 4),
            description=VaderEmotionAnalyzer.EMOTION_DESCRIPTIONS.get(dominant, "emotionally balanced"),
            sentences=sentence_results,
        )

    def _analyze_sentence(self, sentence: str) -> EmotionResult:
        raw_predictions = self._classifier(sentence)
        predictions = raw_predictions[0] if raw_predictions and isinstance(raw_predictions[0], list) else raw_predictions
        best = max(predictions, key=lambda item: float(item["score"]))
        raw_label = str(best["label"]).lower()
        confidence = float(best["score"])
        mapped_emotion = self.LABEL_MAP.get(raw_label, self._vader._emotion_from_sentence(sentence, 0.0))
        vader_result = self._vader._analyze_sentence(sentence)

        return EmotionResult(
            sentence=sentence,
            sentiment=vader_result.sentiment,
            emotion=mapped_emotion,
            compound=vader_result.compound,
            intensity=round(min(max(confidence, vader_result.intensity), 1.0), 4),
            confidence=round(confidence, 4),
            description=VaderEmotionAnalyzer.EMOTION_DESCRIPTIONS.get(mapped_emotion, "emotionally balanced"),
        )

    def _dominant_emotion(self, results: list[EmotionResult]) -> str:
        emotion_scores: dict[str, float] = {}
        for result in results:
            emotion_scores[result.emotion] = emotion_scores.get(result.emotion, 0.0) + result.confidence
        return max(emotion_scores.items(), key=lambda item: item[1])[0]


TransformerEmotionAnalyzer = HuggingFaceEmotionAnalyzer
