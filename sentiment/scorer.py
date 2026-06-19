from sentiment.lexicon import POSITIVE_WORDS, NEGATIVE_WORDS

MAX_TEXT_LENGTH = 5000


class SentimentScorer:
    """
    Lightweight lexicon-based sentiment scorer. No ML model, no training
    step - counts positive/negative word matches and returns a label
    with a confidence score.
    """

    def score(self, text: str) -> dict:
        if text is None or not isinstance(text, str):
            raise TypeError("text must be a string, got: {}".format(type(text).__name__))

        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                "text must not exceed {} characters, got: {}".format(MAX_TEXT_LENGTH, len(text))
            )

        words = text.lower().split()
        positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = positive_count + negative_count

        if total == 0:
            sentiment = "neutral"
            confidence = 0.0
        elif positive_count > negative_count:
            sentiment = "positive"
            confidence = (positive_count - negative_count) / total
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = (negative_count - positive_count) / total
        else:
            sentiment = "neutral"
            confidence = 0.0

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_words": positive_count,
            "negative_words": negative_count,
        }