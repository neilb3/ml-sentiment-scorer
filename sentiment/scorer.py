from sentiment.lexicon import POSITIVE_WORDS, NEGATIVE_WORDS


class SentimentScorer:
    """
    Lightweight lexicon-based sentiment scorer. No ML model, no training
    step - counts positive/negative word matches and returns a label
    with a confidence score.
    """

    def score(self, text: str) -> dict:
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
