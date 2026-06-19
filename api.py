from sentiment.scorer import SentimentScorer

scorer = SentimentScorer()


def analyze(text: str) -> dict:
    """Public entry point for sentiment analysis."""
    return scorer.score(text)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "This is a great example"
    print(analyze(text))
