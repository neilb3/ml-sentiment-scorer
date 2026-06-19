import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.scorer import SentimentScorer

scorer = SentimentScorer()


def test_positive_sentiment():
    result = scorer.score("This is a great and wonderful product")
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0


def test_negative_sentiment():
    result = scorer.score("This is terrible and awful")
    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0


def test_neutral_sentiment():
    result = scorer.score("The cat sat on the mat")
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0


def test_mixed_sentiment_leans_correctly():
    result = scorer.score("good good bad")
    assert result["sentiment"] == "positive"


def test_score_raises_type_error_for_none():
    with pytest.raises(TypeError, match="text must be a string"):
        scorer.score(None)


def test_score_raises_type_error_for_non_string():
    with pytest.raises(TypeError, match="text must be a string"):
        scorer.score(12345)


def test_score_raises_value_error_for_empty_string():
    with pytest.raises(ValueError, match="text must not be empty or whitespace-only"):
        scorer.score("")


def test_score_raises_value_error_for_whitespace_only_string():
    with pytest.raises(ValueError, match="text must not be empty or whitespace-only"):
        scorer.score("   ")


def test_score_raises_value_error_for_text_exceeding_max_length():
    oversized_text = "a" * 5001
    with pytest.raises(ValueError, match="text must not exceed 5000 characters"):
        scorer.score(oversized_text)


def test_score_accepts_text_at_max_length():
    max_length_text = "a" * 5000
    result = scorer.score(max_length_text)
    assert result["sentiment"] == "neutral"