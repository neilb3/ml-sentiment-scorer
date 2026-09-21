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


def test_none_input_raises_type_error():
    with pytest.raises(TypeError, match="text must be a string"):
        scorer.score(None)


def test_integer_input_raises_type_error():
    with pytest.raises(TypeError, match="text must be a string"):
        scorer.score(123)


def test_list_input_raises_type_error():
    with pytest.raises(TypeError, match="text must be a string"):
        scorer.score(["great", "wonderful"])


def test_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="text must not be empty"):
        scorer.score("")


def test_whitespace_only_string_raises_value_error():
    with pytest.raises(ValueError, match="text must not be empty"):
        scorer.score("   ")


def test_newline_only_string_raises_value_error():
    with pytest.raises(ValueError, match="text must not be empty"):
        scorer.score("\n\t")