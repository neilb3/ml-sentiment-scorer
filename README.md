# ml-sentiment-scorer

A lightweight, dependency-free lexicon-based sentiment scorer.

## Usage

```python
from sentiment.scorer import SentimentScorer

scorer = SentimentScorer()
result = scorer.score("This is a great product")
```

## Running Tests

```bash
pytest tests/ -v
```
