import pytest
from unittest.mock import MagicMock

from src.retriever import ReviewRetriever


FAKE_RESULTS = [
    {"text": "Great for beginners.", "metadata": {"rating": 5.0, "asin": "B001"}, "distance": 0.1},
    {"text": "Hard to use at first.", "metadata": {"rating": 2.0, "asin": "B001"}, "distance": 0.3},
    {"text": "Very intuitive.", "metadata": {"rating": 4.0, "asin": "B001"}, "distance": 0.2},
]


def _make_retriever(results=None):
    store = MagicMock()
    store.query.return_value = results or FAKE_RESULTS
    return ReviewRetriever(store=store, max_results=3)


def test_retrieve_returns_results():
    retriever = _make_retriever()
    results = retriever.retrieve("Is this good for beginners?")
    assert len(results) > 0


def test_retrieve_rating_filter():
    retriever = _make_retriever()
    results = retriever.retrieve("Is this good for beginners?", filter_rating=4.0)
    for r in results:
        assert r["metadata"]["rating"] >= 4.0


def test_retrieve_respects_max_results():
    retriever = _make_retriever()
    results = retriever.retrieve("test")
    assert len(results) <= retriever.max_results


def test_format_context_contains_review_number():
    retriever = _make_retriever()
    results = retriever.retrieve("test")
    context = retriever.format_context(results)
    assert "Review 1" in context


def test_format_context_contains_rating():
    retriever = _make_retriever()
    results = retriever.retrieve("test")
    context = retriever.format_context(results)
    assert "/5" in context
