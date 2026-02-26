import config
from src.vector_store import ReviewVectorStore


class ReviewRetriever:
    """Retrieve relevant reviews from the vector store."""

    def __init__(
        self,
        store: ReviewVectorStore | None = None,
        max_results: int = config.MAX_RESULTS,
    ):
        self.store = store or ReviewVectorStore()
        self.max_results = max_results

    def retrieve(self, query: str, filter_rating: float | None = None) -> list[dict]:
        """
        Retrieve top-k reviews for a query.

        Args:
            query: The user question or search string.
            filter_rating: If set, only return reviews with rating >= this value.

        Returns:
            List of dicts with keys: text, metadata, distance.
        """
        results = self.store.query(query, n_results=self.max_results * 2)
        if filter_rating is not None:
            results = [
                r for r in results
                if r["metadata"].get("rating", 0) >= filter_rating
            ]
        return results[: self.max_results]

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved reviews into a context string for the LLM."""
        parts = []
        for i, r in enumerate(results, 1):
            rating = r["metadata"].get("rating", "?")
            parts.append(f"[Review {i} - Rating {rating}/5]\n{r['text']}")
        return "\n\n".join(parts)
