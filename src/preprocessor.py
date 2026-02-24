import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


class ReviewPreprocessor:
    """Clean and chunk reviews for indexing."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        min_rating: int | None = config.MIN_RATING_FILTER,
    ):
        self.min_rating = min_rating
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=[config.REVIEW_TEXT_COL])
        df[config.REVIEW_TEXT_COL] = df[config.REVIEW_TEXT_COL].apply(self._clean_text)
        df = df[df[config.REVIEW_TEXT_COL].str.len() > 20]
        if self.min_rating is not None:
            df = df[df[config.REVIEW_RATING_COL] >= self.min_rating]
        df = df.reset_index(drop=True)
        return df

    def to_documents(self, df: pd.DataFrame) -> list[dict]:
        """Convert rows to document dicts with text + metadata."""
        documents = []
        for _, row in df.iterrows():
            text = self._build_text(row)
            metadata = {
                "asin": str(row.get(config.REVIEW_PRODUCT_COL, "")),
                "rating": float(row.get(config.REVIEW_RATING_COL, 0)),
                "summary": str(row.get(config.REVIEW_SUMMARY_COL, "")),
            }
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                documents.append({"text": chunk, "metadata": metadata})
        return documents

    def _build_text(self, row: pd.Series) -> str:
        summary = row.get(config.REVIEW_SUMMARY_COL, "")
        body = row.get(config.REVIEW_TEXT_COL, "")
        rating = row.get(config.REVIEW_RATING_COL, "")
        parts = []
        if summary:
            parts.append(f"Summary: {summary}")
        if rating:
            parts.append(f"Rating: {rating}/5")
        if body:
            parts.append(f"Review: {body}")
        return "\n".join(parts)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
