import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


class ReviewPreprocessor:
    """Nettoie et découpe les avis pour l'indexation."""

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

    def nettoyer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=[config.REVIEW_TEXT_COL])
        df[config.REVIEW_TEXT_COL] = df[config.REVIEW_TEXT_COL].apply(self._nettoyer_texte)
        df = df[df[config.REVIEW_TEXT_COL].str.len() > 20]
        if self.min_rating is not None:
            df = df[df[config.REVIEW_RATING_COL] >= self.min_rating]
        df = df.reset_index(drop=True)
        return df

    # Alias pour compatibilité avec l'interface existante
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.nettoyer(df)

    def vers_documents(self, df: pd.DataFrame) -> list[dict]:
        """Convertit les lignes en dicts avec texte et métadonnées."""
        documents = []
        for _, row in df.iterrows():
            texte = self._construire_texte(row)
            metadonnees = {
                "asin": str(row.get(config.REVIEW_PRODUCT_COL, "")),
                "note": float(row.get(config.REVIEW_RATING_COL, 0)),
                "resume": str(row.get(config.REVIEW_SUMMARY_COL, "")),
            }
            morceaux = self.splitter.split_text(texte)
            for morceau in morceaux:
                documents.append({"text": morceau, "metadata": metadonnees})
        return documents

    # Alias pour compatibilité avec l'interface existante
    def to_documents(self, df: pd.DataFrame) -> list[dict]:
        return self.vers_documents(df)

    def _construire_texte(self, row: pd.Series) -> str:
        resume = row.get(config.REVIEW_SUMMARY_COL, "")
        corps = row.get(config.REVIEW_TEXT_COL, "")
        note = row.get(config.REVIEW_RATING_COL, "")
        parties = []
        if resume:
            parties.append(f"Résumé : {resume}")
        if note:
            parties.append(f"Note : {note}/5")
        if corps:
            parties.append(f"Avis : {corps}")
        return "\n".join(parties)

    @staticmethod
    def _nettoyer_texte(texte: str) -> str:
        texte = str(texte)
        texte = re.sub(r"<[^>]+>", " ", texte)
        texte = re.sub(r"http\S+", "", texte)
        texte = re.sub(r"\s+", " ", texte)
        return texte.strip()
