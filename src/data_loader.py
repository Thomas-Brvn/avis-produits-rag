import json
import pandas as pd
from pathlib import Path

import config


class ReviewLoader:
    """Charge les avis produits depuis des fichiers CSV ou JSON."""

    REQUIRED_COLS = [
        config.REVIEW_TEXT_COL,
        config.REVIEW_RATING_COL,
        config.REVIEW_PRODUCT_COL,
    ]

    def __init__(self, data_path: str = config.RAW_DATA_PATH):
        self.data_path = Path(data_path)

    def load_csv(self, filename: str) -> pd.DataFrame:
        filepath = self.data_path / filename
        df = pd.read_csv(filepath, low_memory=False)
        self._valider(df)
        return df

    def load_json(self, filename: str) -> pd.DataFrame:
        filepath = self.data_path / filename
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        self._valider(df)
        return df

    def load(self, filename: str) -> pd.DataFrame:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return self.load_csv(filename)
        if suffix == ".json":
            return self.load_json(filename)
        raise ValueError(f"Format de fichier non supporté : {suffix}")

    def _valider(self, df: pd.DataFrame) -> None:
        manquantes = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if manquantes:
            raise ValueError(f"Colonnes requises manquantes : {manquantes}")

    def lister_fichiers(self) -> list[str]:
        return [
            f.name
            for f in self.data_path.iterdir()
            if f.suffix.lower() in {".csv", ".json"}
        ]
