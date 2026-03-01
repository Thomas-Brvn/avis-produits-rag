from sentence_transformers import SentenceTransformer
import numpy as np

import config


class LocalEmbedder:
    """Encapsule sentence-transformers pour l'embedding local."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encoder(self, textes: list[str]) -> list[list[float]]:
        vecteurs = self.model.encode(textes, show_progress_bar=False, convert_to_numpy=True)
        return vecteurs.tolist()

    # Alias pour compatibilité avec l'interface existante
    def embed(self, textes: list[str]) -> list[list[float]]:
        return self.encoder(textes)

    def encoder_requete(self, texte: str) -> list[float]:
        vecteur = self.model.encode([texte], convert_to_numpy=True)
        return vecteur[0].tolist()

    # Alias pour compatibilité avec l'interface existante
    def embed_query(self, texte: str) -> list[float]:
        return self.encoder_requete(texte)

    def similarite(self, vec_a: list[float], vec_b: list[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
