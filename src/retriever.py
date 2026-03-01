import config
from src.vector_store import ReviewVectorStore


class ReviewRetriever:
    """Recherche les avis pertinents dans la base vectorielle."""

    def __init__(
        self,
        store: ReviewVectorStore | None = None,
        max_results: int = config.MAX_RESULTS,
    ):
        self.store = store or ReviewVectorStore()
        self.max_results = max_results

    def rechercher(self, requete: str, filtre_note: float | None = None) -> list[dict]:
        """
        Retourne les k avis les plus pertinents pour une requête.

        Args:
            requete: La question ou la chaîne de recherche de l'utilisateur.
            filtre_note: Si défini, retourne uniquement les avis avec une note >= cette valeur.

        Returns:
            Liste de dicts avec les clés : text, metadata, distance.
        """
        resultats = self.store.rechercher(requete, n_resultats=self.max_results * 2)
        if filtre_note is not None:
            resultats = [
                r for r in resultats
                if r["metadata"].get("note", 0) >= filtre_note
            ]
        return resultats[: self.max_results]

    # Alias pour compatibilité avec l'interface existante
    def retrieve(self, requete: str, filter_rating: float | None = None) -> list[dict]:
        return self.rechercher(requete, filtre_note=filter_rating)

    def formater_contexte(self, resultats: list[dict]) -> str:
        """Formate les avis récupérés en une chaîne de contexte pour le LLM."""
        parties = []
        for i, r in enumerate(resultats, 1):
            note = r["metadata"].get("note", "?")
            parties.append(f"[Avis {i} - Note {note}/5]\n{r['text']}")
        return "\n\n".join(parties)

    # Alias pour compatibilité avec l'interface existante
    def format_context(self, resultats: list[dict]) -> str:
        return self.formater_contexte(resultats)
