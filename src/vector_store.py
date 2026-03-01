import uuid
from pathlib import Path

import chromadb

import config
from src.embeddings import LocalEmbedder


class ReviewVectorStore:
    """Base vectorielle ChromaDB pour les avis produits."""

    NOM_COLLECTION = "avis_produits"

    def __init__(
        self,
        persist_path: str = config.VECTOR_STORE_PATH,
        embedder: LocalEmbedder | None = None,
    ):
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or LocalEmbedder()
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=self.NOM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def ajouter_documents(self, documents: list[dict]) -> None:
        """Indexe une liste de documents (texte + métadonnées)."""
        if not documents:
            return
        textes = [d["text"] for d in documents]
        metadonnees = [d["metadata"] for d in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedder.encoder(textes)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=textes,
            metadatas=metadonnees,
        )

    # Alias pour compatibilité avec l'interface existante
    def add_documents(self, documents: list[dict]) -> None:
        return self.ajouter_documents(documents)

    def rechercher(self, texte_requete: str, n_resultats: int = config.MAX_RESULTS) -> list[dict]:
        """Retourne les k documents les plus proches pour une requête."""
        vecteur_requete = self.embedder.encoder_requete(texte_requete)
        resultats = self.collection.query(
            query_embeddings=[vecteur_requete],
            n_results=n_resultats,
            include=["documents", "metadatas", "distances"],
        )
        sortie = []
        for texte, meta, dist in zip(
            resultats["documents"][0],
            resultats["metadatas"][0],
            resultats["distances"][0],
        ):
            sortie.append({"text": texte, "metadata": meta, "distance": dist})
        return sortie

    # Alias pour compatibilité avec l'interface existante
    def query(self, texte_requete: str, n_results: int = config.MAX_RESULTS) -> list[dict]:
        return self.rechercher(texte_requete, n_results)

    def compter(self) -> int:
        return self.collection.count()

    # Alias pour compatibilité avec l'interface existante
    def count(self) -> int:
        return self.compter()

    def reinitialiser(self) -> None:
        self.client.delete_collection(self.NOM_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=self.NOM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # Alias pour compatibilité avec l'interface existante
    def reset(self) -> None:
        return self.reinitialiser()
