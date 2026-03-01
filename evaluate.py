"""
Script d'évaluation du pipeline RAG.

Mesure la pertinence de la récupération et la qualité des réponses sur un petit jeu de données annoté.
Lancer avec :  python evaluate.py
"""

import json
from dataclasses import dataclass, field

from src.vector_store import ReviewVectorStore
from src.retriever import ReviewRetriever
from src.llm_chain import ReviewQAChain
from src.data_loader import ReviewLoader
from src.preprocessor import ReviewPreprocessor


# ---------------------------------------------------------------------------
# Jeu d'évaluation annoté
# ---------------------------------------------------------------------------

JEU_EVALUATION = [
    {
        "question": "Ce produit convient-il aux débutants ?",
        "mots_cles_attendus": ["débutant", "facile", "simple", "intuitif"],
    },
    {
        "question": "Ce produit est-il solide et durable ?",
        "mots_cles_attendus": ["durable", "années", "fonctionne encore", "solide"],
    },
    {
        "question": "Ce produit est-il bruyant ?",
        "mots_cles_attendus": ["bruyant", "bruit", "fort", "silencieux"],
    },
    {
        "question": "Est-il facile à nettoyer ?",
        "mots_cles_attendus": ["nettoyer", "lave-vaisselle", "facile à nettoyer"],
    },
]


@dataclass
class ResultatEval:
    question: str
    reponse: str
    nb_recuperes: int
    mots_cles_trouves: list[str] = field(default_factory=list)
    score_mots_cles: float = 0.0


def rappel_mots_cles(reponse: str, mots_cles: list[str]) -> tuple[list[str], float]:
    reponse_lower = reponse.lower()
    trouves = [m for m in mots_cles if m.lower() in reponse_lower]
    score = len(trouves) / len(mots_cles) if mots_cles else 0.0
    return trouves, score


def lancer_evaluation() -> None:
    # Indexation des avis exemples
    loader = ReviewLoader(data_path="data")
    preprocessor = ReviewPreprocessor()
    df = loader.load_json("sample_reviews.json")
    df = preprocessor.nettoyer(df)
    docs = preprocessor.vers_documents(df)

    store = ReviewVectorStore()
    store.reinitialiser()
    store.ajouter_documents(docs)
    print(f"{len(docs)} morceaux indexés.\n")

    retriever = ReviewRetriever(store=store)
    chaine = ReviewQAChain(retriever=retriever)

    resultats: list[ResultatEval] = []

    for item in JEU_EVALUATION:
        question = item["question"]
        mots_cles = item["mots_cles_attendus"]
        print(f"Q : {question}")

        sortie = chaine.executer(question=question, mode="qa")
        reponse = sortie["answer"]
        sources = sortie["sources"]

        trouves, score = rappel_mots_cles(reponse, mots_cles)
        resultat = ResultatEval(
            question=question,
            reponse=reponse,
            nb_recuperes=len(sources),
            mots_cles_trouves=trouves,
            score_mots_cles=score,
        )
        resultats.append(resultat)

        print(f"  Réponse : {reponse[:120]}...")
        print(f"  Rappel mots-clés : {score:.0%} ({trouves})")
        print(f"  Récupérés : {resultat.nb_recuperes} morceaux\n")

    score_moyen = sum(r.score_mots_cles for r in resultats) / len(resultats)
    print(f"Rappel moyen des mots-clés : {score_moyen:.0%}")

    with open("eval_resultats.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "question": r.question,
                    "reponse": r.reponse,
                    "mots_cles_trouves": r.mots_cles_trouves,
                    "score_mots_cles": r.score_mots_cles,
                    "nb_recuperes": r.nb_recuperes,
                }
                for r in resultats
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Résultats sauvegardés dans eval_resultats.json")


if __name__ == "__main__":
    lancer_evaluation()
