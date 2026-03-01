from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import config
from src.prompts import REVIEW_QA_PROMPT, FAQ_PROMPT, SUMMARIZE_PROMPT
from src.retriever import ReviewRetriever


class ReviewQAChain:
    """Pipeline RAG complet : récupère les avis puis génère une réponse avec Llama."""

    MODE_QA = "qa"
    MODE_FAQ = "faq"
    MODE_RESUME = "summarize"

    MAP_PROMPTS = {
        MODE_QA: REVIEW_QA_PROMPT,
        MODE_FAQ: FAQ_PROMPT,
        MODE_RESUME: SUMMARIZE_PROMPT,
    }

    def __init__(
        self,
        retriever: ReviewRetriever | None = None,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.retriever = retriever or ReviewRetriever()
        self.llm = Ollama(model=model, base_url=base_url)

    def executer(
        self,
        question: str,
        mode: str = MODE_QA,
        filtre_note: float | None = None,
    ) -> dict:
        """
        Exécute le pipeline RAG complet.

        Retourne :
            Dict avec les clés : reponse, sources (liste des avis récupérés).
        """
        if mode not in self.MAP_PROMPTS:
            raise ValueError(f"Mode inconnu '{mode}'. Choisir parmi : {list(self.MAP_PROMPTS)}")

        resultats = self.retriever.rechercher(question, filtre_note=filtre_note)
        contexte = self.retriever.formater_contexte(resultats)

        template_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.MAP_PROMPTS[mode],
        )
        chaine = LLMChain(llm=self.llm, prompt=template_prompt)
        reponse = chaine.run(context=contexte, question=question)

        return {
            "answer": reponse.strip(),
            "sources": resultats,
        }

    # Alias pour compatibilité avec l'interface existante
    def run(
        self,
        question: str,
        mode: str = MODE_QA,
        filter_rating: float | None = None,
    ) -> dict:
        return self.executer(question, mode=mode, filtre_note=filter_rating)
