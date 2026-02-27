from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import config
from src.prompts import REVIEW_QA_PROMPT, FAQ_PROMPT, SUMMARIZE_PROMPT
from src.retriever import ReviewRetriever


class ReviewQAChain:
    """End-to-end RAG chain: retrieve reviews then generate an answer with Llama."""

    MODE_QA = "qa"
    MODE_FAQ = "faq"
    MODE_SUMMARIZE = "summarize"

    PROMPT_MAP = {
        MODE_QA: REVIEW_QA_PROMPT,
        MODE_FAQ: FAQ_PROMPT,
        MODE_SUMMARIZE: SUMMARIZE_PROMPT,
    }

    def __init__(
        self,
        retriever: ReviewRetriever | None = None,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.retriever = retriever or ReviewRetriever()
        self.llm = Ollama(model=model, base_url=base_url)

    def run(
        self,
        question: str,
        mode: str = MODE_QA,
        filter_rating: float | None = None,
    ) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            Dict with keys: answer, sources (list of retrieved reviews).
        """
        if mode not in self.PROMPT_MAP:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(self.PROMPT_MAP)}")

        results = self.retriever.retrieve(question, filter_rating=filter_rating)
        context = self.retriever.format_context(results)

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self.PROMPT_MAP[mode],
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        answer = chain.run(context=context, question=question)

        return {
            "answer": answer.strip(),
            "sources": results,
        }
