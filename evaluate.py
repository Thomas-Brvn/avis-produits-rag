"""
Evaluation script for the RAG pipeline.

Measures retrieval relevance and answer quality on a small labeled dataset.
Run with:  python evaluate.py
"""

import json
from dataclasses import dataclass, field

from src.vector_store import ReviewVectorStore
from src.retriever import ReviewRetriever
from src.llm_chain import ReviewQAChain
from src.data_loader import ReviewLoader
from src.preprocessor import ReviewPreprocessor


# ---------------------------------------------------------------------------
# Labeled evaluation set
# ---------------------------------------------------------------------------

EVAL_SET = [
    {
        "question": "Is this product suitable for beginners?",
        "expected_keywords": ["beginner", "easy", "simple", "intuitive"],
    },
    {
        "question": "How durable is this product?",
        "expected_keywords": ["durable", "years", "still works", "solid"],
    },
    {
        "question": "Is the product noisy?",
        "expected_keywords": ["noisy", "noise", "loud", "quiet"],
    },
    {
        "question": "How easy is it to clean?",
        "expected_keywords": ["clean", "dishwasher", "easy to clean"],
    },
]


@dataclass
class EvalResult:
    question: str
    answer: str
    retrieved_count: int
    keyword_hits: list[str] = field(default_factory=list)
    keyword_score: float = 0.0


def keyword_recall(answer: str, keywords: list[str]) -> tuple[list[str], float]:
    answer_lower = answer.lower()
    hits = [k for k in keywords if k.lower() in answer_lower]
    score = len(hits) / len(keywords) if keywords else 0.0
    return hits, score


def run_evaluation() -> None:
    # Index sample reviews
    loader = ReviewLoader(data_path="data")
    preprocessor = ReviewPreprocessor()
    df = loader.load_json("sample_reviews.json")
    df = preprocessor.clean(df)
    docs = preprocessor.to_documents(df)

    store = ReviewVectorStore()
    store.reset()
    store.add_documents(docs)
    print(f"Indexed {len(docs)} chunks.\n")

    retriever = ReviewRetriever(store=store)
    chain = ReviewQAChain(retriever=retriever)

    results: list[EvalResult] = []

    for item in EVAL_SET:
        question = item["question"]
        keywords = item["expected_keywords"]
        print(f"Q: {question}")

        output = chain.run(question=question, mode="qa")
        answer = output["answer"]
        sources = output["sources"]

        hits, score = keyword_recall(answer, keywords)
        result = EvalResult(
            question=question,
            answer=answer,
            retrieved_count=len(sources),
            keyword_hits=hits,
            keyword_score=score,
        )
        results.append(result)

        print(f"  Answer: {answer[:120]}...")
        print(f"  Keyword recall: {score:.0%} ({hits})")
        print(f"  Retrieved: {result.retrieved_count} chunks\n")

    avg_score = sum(r.keyword_score for r in results) / len(results)
    print(f"Average keyword recall: {avg_score:.0%}")

    with open("eval_results.json", "w") as f:
        json.dump(
            [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "keyword_hits": r.keyword_hits,
                    "keyword_score": r.keyword_score,
                    "retrieved_count": r.retrieved_count,
                }
                for r in results
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Results saved to eval_results.json")


if __name__ == "__main__":
    run_evaluation()
