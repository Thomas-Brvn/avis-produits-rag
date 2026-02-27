import streamlit as st
from pathlib import Path

from src.data_loader import ReviewLoader
from src.preprocessor import ReviewPreprocessor
from src.vector_store import ReviewVectorStore
from src.retriever import ReviewRetriever
from src.llm_chain import ReviewQAChain
import config


st.set_page_config(page_title="Product Review RAG", layout="wide")
st.title("Product Review RAG")
st.caption("Ask questions about a product based on real customer reviews — powered by Llama3 running locally.")


@st.cache_resource
def load_chain():
    store = ReviewVectorStore()
    retriever = ReviewRetriever(store=store)
    return ReviewQAChain(retriever=retriever), store


chain, store = load_chain()


with st.sidebar:
    st.header("Index reviews")

    uploaded = st.file_uploader("Upload a review file (CSV or JSON)", type=["csv", "json"])

    use_sample = st.checkbox("Use sample reviews (data/sample_reviews.json)", value=True)

    if st.button("Index"):
        loader = ReviewLoader()
        preprocessor = ReviewPreprocessor()

        if uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            tmp_path = Path(config.RAW_DATA_PATH) / uploaded.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            df = loader.load(uploaded.name)
        elif use_sample:
            import json, pandas as pd
            with open("data/sample_reviews.json", "r") as f:
                df = pd.DataFrame(json.load(f))
        else:
            st.warning("No file selected.")
            st.stop()

        df = preprocessor.clean(df)
        docs = preprocessor.to_documents(df)
        store.reset()
        store.add_documents(docs)
        st.success(f"Indexed {len(docs)} chunks from {len(df)} reviews.")

    st.metric("Chunks in store", store.count())

    st.divider()
    mode = st.radio(
        "Mode",
        [ReviewQAChain.MODE_QA, ReviewQAChain.MODE_FAQ, ReviewQAChain.MODE_SUMMARIZE],
        format_func=lambda x: {"qa": "Q&A", "faq": "FAQ", "summarize": "Summarize"}.get(x, x),
    )
    filter_rating = st.slider("Minimum rating filter (0 = no filter)", 0, 5, 0)


question = st.text_input("Your question", placeholder="Is this product suitable for beginners?")

if st.button("Ask", type="primary") and question:
    if store.count() == 0:
        st.error("No reviews indexed yet. Use the sidebar to index reviews first.")
    else:
        with st.spinner("Thinking..."):
            result = chain.run(
                question=question,
                mode=mode,
                filter_rating=filter_rating if filter_rating > 0 else None,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Source reviews used"):
            for i, src in enumerate(result["sources"], 1):
                rating = src["metadata"].get("rating", "?")
                st.markdown(f"**Review {i}** (rating {rating}/5)")
                st.write(src["text"])
                st.divider()
