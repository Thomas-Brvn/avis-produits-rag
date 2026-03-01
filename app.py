import streamlit as st
from pathlib import Path

from src.data_loader import ReviewLoader
from src.preprocessor import ReviewPreprocessor
from src.vector_store import ReviewVectorStore
from src.retriever import ReviewRetriever
from src.llm_chain import ReviewQAChain
import config


st.set_page_config(page_title="RAG Avis Produits", layout="wide")
st.title("RAG Avis Produits")
st.caption("Posez des questions sur un produit à partir de vrais avis clients — propulsé par Llama3 en local.")


@st.cache_resource
def charger_chaine():
    store = ReviewVectorStore()
    retriever = ReviewRetriever(store=store)
    return ReviewQAChain(retriever=retriever), store


chaine, store = charger_chaine()


with st.sidebar:
    st.header("Indexer les avis")

    fichier = st.file_uploader("Importer un fichier d'avis (CSV ou JSON)", type=["csv", "json"])

    utiliser_exemple = st.checkbox("Utiliser les avis exemples (data/sample_reviews.json)", value=True)

    if st.button("Indexer"):
        loader = ReviewLoader()
        preprocessor = ReviewPreprocessor()

        if fichier is not None:
            suffix = Path(fichier.name).suffix.lower()
            tmp_path = Path(config.RAW_DATA_PATH) / fichier.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(fichier.read())
            df = loader.load(fichier.name)
        elif utiliser_exemple:
            import json, pandas as pd
            with open("data/sample_reviews.json", "r") as f:
                df = pd.DataFrame(json.load(f))
        else:
            st.warning("Aucun fichier sélectionné.")
            st.stop()

        df = preprocessor.nettoyer(df)
        docs = preprocessor.vers_documents(df)
        store.reinitialiser()
        store.ajouter_documents(docs)
        st.success(f"{len(docs)} morceaux indexés depuis {len(df)} avis.")

    st.metric("Morceaux dans la base", store.compter())

    st.divider()
    mode = st.radio(
        "Mode",
        [ReviewQAChain.MODE_QA, ReviewQAChain.MODE_FAQ, ReviewQAChain.MODE_RESUME],
        format_func=lambda x: {"qa": "Q&R", "faq": "FAQ", "summarize": "Résumé"}.get(x, x),
    )
    filtre_note = st.slider("Filtre note minimale (0 = pas de filtre)", 0, 5, 0)


question = st.text_input("Votre question", placeholder="Ce produit convient-il aux débutants ?")

if st.button("Demander", type="primary") and question:
    if store.compter() == 0:
        st.error("Aucun avis indexé. Utilisez le panneau latéral pour indexer des avis d'abord.")
    else:
        with st.spinner("Analyse en cours..."):
            resultat = chaine.executer(
                question=question,
                mode=mode,
                filtre_note=filtre_note if filtre_note > 0 else None,
            )

        st.subheader("Réponse")
        st.write(resultat["answer"])

        with st.expander("Avis sources utilisés"):
            for i, src in enumerate(resultat["sources"], 1):
                note = src["metadata"].get("note", "?")
                st.markdown(f"**Avis {i}** (note {note}/5)")
                st.write(src["text"])
                st.divider()
