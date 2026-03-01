# RAG Avis Produits

Un pipeline de Génération Augmentée par Récupération (RAG) qui analyse les avis clients pour répondre aux questions des utilisateurs et générer des FAQ à partir de retours réels. Fonctionne entièrement en local avec Ollama et des modèles open-source — aucune clé API requise.

## Fonctionnalités

- Ingestion et indexation d'avis produits (formats CSV ou JSON)
- Embedding local des avis avec sentence-transformers
- Récupération des avis les plus pertinents selon la requête utilisateur
- Génération de réponses précises et contextualisées avec Llama3 via Ollama en local

## Cas d'usage

- Répondre à des questions comme "Ce produit est-il adapté aux débutants ?" à partir d'expériences clients réelles
- Résumer les plaintes ou compliments les plus fréquents sur un produit
- Construire un bot FAQ alimenté par les retours clients
- Entièrement privé : aucune donnée ne quitte votre machine

## Architecture

```
Requête utilisateur
        |
        v
sentence-transformers  -->  ChromaDB (base vectorielle locale)
                                    |
                          Top-k avis récupérés
                                    |
                      Llama3 via Ollama + prompt
                                    |
                          Réponse contextualisée
```

## Stack technique

| Composant        | Outil                                      |
|------------------|-------------------------------------------|
| Langage          | Python 3.11+                              |
| Orchestration    | LangChain                                 |
| Embeddings       | sentence-transformers (`all-MiniLM-L6-v2`) |
| Base vectorielle | ChromaDB                                  |
| LLM              | Llama3 via Ollama (local)                 |
| Interface        | Streamlit                                 |
| Données          | pandas                                    |

## Prérequis

### Installer Ollama

Téléchargez et installez Ollama depuis [https://ollama.com](https://ollama.com), puis récupérez le modèle :

```bash
ollama pull llama3.2
```

Démarrez le serveur Ollama :

```bash
ollama serve
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Thomas-Brvn/avis-produits-rag.git
cd avis-produits-rag
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer les paramètres

Modifiez `config.py` pour ajuster le nom du modèle, la taille des chunks ou le nombre de résultats récupérés.

### 4. Ajouter vos données d'avis

Placez votre fichier d'avis dans `data/raw/`. Formats supportés :

- CSV avec les colonnes : `reviewText`, `summary`, `rating`, `asin`
- Tableau JSON d'objets avis

### 5. Lancer l'application

```bash
streamlit run app.py
```

## Structure du projet

```
avis-produits-rag/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── raw/           # Fichiers d'avis bruts (CSV ou JSON)
│   └── processed/     # Données nettoyées et découpées
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_chain.py
│   └── prompts.py
├── tests/
│   ├── test_loader.py
│   └── test_retriever.py
├── app.py
└── evaluate.py
```

## Format des données

### CSV (style Amazon)

```csv
asin,reviewText,summary,rating
B001E4KFG0,"Super produit, je recommande","Excellente qualité",5
```

### JSON

```json
[
  {
    "asin": "B001E4KFG0",
    "reviewText": "Super produit, je recommande",
    "summary": "Excellente qualité",
    "rating": 5
  }
]
```

## Licence

MIT
