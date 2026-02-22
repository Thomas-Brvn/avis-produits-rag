# Product Review RAG

A Retrieval-Augmented Generation pipeline that processes customer product reviews to answer user questions and handle frequently asked questions (FAQ) with context from real customer feedback. Runs entirely locally using Ollama and open-source models — no API key required.

## What it does

- Ingests and indexes product reviews (CSV or JSON format)
- Embeds reviews locally using sentence-transformers
- Retrieves the most relevant reviews given a user query
- Generates accurate, grounded answers using Llama3 running locally via Ollama

## Use cases

- Answer questions like "Is this product good for beginners?" using real customer experiences
- Summarize the most common complaints or praises about a product
- Build a FAQ bot trained on customer feedback
- Fully private: no data leaves your machine

## Architecture

```
User query
    |
    v
sentence-transformers  -->  ChromaDB (local vector store)
                                  |
                          Top-k reviews retrieved
                                  |
                      Llama3 via Ollama + prompt
                                  |
                          Grounded answer
```

## Tech stack

| Component      | Tool                                      |
|----------------|-------------------------------------------|
| Language        | Python 3.11+                             |
| Orchestration   | LangChain                                |
| Embeddings      | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store    | ChromaDB                                 |
| LLM             | Llama3 via Ollama (local)                |
| Interface       | Streamlit                                |
| Data handling   | pandas                                   |

## Prerequisites

### Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llama3.2
```

Start the Ollama server:

```bash
ollama serve
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Thomas-Brvn/product-review-rag.git
cd product-review-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure settings

Edit `config.py` to adjust the model name, chunk size, or number of retrieved results.

### 4. Add your review data

Place your review file in `data/raw/`. Supported formats:

- CSV with columns: `reviewText`, `summary`, `rating`, `asin`
- JSON array of review objects

### 5. Run the app

```bash
streamlit run app.py
```

## Project structure

```
product-review-rag/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── raw/           # Raw review files (CSV or JSON)
│   └── processed/     # Cleaned and chunked data
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

## Data format

### CSV (Amazon-style)

```csv
asin,reviewText,summary,rating
B001E4KFG0,"Great product, highly recommend","Amazing quality",5
```

### JSON

```json
[
  {
    "asin": "B001E4KFG0",
    "reviewText": "Great product, highly recommend",
    "summary": "Amazing quality",
    "rating": 5
  }
]
```

## License

MIT
