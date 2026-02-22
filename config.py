OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

VECTOR_STORE_PATH = "data/vector_store"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

MAX_RESULTS = 5
MIN_RATING_FILTER = None  # set to 1-5 to filter by minimum rating

REVIEW_TEXT_COL = "reviewText"
REVIEW_SUMMARY_COL = "summary"
REVIEW_RATING_COL = "rating"
REVIEW_PRODUCT_COL = "asin"
