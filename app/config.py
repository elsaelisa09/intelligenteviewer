__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "document_chunks")

# App configuration
MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB

# Directory configuration
TEMP_DIR = Path(os.getenv('TEMP_DIR', Path.home() / '.document_qa' / 'faiss_indices'))
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Text splitting configuration
TEXT_SPLITTER_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 50,
    'separators': ["\n\n", "\n", ". ", " ", ""]
}

# Embeddings configuration
EMBEDDINGS_MODEL = {
    'name': "sentence-transformers/all-MiniLM-L6-v2",
    'device': 'cpu'
}