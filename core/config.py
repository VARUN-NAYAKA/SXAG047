"""
ScholAR - Autonomous Research Intelligence Agent
Core configuration and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"
REPORT_TEMPLATE_DIR = BASE_DIR / "report" / "templates"

# Create directories
for d in [DATA_DIR, CACHE_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Agent Settings
# ──────────────────────────────────────────────
MAX_ITERATIONS = 3                  # Max search-analyze loops
MAX_PAPERS = 200                    # Hard cap on total papers
MIN_PAPERS_FOR_ANALYSIS = 30       # Minimum papers before analyzing
MIN_CLUSTERS_REQUIRED = 3          # Minimum clusters to proceed to report
RELEVANCE_THRESHOLD = 0.3          # Min relevance score to keep a paper
TOP_K_PAPERS = 100                 # Max papers to keep after scoring

# ──────────────────────────────────────────────
# Relevance Scoring Weights
# ──────────────────────────────────────────────
WEIGHT_SEMANTIC = 0.40             # Cosine similarity weight
WEIGHT_CITATIONS = 0.25           # Citation impact weight
WEIGHT_RECENCY = 0.20             # Publication year weight
WEIGHT_VENUE = 0.15               # Venue prestige weight

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ──────────────────────────────────────────────
# Graph Settings
# ──────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.7        # Cosine similarity for graph edges
MAX_GRAPH_NODES = 500

# ──────────────────────────────────────────────
# LLM Settings
# ──────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 8192

# ──────────────────────────────────────────────
# API Rate Limits (requests per second)
# ──────────────────────────────────────────────
SEMANTIC_SCHOLAR_RPS = 1           # ~100 per 5 min
ARXIV_RPS = 3
CROSSREF_RPS = 10
OPENALEX_RPS = 10

# ──────────────────────────────────────────────
# Prestigious Venues (for scoring)
# ──────────────────────────────────────────────
PRESTIGIOUS_VENUES = {
    # Conferences
    "neurips", "nips", "icml", "iclr", "aaai", "ijcai", "cvpr", "iccv",
    "eccv", "acl", "emnlp", "naacl", "sigir", "www", "kdd", "icde",
    "vldb", "sigmod", "chi", "uist", "cscw",
    # Journals
    "nature", "science", "cell", "lancet", "jama", "bmj",
    "ieee transactions", "acm computing surveys", "artificial intelligence",
    "journal of machine learning research", "jmlr",
}
