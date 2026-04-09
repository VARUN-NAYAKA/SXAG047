"""
ScholAR - Search Agent.
Handles query expansion, multi-API search, deduplication, relevance scoring, and embeddings.
"""

import json
import os
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from sentence_transformers import SentenceTransformer

from core.llm_utils import call_gemini

from core.config import (
    GEMINI_MODEL, LLM_TEMPERATURE,
    WEIGHT_SEMANTIC, WEIGHT_CITATIONS, WEIGHT_RECENCY, WEIGHT_VENUE,
    PRESTIGIOUS_VENUES, EMBEDDING_MODEL, RELEVANCE_THRESHOLD, TOP_K_PAPERS,
)
from core.models import Paper, SearchQuery
from core.prompts import QUERY_EXPANSION_PROMPT
from apis import semantic_scholar, arxiv_client, openalex, crossref
from rich.console import Console

console = Console()

# LLM calls go through core.llm_utils.call_gemini (retry + fallback)

# Lazy-loaded embedding model
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        console.print("[yellow]Loading embedding model...[/yellow]")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        console.print("[green]Embedding model loaded![/green]")
    return _embedding_model


# ──────────────────────────────────────────────
# Query Expansion
# ──────────────────────────────────────────────
def expand_queries(topic: str, num_queries: int = 12) -> list[SearchQuery]:
    """
    Use Gemini LLM to expand a user topic into diverse search queries.

    Args:
        topic: The user's research topic
        num_queries: How many queries to generate

    Returns:
        List of SearchQuery objects
    """
    console.print(f"\n[bold cyan]🔍 Expanding queries for:[/bold cyan] {topic}")

    prompt = QUERY_EXPANSION_PROMPT.format(topic=topic, num_queries=num_queries)

    try:
        text = call_gemini(prompt, max_tokens=2048, temperature=LLM_TEMPERATURE)
        # Clean up potential markdown wrapping
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        queries_data = json.loads(text)
        queries = [
            SearchQuery(query_text=q["query"], strategy=q.get("strategy", "direct"))
            for q in queries_data
        ]

        console.print(f"[green]Generated {len(queries)} search queries[/green]")
        for q in queries:
            console.print(f"  [{q.strategy}] {q.query_text}")

        return queries

    except Exception as e:
        console.print(f"[red]Query expansion failed: {e}[/red]")
        # Fallback: use the topic directly
        return [
            SearchQuery(query_text=topic, strategy="direct"),
            SearchQuery(query_text=f"{topic} survey", strategy="broader"),
            SearchQuery(query_text=f"{topic} recent advances", strategy="related"),
        ]


# ──────────────────────────────────────────────
# Multi-API Search
# ──────────────────────────────────────────────
def search_all_apis(
    query: str,
    limit_per_api: int = 20,
) -> list[Paper]:
    """
    Search all APIs in parallel for a single query.

    Args:
        query: Search query string
        limit_per_api: Max results per API

    Returns:
        Combined list of papers from all APIs
    """
    all_papers = []

    def _search_semantic(q, lim):
        return semantic_scholar.search_papers(q, limit=lim)

    def _search_arxiv(q, lim):
        return arxiv_client.search_papers(q, limit=lim)

    def _search_openalex(q, lim):
        return openalex.search_papers(q, limit=lim)

    def _search_crossref(q, lim):
        return crossref.search_papers(q, limit=lim)

    search_funcs = [
        ("Semantic Scholar", _search_semantic),
        ("ArXiv", _search_arxiv),
        ("OpenAlex", _search_openalex),
        ("CrossRef", _search_crossref),
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, func in search_funcs:
            future = executor.submit(func, query, limit_per_api)
            futures[future] = name

        for future in as_completed(futures):
            api_name = futures[future]
            try:
                papers = future.result()
                all_papers.extend(papers)
            except Exception as e:
                console.print(f"[red]{api_name} search failed: {e}[/red]")

    return all_papers


def search_with_queries(
    queries: list[SearchQuery],
    limit_per_api: int = 15,
) -> list[Paper]:
    """
    Execute all search queries across all APIs.

    Args:
        queries: List of SearchQuery objects
        limit_per_api: Results per API per query

    Returns:
        Combined (not yet deduplicated) list of papers
    """
    all_papers = []

    for i, query in enumerate(queries):
        console.print(f"\n[bold]Query {i+1}/{len(queries)}:[/bold] {query.query_text}")
        papers = search_all_apis(query.query_text, limit_per_api=limit_per_api)
        all_papers.extend(papers)
        console.print(f"  → {len(papers)} papers from this query")

    console.print(f"\n[bold green]Total papers before dedup: {len(all_papers)}[/bold green]")
    return all_papers


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────
def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """
    Remove duplicate papers using DOI matching and title similarity.

    Args:
        papers: List of papers (potentially with duplicates)

    Returns:
        Deduplicated list of papers
    """
    seen_dois: set[str] = set()
    seen_title_hashes: set[str] = set()
    unique_papers: list[Paper] = []

    for paper in papers:
        # Check DOI
        if paper.doi:
            doi_lower = paper.doi.lower().strip()
            if doi_lower in seen_dois:
                continue
            seen_dois.add(doi_lower)

        # Check title hash (normalize whitespace + lowercase)
        title_normalized = " ".join(paper.title.lower().split())
        title_hash = hashlib.md5(title_normalized.encode()).hexdigest()

        if title_hash in seen_title_hashes:
            continue
        seen_title_hashes.add(title_hash)

        unique_papers.append(paper)

    removed = len(papers) - len(unique_papers)
    console.print(f"[cyan]Dedup:[/cyan] Removed {removed} duplicates → {len(unique_papers)} unique papers")
    return unique_papers


# ──────────────────────────────────────────────
# Relevance Scoring
# ──────────────────────────────────────────────
def score_papers(
    papers: list[Paper],
    topic: str,
    top_k: int = None,
) -> list[Paper]:
    """
    Score and rank papers by relevance to the topic.

    Scoring formula:
        score = α·semantic + β·citations + γ·recency + δ·venue

    Args:
        papers: List of papers to score
        topic: Original user topic
        top_k: Number of top papers to keep (default from config)

    Returns:
        Scored and sorted list of papers
    """
    if top_k is None:
        top_k = TOP_K_PAPERS

    if not papers:
        return []

    model = _get_embedding_model()

    # Embed the topic
    topic_embedding = model.encode(topic, normalize_embeddings=True)

    # Embed all paper abstracts/titles
    texts = [
        f"{p.title}. {p.abstract[:500]}" if p.abstract else p.title
        for p in papers
    ]
    paper_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Compute semantic similarity
    semantic_scores = np.dot(paper_embeddings, topic_embedding)

    # Normalize citation counts (log scale)
    max_citations = max((p.citation_count for p in papers), default=1)
    if max_citations == 0:
        max_citations = 1

    import datetime
    current_year = datetime.datetime.now().year

    for i, paper in enumerate(papers):
        # Semantic similarity (already 0-1 from cosine)
        sem_score = float(max(semantic_scores[i], 0))

        # Citation score (log-normalized)
        cite_score = np.log1p(paper.citation_count) / np.log1p(max_citations)

        # Recency score (newer = higher)
        if paper.year:
            age = max(current_year - paper.year, 0)
            recency_score = max(1.0 - (age / 20.0), 0)  # Papers older than 20 years get 0
        else:
            recency_score = 0.3  # Unknown year

        # Venue prestige score
        venue_score = 0.0
        if paper.venue:
            venue_lower = paper.venue.lower()
            for prestigious in PRESTIGIOUS_VENUES:
                if prestigious in venue_lower:
                    venue_score = 1.0
                    break
            else:
                venue_score = 0.3  # Known venue but not prestigious

        # Combined score
        paper.relevance_score = (
            WEIGHT_SEMANTIC * sem_score
            + WEIGHT_CITATIONS * cite_score
            + WEIGHT_RECENCY * recency_score
            + WEIGHT_VENUE * venue_score
        )

        # Store embedding for later use
        paper.embedding = paper_embeddings[i].tolist()

    # Sort by relevance score
    papers.sort(key=lambda p: p.relevance_score, reverse=True)

    # Filter by threshold and top_k
    filtered = [p for p in papers if p.relevance_score >= RELEVANCE_THRESHOLD][:top_k]

    console.print(
        f"[cyan]Scoring:[/cyan] {len(filtered)} papers above threshold "
        f"(top score: {filtered[0].relevance_score:.3f}, "
        f"min: {filtered[-1].relevance_score:.3f})"
    )

    return filtered


# ──────────────────────────────────────────────
# Embedding Generation
# ──────────────────────────────────────────────
def generate_embeddings(papers: list[Paper]) -> dict[str, list[float]]:
    """
    Generate/retrieve embeddings for all papers.

    Returns:
        Dict mapping paper_id -> embedding vector
    """
    model = _get_embedding_model()
    embeddings = {}

    papers_needing_embeddings = [p for p in papers if p.embedding is None]

    if papers_needing_embeddings:
        texts = [
            f"{p.title}. {p.abstract[:500]}" if p.abstract else p.title
            for p in papers_needing_embeddings
        ]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        for paper, vec in zip(papers_needing_embeddings, vectors):
            paper.embedding = vec.tolist()

    for paper in papers:
        if paper.embedding:
            embeddings[paper.paper_id] = paper.embedding

    console.print(f"[cyan]Embeddings:[/cyan] {len(embeddings)} papers embedded")
    return embeddings
