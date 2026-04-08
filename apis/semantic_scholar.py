"""
ScholAR - Semantic Scholar API wrapper.
Docs: https://api.semanticscholar.org/api-docs/graph
"""

import time
import requests
from typing import Optional
from core.config import SEMANTIC_SCHOLAR_API_KEY, SEMANTIC_SCHOLAR_RPS
from core.models import Paper, Author
from rich.console import Console

console = Console()

BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_URL = f"{BASE_URL}/paper/search"
PAPER_URL = f"{BASE_URL}/paper"

HEADERS = {}
if SEMANTIC_SCHOLAR_API_KEY:
    HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY


def _rate_limit():
    """Simple rate limiter."""
    time.sleep(1.0 / SEMANTIC_SCHOLAR_RPS)


def search_papers(
    query: str,
    limit: int = 50,
    year_range: Optional[str] = None,
    fields_of_study: Optional[list[str]] = None,
) -> list[Paper]:
    """
    Search for papers on Semantic Scholar.

    Args:
        query: Search query string
        limit: Maximum number of results (max 100)
        year_range: e.g., "2020-2024" or "2020-"
        fields_of_study: e.g., ["Computer Science", "Medicine"]

    Returns:
        List of Paper objects
    """
    papers = []
    offset = 0
    per_page = min(limit, 100)

    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,tldr,fieldsOfStudy,"
        "citations.paperId,references.paperId"
    )

    while len(papers) < limit:
        params = {
            "query": query,
            "offset": offset,
            "limit": per_page,
            "fields": fields,
        }

        if year_range:
            params["year"] = year_range
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        _rate_limit()

        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            console.print(f"[red]Semantic Scholar API error: {e}[/red]")
            break

        results = data.get("data", [])
        if not results:
            break

        for item in results:
            paper = _parse_paper(item)
            if paper:
                papers.append(paper)

        total_available = data.get("total", 0)
        offset += per_page

        if offset >= total_available:
            break

    console.print(f"[green]Semantic Scholar:[/green] Found {len(papers)} papers for '{query}'")
    return papers[:limit]


def get_paper_details(paper_id: str) -> Optional[Paper]:
    """Fetch detailed info for a specific paper by its Semantic Scholar ID."""
    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,tldr,fieldsOfStudy,"
        "citations.paperId,references.paperId"
    )

    _rate_limit()

    try:
        resp = requests.get(
            f"{PAPER_URL}/{paper_id}",
            params={"fields": fields},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return _parse_paper(resp.json())
    except requests.RequestException as e:
        console.print(f"[red]Semantic Scholar detail error: {e}[/red]")
        return None


def _parse_paper(data: dict) -> Optional[Paper]:
    """Parse API response into a Paper object."""
    if not data.get("paperId") or not data.get("title"):
        return None

    authors = []
    for a in data.get("authors", []) or []:
        authors.append(
            Author(
                author_id=a.get("authorId", ""),
                name=a.get("name", "Unknown"),
            )
        )

    # Extract DOI from externalIds
    external_ids = data.get("externalIds") or {}
    doi = external_ids.get("DOI", None)

    # Extract references and citations as ID lists
    refs = [r["paperId"] for r in (data.get("references") or []) if r.get("paperId")]
    cites = [c["paperId"] for c in (data.get("citations") or []) if c.get("paperId")]

    # TLDR
    tldr_data = data.get("tldr")
    tldr = tldr_data.get("text", "") if isinstance(tldr_data, dict) else None

    return Paper(
        paper_id=data["paperId"],
        title=data["title"],
        abstract=data.get("abstract") or "",
        authors=authors,
        year=data.get("year"),
        doi=doi,
        url=data.get("url", ""),
        venue=data.get("venue", ""),
        citation_count=data.get("citationCount", 0) or 0,
        reference_count=data.get("referenceCount", 0) or 0,
        references=refs,
        citations=cites,
        source_api="semantic_scholar",
        tldr=tldr,
        fields_of_study=data.get("fieldsOfStudy") or [],
    )


def search_author_papers(author_name: str, limit: int = 50) -> list[Paper]:
    """
    Search for an author by name and return their papers.
    Strategy: Try Author Search API first, then fall back to paper search with author name.
    """
    papers = _try_author_api(author_name, limit)
    if papers:
        return papers

    # Fallback: search papers with author name as query
    console.print(f"[yellow]Author API failed, falling back to paper search for '{author_name}'[/yellow]")
    return _author_paper_search_fallback(author_name, limit)


def _try_author_api(author_name: str, limit: int) -> list[Paper]:
    """Try the Semantic Scholar Author Search endpoint."""
    AUTHOR_SEARCH_URL = f"{BASE_URL}/author/search"

    _rate_limit()

    try:
        resp = requests.get(
            AUTHOR_SEARCH_URL,
            params={"query": author_name, "limit": 10, "fields": "name,paperCount,hIndex"},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]Author search API error: {e}[/red]")
        return []

    authors_found = data.get("data", [])
    if not authors_found:
        console.print(f"[yellow]No authors found via API for '{author_name}'[/yellow]")
        return []

    # Find the best matching author (prefer one with most papers)
    best_author = max(authors_found, key=lambda a: a.get("paperCount", 0))
    author_id = best_author.get("authorId")
    author_display = best_author.get("name", author_name)
    paper_count = best_author.get("paperCount", 0)
    console.print(f"[green]Found author:[/green] {author_display} (ID: {author_id}, {paper_count} papers)")

    if not author_id:
        return []

    # Fetch their papers
    AUTHOR_PAPERS_URL = f"{BASE_URL}/author/{author_id}/papers"
    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,tldr,fieldsOfStudy,"
        "citations.paperId,references.paperId"
    )

    _rate_limit()

    try:
        resp = requests.get(
            AUTHOR_PAPERS_URL,
            params={"fields": fields, "limit": min(limit, 100)},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]Author papers fetch error: {e}[/red]")
        return []

    papers = []
    for item in data.get("data", []):
        paper = _parse_paper(item)
        if paper:
            papers.append(paper)

    console.print(f"[green]Author '{author_display}':[/green] Fetched {len(papers)} papers")
    return papers[:limit]


def _author_paper_search_fallback(author_name: str, limit: int) -> list[Paper]:
    """Fallback: search for papers containing the author's name."""
    query = f"author:{author_name}"
    console.print(f"[yellow]Fallback search:[/yellow] '{query}'")
    return search_papers(query, limit=limit)

