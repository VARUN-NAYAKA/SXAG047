"""
ScholAR - PDF Utility Module.
Extracts text and research metadata from uploaded PDF papers.
"""

import os
from core.llm_utils import call_gemini
from rich.console import Console

console = Console()


def extract_text_from_pdf(pdf_buffer, max_pages: int = 3) -> str:
    """Extract text from a PDF file buffer (UploadedFile from Streamlit)."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_buffer)
        text_parts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n".join(text_parts)
        console.print(f"[green]PDF:[/green] Extracted {len(full_text)} chars from {min(len(reader.pages), max_pages)} pages")
        return full_text
    except Exception as e:
        console.print(f"[red]PDF extraction error: {e}[/red]")
        return ""


def extract_seed_topic(pdf_text: str) -> dict:
    """Use LLM to extract the primary research topic, title, and keywords from PDF text.
    
    Returns:
        dict with keys: title, topic, keywords, abstract_summary
    """
    # Use first 3000 chars to stay within limits and be fast
    snippet = pdf_text[:3000]

    prompt = f"""Analyze this academic paper text and extract the following information.
Respond in EXACTLY this format (no markdown, no extra text):

TITLE: <paper title>
TOPIC: <primary research topic as a short search query, 5-10 words>
KEYWORDS: <comma-separated list of 5-8 research keywords>
SUMMARY: <one paragraph summary of the paper's main contribution, 2-3 sentences>

Paper text:
---
{snippet}
---"""

    try:
        text = call_gemini(prompt, max_tokens=500, temperature=0.2)

        result = {
            "title": "",
            "topic": "",
            "keywords": [],
            "abstract_summary": "",
        }

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("TITLE:"):
                result["title"] = line[6:].strip()
            elif line.startswith("TOPIC:"):
                result["topic"] = line[6:].strip()
            elif line.startswith("KEYWORDS:"):
                result["keywords"] = [k.strip() for k in line[9:].split(",")]
            elif line.startswith("SUMMARY:"):
                result["abstract_summary"] = line[8:].strip()

        # Fallback if LLM failed to give a topic
        if not result["topic"]:
            return _fallback_extraction(pdf_text)

        console.print(f"[green]PDF Seed:[/green] Topic = '{result['topic']}'")
        return result

    except Exception as e:
        console.print(f"[red]LLM extraction error: {e}[/red]")
        return _fallback_extraction(pdf_text)


def _fallback_extraction(pdf_text: str) -> dict:
    """Simple keyword-based extraction when LLM is unavailable."""
    lines = pdf_text.strip().split("\n")
    # First non-empty line is usually the title
    title = ""
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            title = line
            break

    # Use title as topic
    return {
        "title": title,
        "topic": title[:80] if title else "research paper analysis",
        "keywords": [],
        "abstract_summary": "",
    }
