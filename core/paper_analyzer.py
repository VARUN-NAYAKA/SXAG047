"""
ScholAR - PDF Deep Analysis Module.
Performs comprehensive analysis of an uploaded paper using Gemini LLM.
With automatic retry and model fallback for quota issues.
"""

import os
from core.llm_utils import call_gemini
from rich.console import Console

console = Console()



def analyze_paper_deeply(pdf_text: str) -> dict:
    """
    Perform a comprehensive analysis of an uploaded research paper.
    Returns a structured dict with summary, pros/cons, key points, trends, etc.
    """
    # Use up to 8000 chars for thorough analysis
    snippet = pdf_text[:8000]

    prompt = f"""You are an expert academic reviewer. Analyze this research paper text thoroughly and provide a detailed, structured analysis.

IMPORTANT: Be specific, cite actual claims from the text, and provide real insights — not generic statements.

Respond in EXACTLY this structured format (use the exact headers):

## TITLE
[Extract the exact paper title]

## SUMMARY
[Write a comprehensive 4-6 sentence summary of the paper, covering the problem, methodology, key findings, and conclusions. Be specific about what the paper actually says.]

## KEY CONTRIBUTIONS
- [List 3-5 specific contributions this paper makes to the field]

## METHODOLOGY
[Describe the methodology/approach used in 2-3 sentences]

## STRENGTHS (PROS)
- [List 3-5 specific strengths of this paper with justification]

## WEAKNESSES (CONS)  
- [List 3-5 specific weaknesses, limitations, or gaps in this paper]

## KEY FINDINGS
- [List 4-6 specific, concrete findings or results mentioned in the paper]

## TRENDS AND EVOLUTION
- [List 3-5 trends, developments, or evolutionary patterns discussed or implied in this paper]

## PRACTICAL IMPLICATIONS
- [List 2-3 real-world applications or implications of this research]

## SEARCH QUERIES
[Provide exactly 3 specific search queries that would find papers most similar to this one, separated by newlines]

Paper text:
---
{snippet}
---"""

    try:
        text = call_gemini(prompt, max_tokens=4096)
        return _parse_analysis(text)
    except Exception as e:
        console.print(f"[red]Paper analysis error: {e}[/red]")
        return _empty_analysis(str(e))


def generate_comparison_report(pdf_text: str, similar_papers: list) -> str:
    """
    Compare the uploaded paper with similar papers found by ScholAR.
    Returns a markdown comparison report.
    """
    if not similar_papers:
        return ""

    snippet = pdf_text[:3000]

    papers_text = ""
    for i, p in enumerate(similar_papers[:8], 1):
        papers_text += (
            f"{i}. **{p.title}** ({p.year or 'N/A'}) — {p.citation_count} citations\n"
            f"   Abstract: {(p.abstract or 'N/A')[:200]}\n"
            f"   Venue: {p.venue or 'N/A'}\n\n"
        )

    prompt = f"""You are an expert academic reviewer. Compare the uploaded paper with these related papers found in the literature.

Uploaded paper (excerpt):
---
{snippet[:2000]}
---

Similar papers found:
{papers_text}

Provide a DETAILED comparison report in markdown format covering:

### How This Paper Fits in the Literature
[2-3 sentences on where this paper sits among the related work]

### Key Differences from Existing Work
- [List 3-4 specific differences between the uploaded paper and the related papers]

### What This Paper Adds That Others Don't
- [List 2-3 unique contributions]

### Papers That Complement This Work
- [Recommend 2-3 specific papers from the list above that readers should also read, with reasons]

### Overall Assessment
[2-3 sentence assessment of the paper's novelty and significance relative to the field]

Be SPECIFIC — reference actual paper titles and findings. Do NOT be generic."""

    try:
        return call_gemini(prompt, max_tokens=3000)
    except Exception as e:
        console.print(f"[red]Comparison report error: {e}[/red]")
        return ""


def _parse_analysis(text: str) -> dict:
    """Parse the structured LLM analysis response into a dict."""
    result = {
        "title": "",
        "summary": "",
        "contributions": [],
        "methodology": "",
        "pros": [],
        "cons": [],
        "key_findings": [],
        "trends": [],
        "implications": [],
        "search_queries": [],
        "raw_text": text,
    }

    current_section = None
    current_content = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Detect section headers
        if stripped.startswith("## TITLE"):
            _save_section(result, current_section, current_content)
            current_section = "title"
            current_content = []
        elif stripped.startswith("## SUMMARY"):
            _save_section(result, current_section, current_content)
            current_section = "summary"
            current_content = []
        elif stripped.startswith("## KEY CONTRIBUTIONS"):
            _save_section(result, current_section, current_content)
            current_section = "contributions"
            current_content = []
        elif stripped.startswith("## METHODOLOGY"):
            _save_section(result, current_section, current_content)
            current_section = "methodology"
            current_content = []
        elif stripped.startswith("## STRENGTHS") or stripped.startswith("## PROS"):
            _save_section(result, current_section, current_content)
            current_section = "pros"
            current_content = []
        elif stripped.startswith("## WEAKNESSES") or stripped.startswith("## CONS"):
            _save_section(result, current_section, current_content)
            current_section = "cons"
            current_content = []
        elif stripped.startswith("## KEY FINDINGS"):
            _save_section(result, current_section, current_content)
            current_section = "key_findings"
            current_content = []
        elif stripped.startswith("## TRENDS"):
            _save_section(result, current_section, current_content)
            current_section = "trends"
            current_content = []
        elif stripped.startswith("## PRACTICAL"):
            _save_section(result, current_section, current_content)
            current_section = "implications"
            current_content = []
        elif stripped.startswith("## SEARCH"):
            _save_section(result, current_section, current_content)
            current_section = "search_queries"
            current_content = []
        elif stripped:
            current_content.append(stripped)

    _save_section(result, current_section, current_content)
    return result


def _save_section(result: dict, section: str, content: list):
    """Save accumulated content to the result dict."""
    if not section or not content:
        return

    if section in ("title", "summary", "methodology"):
        result[section] = "\n".join(content)
    elif section == "search_queries":
        queries = []
        for line in content:
            cleaned = line.lstrip("- •0123456789.)")
            if cleaned and len(cleaned) > 5:
                queries.append(cleaned.strip())
        result[section] = queries[:3]
    else:
        items = []
        for line in content:
            if line.startswith("- ") or line.startswith("• "):
                items.append(line[2:])
            elif line.startswith("* "):
                items.append(line[2:])
            elif items:
                items[-1] += " " + line
            else:
                items.append(line)
        result[section] = items


def _empty_analysis(error: str) -> dict:
    return {
        "title": "Analysis unavailable",
        "summary": f"Could not analyze the paper: {error}",
        "contributions": [],
        "methodology": "",
        "pros": [],
        "cons": [],
        "key_findings": [],
        "trends": [],
        "implications": [],
        "search_queries": [],
        "raw_text": "",
    }
