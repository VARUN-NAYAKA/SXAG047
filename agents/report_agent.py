"""
ScholAR - Report Generator Agent.
Creates structured literature review reports with visualizations and PDF export.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px

from core.config import (
    GEMINI_MODEL, LLM_TEMPERATURE,
    DATA_DIR,
)
from core.models import (
    Paper, AnalysisResult, LiteratureReport, ReportSection, TopPaperRecommendation,
)
from core.prompts import REPORT_SECTION_PROMPT, EXECUTIVE_SUMMARY_PROMPT
from rich.console import Console

console = Console()

# ──────────────────────────────────────────────
# LAZY LLM INITIALIZATION
# ──────────────────────────────────────────────
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        _llm = genai.GenerativeModel(GEMINI_MODEL)
    return _llm


def _call_llm(prompt: str, max_tokens: int = 4096) -> str:
    llm = _get_llm()
    try:
        response = llm.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except Exception as e:
        console.print(f"[red]LLM error: {e}[/red]")
        return ""


# ──────────────────────────────────────────────
# Visualization Generation
# ──────────────────────────────────────────────
def create_trend_chart(analysis: AnalysisResult, topic: str) -> str:
    """Create a publication trend chart — works from raw year_counts even without LLM trends."""
    # Get year data from raw counts OR from trend objects
    year_counts = {}

    # Prefer raw year_counts stored in analysis
    if analysis.year_counts:
        year_counts = analysis.year_counts
    elif analysis.trends:
        for trend in analysis.trends:
            for year, count in trend.paper_count_by_year.items():
                year_counts[year] = count

    if not year_counts:
        console.print("[yellow]No year data for trend chart[/yellow]")
        return ""

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=counts,
        mode='lines+markers',
        name='Publications',
        line=dict(color='#e94560', width=3),
        marker=dict(size=8, color='#e94560'),
        fill='tozeroy',
        fillcolor='rgba(233, 69, 96, 0.1)',
    ))

    fig.update_layout(
        title=dict(
            text=f"Publication Trends: {topic}",
            font=dict(size=18, color='#1e293b'),
        ),
        xaxis_title="Year",
        yaxis_title="Number of Papers",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fc",
        font=dict(color="#334155"),
        showlegend=False,
    )

    output_path = DATA_DIR / "trend_chart.html"
    fig.write_html(str(output_path))
    console.print(f"[green]Trend chart created with {len(years)} years of data[/green]")
    return str(output_path)


def create_cluster_chart(analysis: AnalysisResult) -> str:
    """Create a cluster distribution chart — fixed for legibility."""
    if not analysis.clusters:
        return ""

    sorted_clusters = sorted(analysis.clusters, key=lambda c: c.cluster_id or 0)

    labels = []
    for c in sorted_clusters:
        keywords = ", ".join(c.keyword_list[:2]) if c.keyword_list else "misc"
        labels.append(f"C{c.cluster_id}: {keywords[:25]}")

    sizes = [len(c.paper_ids) for c in sorted_clusters]

    colors = [
        '#e94560', '#3498db', '#2ecc71', '#f39c12',
        '#9b59b6', '#1abc9c', '#e67e22', '#e74c3c',
        '#2980b9', '#27ae60', '#c0392b', '#8e44ad',
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        marker=dict(
            colors=colors[:len(labels)],
            line=dict(color='#ffffff', width=2),  # White border between slices
        ),
        textinfo='percent',
        textposition='inside',
        textfont=dict(color='white', size=13, family='Arial Black'),
        hoverinfo='label+value+percent',
        hole=0.45,
    )])

    fig.update_layout(
        title=dict(
            text="Research Cluster Distribution",
            font=dict(size=18, color='#1e293b'),
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155"),
        showlegend=True,
        legend=dict(
            font=dict(size=12, color="#334155"),
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            borderwidth=1,
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
        margin=dict(r=220),
        height=500,
    )

    output_path = DATA_DIR / "cluster_chart.html"
    fig.write_html(str(output_path))
    return str(output_path)


def create_citation_chart(papers: list[Paper], top_n: int = 15) -> str:
    """Create a bar chart of most cited papers — improved text visibility."""
    sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)[:top_n]

    if not sorted_papers:
        return ""

    titles = [p.title[:50] + "..." if len(p.title) > 50 else p.title for p in sorted_papers]
    citations = [p.citation_count for p in sorted_papers]

    fig = go.Figure(data=[go.Bar(
        x=citations,
        y=titles,
        orientation='h',
        marker=dict(
            color=citations,
            colorscale='Viridis',
            line=dict(color='rgba(255,255,255,0.3)', width=1),
        ),
        texttemplate='%{x}',
        textposition='outside',
        textfont=dict(color='#334155', size=12),
    )])

    fig.update_layout(
        title=dict(
            text="Most Cited Papers",
            font=dict(size=18, color='#1e293b'),
        ),
        xaxis_title="Citation Count",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fc",
        font=dict(color="#334155", size=11),
        height=400 + len(sorted_papers) * 30,
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#334155", size=11),
        ),
        margin=dict(l=350),
    )

    output_path = DATA_DIR / "citation_chart.html"
    fig.write_html(str(output_path))
    return str(output_path)


# ──────────────────────────────────────────────
# Top Paper Recommendations
# ──────────────────────────────────────────────
def generate_top_recommendations(papers: list[Paper], analysis: AnalysisResult) -> list[TopPaperRecommendation]:
    """Pick top 3 papers based on a weighted score of relevance, citations, and centrality."""
    if not papers:
        return []

    pagerank = getattr(analysis, '_pagerank', {})

    scored = []
    for paper in papers:
        # Weighted composite score
        rel_score = paper.relevance_score or 0
        cite_score = min(paper.citation_count / 500, 1.0) if paper.citation_count else 0
        pr_score = pagerank.get(paper.paper_id, 0) * 100  # Scale up

        composite = (0.4 * rel_score) + (0.35 * cite_score) + (0.25 * pr_score)
        confidence = min(composite * 100, 99)

        reason = []
        if paper.citation_count > 50:
            reason.append(f"Highly cited ({paper.citation_count} citations)")
        if rel_score > 0.7:
            reason.append(f"Strong topic relevance ({rel_score:.0%})")
        if pr_score > 0.01:
            reason.append("Central in knowledge graph")
        if paper.year and paper.year >= 2023:
            reason.append("Recent publication")
        if not reason:
            reason.append("Good overall research quality")

        scored.append((paper, confidence, " • ".join(reason)))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        TopPaperRecommendation(paper=paper, confidence=conf, reason=reason)
        for paper, conf, reason in scored[:3]
    ]


# ──────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────
def generate_report(
    topic: str,
    papers: list[Paper],
    analysis: AnalysisResult,
) -> LiteratureReport:
    console.print(f"\n[bold yellow]📝 Generating literature review for: {topic}[/bold yellow]")

    paper_dict = {p.paper_id: p for p in papers}
    sections: list[ReportSection] = []

    # ── 1. Executive Summary ──
    console.print("[yellow]Section 1/5:[/yellow] Executive summary...")
    exec_summary = _generate_executive_summary(topic, papers, analysis)
    if not exec_summary:
        exec_summary = _generate_fallback_summary(topic, papers, analysis)
    sections.append(ReportSection(title="Executive Summary", content=exec_summary, order=1))

    # ── 2. Research Landscape ──
    console.print("[yellow]Section 2/5:[/yellow] Research landscape...")
    landscape = _generate_landscape_section(topic, papers, analysis, paper_dict)
    if not landscape:
        landscape = _generate_fallback_landscape(topic, papers, analysis)
    sections.append(ReportSection(title="Research Landscape", content=landscape, order=2))

    # ── 3. Research Clusters (for KG tab) ──
    console.print("[yellow]Section 3/5:[/yellow] Research clusters...")
    for i, cluster in enumerate(sorted(analysis.clusters, key=lambda c: c.cluster_id or 0)):
        cluster_content = generate_cluster_section(topic, cluster, paper_dict)
        sections.append(ReportSection(
            title=f"Research Theme: {cluster.name}",
            content=cluster_content,
            order=10 + i,
        ))

    # ── 4. Temporal Trends ──
    console.print("[yellow]Section 4/5:[/yellow] Temporal trends...")
    trends_content = _generate_trends_section(topic, analysis)
    sections.append(ReportSection(
        title="Temporal Trends and Evolution",
        content=trends_content,
        order=50,
    ))

    # ── 5. Research Gaps ──
    console.print("[yellow]Section 5/5:[/yellow] Research gaps...")
    gaps_content = _generate_gaps_section(topic, analysis)
    sections.append(ReportSection(
        title="Research Gaps and Future Directions",
        content=gaps_content,
        order=51,
    ))

    # ── Recommended Reading ──
    reading_content = _generate_reading_list(analysis, paper_dict)
    sections.append(ReportSection(title="Recommended Reading Path", content=reading_content, order=52))

    # ── References ──
    refs_content = _generate_references(papers)
    sections.append(ReportSection(title="References", content=refs_content, order=100))

    # Generate visualizations
    console.print("[yellow]Creating visualizations...[/yellow]")
    create_trend_chart(analysis, topic)
    create_cluster_chart(analysis)
    create_citation_chart(papers)

    # Top 3 recommendations
    top_recs = generate_top_recommendations(papers, analysis)

    report = LiteratureReport(
        topic=topic,
        total_papers_analyzed=len(papers),
        sections=sections,
        references=papers,
        analysis=analysis,
        top_recommendations=top_recs,
    )

    console.print(f"\n[bold green]✅ Literature review generated! ({len(sections)} sections)[/bold green]")
    return report


# ──────────────────────────────────────────────
# Section Generators
# ──────────────────────────────────────────────
def _generate_executive_summary(topic, papers, analysis):
    years = [p.year for p in papers if p.year]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    cluster_summaries = "\n".join(
        f"- {c.name}: {len(c.paper_ids)} papers, Keywords: {', '.join(c.keyword_list[:3])}"
        for c in analysis.clusters
    )

    key_findings = ""
    for gap in analysis.gaps[:3]:
        key_findings += f"- Gap: {gap.description}\n"
    for trend in analysis.trends[:3]:
        key_findings += f"- Trend: {trend.description}\n"

    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        topic=topic,
        total_papers=len(papers),
        num_clusters=len(analysis.clusters),
        year_range=year_range,
        num_gaps=len(analysis.gaps),
        num_contradictions=len(analysis.contradictions),
        cluster_summaries=cluster_summaries or "No clusters identified.",
        key_findings=key_findings or "General survey of the field.",
    )

    return _call_llm(prompt, max_tokens=2048)


def _generate_fallback_summary(topic, papers, analysis):
    years = [p.year for p in papers if p.year]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"
    return (
        f"This literature review analyzed **{len(papers)} papers** on the topic of "
        f"**{topic}**, spanning the years {year_range}. "
        f"The analysis identified **{len(analysis.clusters)} distinct research clusters**, "
        f"**{len(analysis.gaps)} research gaps**, and **{len(analysis.trends)} temporal trends**."
    )


def _generate_fallback_landscape(topic, papers, analysis):
    top_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)[:5]
    content = f"A total of **{len(papers)} papers** were analyzed.\n\n### Most Cited Papers\n"
    for p in top_papers:
        authors = ", ".join(a.name for a in p.authors[:3])
        content += f"- **{p.title}** ({p.year}) — {p.citation_count} citations — {authors}\n"
    return content


def _generate_landscape_section(topic, papers, analysis, paper_dict):
    top_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:10]
    papers_text = "\n".join(
        f"- [{p.title}] ({p.year}) - Citations: {p.citation_count}, Authors: {', '.join(a.name for a in p.authors[:3])}"
        for p in top_papers
    )
    insights = f"Total papers: {len(papers)}\nClusters: {len(analysis.clusters)}\nKey papers: {len(analysis.key_papers)}\n"

    prompt = REPORT_SECTION_PROMPT.format(
        topic=topic,
        section_title="Research Landscape Overview",
        section_purpose="Provide a broad overview of the research field, its scope, major contributors, and key publications.",
        papers_text=papers_text,
        insights_text=insights,
        target_length=400,
    )
    return _call_llm(prompt, max_tokens=3000)


def generate_cluster_section(topic: str, cluster, paper_dict: dict) -> str:
    """Generate a section for a specific research cluster. PUBLIC for app.py."""
    cluster_papers = [paper_dict[pid] for pid in cluster.paper_ids if pid in paper_dict]
    top_papers = sorted(cluster_papers, key=lambda p: p.citation_count, reverse=True)[:8]

    if not top_papers:
        return f"This cluster contains {len(cluster.paper_ids)} papers."

    papers_text = "\n".join(
        f"- [{p.title}] ({p.year}) - Citations: {p.citation_count}. "
        f"{'TLDR: ' + p.tldr if p.tldr else 'Abstract: ' + (p.abstract[:200] if p.abstract else 'N/A')}"
        for p in top_papers
    )

    prompt = REPORT_SECTION_PROMPT.format(
        topic=topic,
        section_title=cluster.name,
        section_purpose=f"Discuss the research theme '{cluster.name}' in detail.",
        papers_text=papers_text,
        insights_text=f"This cluster contains {len(cluster.paper_ids)} papers. Keywords: {', '.join(cluster.keyword_list)}",
        target_length=300,
    )

    result = _call_llm(prompt, max_tokens=2048)
    if not result:
        result = f"**{cluster.name}** — {len(cluster.paper_ids)} papers.\n\n**Key papers:**\n"
        for p in top_papers[:5]:
            result += f"- {p.title} ({p.year}) — {p.citation_count} citations\n"
    return result


def _generate_trends_section(topic, analysis):
    if not analysis.trends:
        return "No temporal trends could be identified from the collected papers."

    content = "The following temporal trends were identified:\n\n"
    for trend in analysis.trends:
        content += (
            f"**{trend.topic}** ({trend.direction.title()}): {trend.description}\n"
            f"- Strength: {trend.strength:.0%}\n"
            f"- Start year: {trend.start_year or 'N/A'}\n\n"
        )
    return content


def _generate_gaps_section(topic, analysis):
    if not analysis.gaps:
        return "No research gaps identified."

    content = "The following research gaps were identified:\n\n"
    for i, gap in enumerate(analysis.gaps, 1):
        content += (
            f"### Gap {i}: {gap.description}\n"
            f"- **Confidence:** {gap.confidence:.0%}\n"
            f"- **Related Topics:** {', '.join(gap.related_topics)}\n"
            f"- **Suggested Directions:**\n"
        )
        for direction in gap.suggested_directions:
            content += f"  - {direction}\n"
        content += "\n"
    return content


def _generate_reading_list(analysis, paper_dict):
    content = "### Must-Read Foundational Papers\n"
    for pid in analysis.key_papers[:5]:
        paper = paper_dict.get(pid)
        if paper:
            content += f"1. **{paper.title}** ({paper.year}) - {paper.citation_count} citations\n"
            if paper.tldr:
                content += f"   > {paper.tldr}\n"

    content += "\n### Recent Key Papers\n"
    recent = [paper_dict.get(pid) for pid in analysis.key_papers if paper_dict.get(pid) and paper_dict.get(pid).year]
    recent = sorted([p for p in recent if p], key=lambda p: p.year or 0, reverse=True)[:5]
    for paper in recent:
        content += f"1. **{paper.title}** ({paper.year}) - {paper.citation_count} citations\n"
    return content


def _generate_references(papers):
    sorted_papers = sorted(papers, key=lambda p: (p.authors[0].name if p.authors else "ZZZ", p.year or 0))
    content = ""
    for p in sorted_papers:
        authors_str = ", ".join(a.name for a in p.authors[:5])
        if len(p.authors) > 5:
            authors_str += ", et al."
        year = p.year or "n.d."
        venue = p.venue or ""
        doi_str = f" https://doi.org/{p.doi}" if p.doi else ""
        content += f"- {authors_str} ({year}). {p.title}. *{venue}*.{doi_str}\n"
    return content


# ──────────────────────────────────────────────
# PDF Export — Comprehensive Report
# ──────────────────────────────────────────────
def export_report_to_pdf(report: LiteratureReport) -> str:
    """Export the full report to PDF with all sections."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.ln(50)
    pdf.cell(0, 15, "Literature Review Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.ln(10)
    safe_topic = report.topic.encode("latin-1", errors="replace").decode("latin-1")
    pdf.cell(0, 10, safe_topic, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, f"Generated by ScholAR | {report.generated_at[:10]}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Papers analyzed: {report.total_papers_analyzed}", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Table of Contents ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    for i, section in enumerate(report.sections, 1):
        safe_title = section.title.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 7, f"{i}. {safe_title}", new_x="LMARGIN", new_y="NEXT")

    # ── Top 3 Recommendations ──
    if report.top_recommendations:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Top Recommended Papers", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        for i, rec in enumerate(report.top_recommendations, 1):
            pdf.set_font("Helvetica", "B", 12)
            safe_title = rec.paper.title.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, f"{i}. {safe_title}")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 5, f"   Confidence: {rec.confidence:.0f}% | Year: {rec.paper.year or 'N/A'} | Citations: {rec.paper.citation_count}", new_x="LMARGIN", new_y="NEXT")
            safe_reason = rec.reason.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 5, f"   Reason: {safe_reason}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ── Content Sections ──
    for section in report.sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        safe_title = section.title.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 12, safe_title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 10)
        content = section.content or "Content not available."
        content = content.replace("**", "").replace("###", "").replace("> ", "  ")

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            try:
                pdf.multi_cell(0, 5, safe_line)
            except Exception:
                continue

    output_path = DATA_DIR / "scholar_report.pdf"
    pdf.output(str(output_path))
    console.print(f"[green]PDF exported:[/green] {output_path}")
    return str(output_path)
