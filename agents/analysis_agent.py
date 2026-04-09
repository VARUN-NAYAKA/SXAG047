"""
ScholAR - Analysis Agent.
Orchestrates graph analysis: clustering, gap detection, trend analysis, contradictions.
"""

import json
import os
import uuid
from typing import Optional
from collections import Counter, defaultdict

from core.llm_utils import call_gemini

from core.config import (
    GEMINI_MODEL, LLM_TEMPERATURE,
    MIN_CLUSTERS_REQUIRED,
)
from core.models import (
    Paper, Topic, ResearchGap, Contradiction, Trend, AnalysisResult,
)
from core.prompts import (
    TOPIC_EXTRACTION_PROMPT, GAP_DETECTION_PROMPT,
    CONTRADICTION_DETECTION_PROMPT, TREND_ANALYSIS_PROMPT,
)
from graph.builder import KnowledgeGraphBuilder
from graph.algorithms import (
    detect_communities, compute_pagerank, compute_betweenness_centrality,
    find_key_papers, get_community_labels, detect_temporal_trends,
)
from rich.console import Console

console = Console()

# LLM calls go through core.llm_utils.call_gemini (retry + fallback)

def _call_llm(prompt: str) -> str:
    try:
        return call_gemini(prompt, max_tokens=4096)
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        return ""


def _parse_json(text: str) -> any:
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()
    # Handle ```json ... ``` wrapping
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object in the text
    for i, ch in enumerate(text):
        if ch in ('[', '{'):
            bracket = ']' if ch == '[' else '}'
            depth = 0
            for j in range(i, len(text)):
                if text[j] == ch:
                    depth += 1
                elif text[j] == bracket:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j+1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


# ──────────────────────────────────────────────
# Topic Extraction
# ──────────────────────────────────────────────
def extract_topics_batch(papers: list[Paper], batch_size: int = 5) -> dict[str, list[str]]:
    console.print(f"\n[bold magenta]📚 Extracting topics from {len(papers)} papers...[/bold magenta]")

    paper_topics: dict[str, list[str]] = {}

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]

        for paper in batch:
            if not paper.abstract:
                paper_topics[paper.paper_id] = paper.topics or paper.fields_of_study[:3]
                continue

            prompt = TOPIC_EXTRACTION_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract[:500],
            )

            try:
                result_text = _call_llm(prompt)
                data = _parse_json(result_text)
                topics = data.get("topics", []) + data.get("keywords", [])
                paper_topics[paper.paper_id] = topics[:5]
                paper.topics = topics[:5]
            except Exception:
                paper_topics[paper.paper_id] = paper.topics or paper.fields_of_study[:3]

        console.print(f"  Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")

    return paper_topics


# ──────────────────────────────────────────────
# Full Analysis Pipeline
# ──────────────────────────────────────────────
def run_full_analysis(
    kg: KnowledgeGraphBuilder,
    topic: str,
) -> AnalysisResult:
    console.print("\n[bold magenta]📊 Running full graph analysis...[/bold magenta]")

    papers = list(kg.papers.values())
    graph = kg.graph
    undirected = kg.get_undirected_copy()

    # ── 1. Community Detection ──
    console.print("\n[magenta]Step 1/5:[/magenta] Community detection...")
    partition = detect_communities(undirected)

    # ── 2. Centrality Analysis ──
    console.print("[magenta]Step 2/5:[/magenta] Centrality analysis...")
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness_centrality(undirected)

    # ── 3. Key Papers ──
    console.print("[magenta]Step 3/5:[/magenta] Identifying key papers...")
    key_paper_list = find_key_papers(graph, pagerank, betweenness, top_k=20)
    key_paper_ids = [pid for pid, _ in key_paper_list]

    # ── 4. Community Labels (sorted by cluster_id) ──
    community_info = get_community_labels(undirected, partition, kg.papers)

    clusters = []
    for comm_id in sorted(community_info.keys()):
        info = community_info[comm_id]
        cluster = Topic(
            topic_id=f"cluster_{comm_id}",
            name=f"Cluster {comm_id}: {', '.join(info['top_keywords'][:3])}" if info['top_keywords'] else f"Cluster {comm_id}",
            description=f"Contains {info['paper_count']} papers. Keywords: {', '.join(info['top_keywords'])}. Year range: {info['year_range']}",
            paper_ids=info["paper_ids"],
            cluster_id=comm_id,
            keyword_list=info["top_keywords"],
        )
        clusters.append(cluster)

    # ── 5. Gap Detection (LLM + algorithmic fallback) ──
    console.print("[magenta]Step 4/5:[/magenta] Detecting research gaps...")
    gaps = _detect_gaps_with_llm(topic, clusters, papers)
    if not gaps:
        console.print("[yellow]LLM gap detection failed, using algorithmic fallback...[/yellow]")
        gaps = _detect_gaps_algorithmic(topic, clusters, papers)
    console.print(f"  → Found {len(gaps)} gaps")

    # ── 6. Trend Analysis (LLM + algorithmic fallback) ──
    console.print("[magenta]Step 5/5:[/magenta] Analyzing trends...")
    year_counts = detect_temporal_trends(papers)
    console.print(f"  → Year distribution: {dict(sorted(year_counts.items()))}")
    trends = _analyze_trends_with_llm(topic, year_counts, clusters)
    if not trends:
        console.print("[yellow]LLM trend analysis failed, using algorithmic fallback...[/yellow]")
        trends = _analyze_trends_algorithmic(topic, year_counts, clusters, papers)
    console.print(f"  → Found {len(trends)} trends")

    # ── 7. Coverage Score ──
    num_clusters = len(clusters)
    coverage = min(num_clusters / max(MIN_CLUSTERS_REQUIRED, 1), 1.0) * 0.5
    coverage += min(len(papers) / 50, 1.0) * 0.3
    coverage += min(len(key_paper_ids) / 10, 1.0) * 0.2

    result = AnalysisResult(
        clusters=clusters,
        key_papers=key_paper_ids,
        gaps=gaps,
        contradictions=[],
        trends=trends,
        coverage_score=coverage,
        year_counts=year_counts,
    )

    console.print(f"\n[bold green]✅ Analysis complete![/bold green]")
    console.print(f"  Clusters: {len(clusters)}")
    console.print(f"  Key papers: {len(key_paper_ids)}")
    console.print(f"  Research gaps: {len(gaps)}")
    console.print(f"  Trends: {len(trends)}")
    console.print(f"  Coverage score: {coverage:.2f}")

    result._partition = partition
    result._pagerank = pagerank

    return result


# ──────────────────────────────────────────────
# Gap Detection (LLM-powered)
# ──────────────────────────────────────────────
def _detect_gaps_with_llm(topic, clusters, papers) -> list[ResearchGap]:
    if not clusters:
        return []

    clusters_text = ""
    for c in clusters:
        cluster_papers = [p for p in papers if p.paper_id in c.paper_ids]
        sample_titles = [p.title for p in cluster_papers[:3]]
        clusters_text += (
            f"- {c.name}\n"
            f"  Papers: {len(c.paper_ids)}\n"
            f"  Keywords: {', '.join(c.keyword_list)}\n"
            f"  Sample papers: {'; '.join(sample_titles)}\n\n"
        )

    years = [p.year for p in papers if p.year]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    prompt = GAP_DETECTION_PROMPT.format(
        topic=topic,
        clusters_text=clusters_text,
        total_papers=len(papers),
        num_clusters=len(clusters),
        year_range=year_range,
    )

    try:
        result_text = _call_llm(prompt)
        if not result_text:
            return []
        data = _parse_json(result_text)
        gaps = []
        for item in data:
            gaps.append(ResearchGap(
                gap_id=f"gap_{uuid.uuid4().hex[:8]}",
                description=item.get("description", "Unknown gap"),
                confidence=item.get("confidence", 0.5),
                related_topics=item.get("related_topics", []),
                suggested_directions=item.get("suggested_directions", []),
            ))
        return gaps
    except Exception as e:
        console.print(f"[red]Gap detection error: {e}[/red]")
        return []


def _detect_gaps_algorithmic(topic, clusters, papers) -> list[ResearchGap]:
    """Algorithmic fallback: detect gaps from data patterns without LLM."""
    gaps = []
    years = [p.year for p in papers if p.year]

    # Gap 1: Temporal coverage gaps
    if years:
        min_y, max_y = min(years), max(years)
        year_counter = Counter(years)
        for y in range(min_y, max_y + 1):
            if year_counter.get(y, 0) == 0:
                gaps.append(ResearchGap(
                    gap_id=f"gap_temporal_{y}",
                    description=f"No papers found from {y}, suggesting a gap in research continuity for '{topic}'.",
                    confidence=0.6,
                    related_topics=[topic, "temporal coverage"],
                    suggested_directions=[f"Investigate what research on {topic} was conducted in {y}"],
                ))
                break  # Only report one temporal gap

    # Gap 2: Small clusters indicate under-explored areas
    if clusters:
        avg_size = sum(len(c.paper_ids) for c in clusters) / len(clusters)
        for c in clusters:
            if len(c.paper_ids) < avg_size * 0.4 and len(c.paper_ids) >= 2:
                gaps.append(ResearchGap(
                    gap_id=f"gap_small_{c.cluster_id}",
                    description=f"Under-explored research area: '{c.name}' has only {len(c.paper_ids)} papers, significantly fewer than the average of {avg_size:.0f} papers per cluster.",
                    confidence=0.7,
                    related_topics=c.keyword_list[:3],
                    suggested_directions=[
                        f"Conduct more research combining {' and '.join(c.keyword_list[:2])}",
                        f"Explore interdisciplinary approaches to {c.keyword_list[0] if c.keyword_list else topic}",
                    ],
                ))

    # Gap 3: Missing cross-cluster connections
    if len(clusters) >= 2:
        all_keywords = set()
        for c in clusters:
            all_keywords.update(c.keyword_list[:3])
        cluster_keyword_sets = [set(c.keyword_list[:3]) for c in clusters]
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                overlap = cluster_keyword_sets[i] & cluster_keyword_sets[j]
                if not overlap:
                    kw_i = clusters[i].keyword_list[:2]
                    kw_j = clusters[j].keyword_list[:2]
                    gaps.append(ResearchGap(
                        gap_id=f"gap_cross_{i}_{j}",
                        description=f"No connection between '{clusters[i].name}' and '{clusters[j].name}'. Combining {', '.join(kw_i)} with {', '.join(kw_j)} could yield novel insights.",
                        confidence=0.65,
                        related_topics=kw_i + kw_j,
                        suggested_directions=[
                            f"Investigate intersection of {kw_i[0] if kw_i else 'cluster ' + str(i)} and {kw_j[0] if kw_j else 'cluster ' + str(j)}",
                        ],
                    ))
                    if len(gaps) >= 5:
                        break
            if len(gaps) >= 5:
                break

    # Gap 4: Recency gap
    if years:
        recent_papers = [p for p in papers if p.year and p.year >= max(years) - 1]
        if len(recent_papers) < len(papers) * 0.15:
            gaps.append(ResearchGap(
                gap_id=f"gap_recency",
                description=f"Only {len(recent_papers)} of {len(papers)} papers are from the last 2 years ({max(years)-1}-{max(years)}), suggesting the field may be maturing or shifting focus.",
                confidence=0.6,
                related_topics=[topic, "recent developments"],
                suggested_directions=["Investigate latest developments and emerging sub-fields"],
            ))

    return gaps[:5]  # Limit to 5


# ──────────────────────────────────────────────
# Trend Analysis (LLM-powered)
# ──────────────────────────────────────────────
def _analyze_trends_with_llm(topic, year_counts, clusters) -> list[Trend]:
    if not year_counts:
        return []

    yearly_data = "\n".join(f"  {year}: {count} papers" for year, count in sorted(year_counts.items()))
    topic_yearly_data = ""
    for cluster in clusters:
        topic_yearly_data += f"  {cluster.name}: {len(cluster.paper_ids)} papers\n"

    prompt = TREND_ANALYSIS_PROMPT.format(
        topic=topic,
        yearly_data=yearly_data,
        topic_yearly_data=topic_yearly_data,
    )

    try:
        result_text = _call_llm(prompt)
        if not result_text:
            return []
        data = _parse_json(result_text)
        trends = []
        for item in data:
            trends.append(Trend(
                trend_id=f"trend_{uuid.uuid4().hex[:8]}",
                description=item.get("description", ""),
                topic=item.get("topic", ""),
                direction=item.get("direction", "rising"),
                start_year=item.get("start_year"),
                strength=item.get("strength", 0.5),
                paper_count_by_year=year_counts,
            ))
        return trends
    except Exception as e:
        console.print(f"[red]Trend analysis error: {e}[/red]")
        return []


def _analyze_trends_algorithmic(topic, year_counts, clusters, papers) -> list[Trend]:
    """Algorithmic fallback: compute trends from year distribution without LLM."""
    if not year_counts or len(year_counts) < 2:
        return []

    trends = []
    sorted_years = sorted(year_counts.items())
    years = [y for y, _ in sorted_years]
    counts = [c for _, c in sorted_years]

    # Overall publication trend
    if len(counts) >= 3:
        first_half = sum(counts[:len(counts)//2])
        second_half = sum(counts[len(counts)//2:])
        if second_half > first_half * 1.5:
            direction = "rising"
            strength = min((second_half / max(first_half, 1)), 2.0) / 2.0
        elif first_half > second_half * 1.5:
            direction = "declining"
            strength = min((first_half / max(second_half, 1)), 2.0) / 2.0
        else:
            direction = "stable"
            strength = 0.5

        trends.append(Trend(
            trend_id="trend_overall",
            description=f"Overall publication rate for '{topic}' is {direction}. "
                        f"First half: {first_half} papers, second half: {second_half} papers.",
            topic=topic,
            direction=direction,
            start_year=years[0],
            end_year=years[-1],
            strength=strength,
            paper_count_by_year=year_counts,
        ))

    # Peak year
    peak_year = max(year_counts, key=year_counts.get)
    peak_count = year_counts[peak_year]
    trends.append(Trend(
        trend_id="trend_peak",
        description=f"Peak publication year was {peak_year} with {peak_count} papers, "
                    f"indicating strongest research activity during that period.",
        topic=f"{topic} (peak activity)",
        direction="rising" if peak_year >= years[-1] - 2 else "stable",
        start_year=peak_year,
        strength=0.8,
        paper_count_by_year=year_counts,
    ))

    # Recent surge/decline
    if len(years) >= 3:
        last_3 = [year_counts.get(y, 0) for y in range(years[-1]-2, years[-1]+1)]
        prev_3 = [year_counts.get(y, 0) for y in range(years[-1]-5, years[-1]-2)]
        if sum(last_3) > sum(prev_3) * 1.3 and sum(prev_3) > 0:
            trends.append(Trend(
                trend_id="trend_recent_surge",
                description=f"Recent surge in research: {sum(last_3)} papers in last 3 years vs {sum(prev_3)} in prior 3 years.",
                topic=f"{topic} (emerging)",
                direction="emerging",
                start_year=years[-1] - 2,
                strength=0.75,
                paper_count_by_year=year_counts,
            ))

    return trends


# ──────────────────────────────────────────────
# Contradiction Detection
# ──────────────────────────────────────────────
def detect_contradictions(papers, similarity_pairs, max_pairs=10) -> list[Contradiction]:
    paper_dict = {p.paper_id: p for p in papers}
    contradictions = []
    sorted_pairs = sorted(similarity_pairs, key=lambda x: x[2], reverse=True)[:max_pairs]

    for pid_a, pid_b, sim in sorted_pairs:
        paper_a = paper_dict.get(pid_a)
        paper_b = paper_dict.get(pid_b)
        if not paper_a or not paper_b or not paper_a.abstract or not paper_b.abstract:
            continue

        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            title_a=paper_a.title,
            abstract_a=paper_a.abstract[:500],
            title_b=paper_b.title,
            abstract_b=paper_b.abstract[:500],
        )

        try:
            result_text = _call_llm(prompt)
            data = _parse_json(result_text)
            if data.get("is_contradictory", False) and data.get("confidence", 0) > 0.6:
                contradictions.append(Contradiction(
                    paper_a_id=pid_a,
                    paper_b_id=pid_b,
                    paper_a_claim=data.get("paper_a_claim", ""),
                    paper_b_claim=data.get("paper_b_claim", ""),
                    description=data.get("description", ""),
                    confidence=data.get("confidence", 0.0),
                ))
        except Exception:
            continue

    return contradictions
