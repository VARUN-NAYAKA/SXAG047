"""
ScholAR - Orchestrator Agent (LangGraph State Machine).
The brain that coordinates all other agents: search → graph → analysis → report.
"""

import json
import os
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from core.llm_utils import call_gemini

from core.config import (
    GEMINI_MODEL, LLM_TEMPERATURE,
    MAX_ITERATIONS, MIN_PAPERS_FOR_ANALYSIS, MIN_CLUSTERS_REQUIRED,
    SIMILARITY_THRESHOLD,
)
from core.models import Paper, AnalysisResult, LiteratureReport, SearchQuery
from core.prompts import STOPPING_DECISION_PROMPT
from agents.search_agent import (
    expand_queries, search_with_queries, deduplicate_papers,
    score_papers, generate_embeddings,
)
from agents.analysis_agent import extract_topics_batch, run_full_analysis
from agents.report_agent import generate_report
from graph.builder import KnowledgeGraphBuilder
from graph.visualizer import create_interactive_graph
from rich.console import Console

console = Console()

# LLM calls go through core.llm_utils.call_gemini (retry + fallback)


# ──────────────────────────────────────────────
# Agent State
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    """The state that flows through the agent graph."""
    topic: str
    queries: list[SearchQuery]
    all_papers: list[Paper]          # Cumulative papers across iterations
    scored_papers: list[Paper]       # After scoring and filtering
    embeddings: dict                 # paper_id -> embedding vector
    knowledge_graph: KnowledgeGraphBuilder
    analysis: AnalysisResult
    report: LiteratureReport
    iteration: int
    status: str                      # Current step label for UI
    log: Annotated[list[str], operator.add]   # Log messages for UI streaming


# ──────────────────────────────────────────────
# Node Functions
# ──────────────────────────────────────────────
def plan_search(state: AgentState) -> dict:
    """Step 1: Expand the topic into diverse search queries."""
    topic = state["topic"]
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # First iteration: expand from topic
        queries = expand_queries(topic, num_queries=12)
        return {
            "queries": queries,
            "iteration": 1,
            "status": "Planning search strategy...",
            "log": [f"🔍 Generated {len(queries)} search queries for: {topic}"],
        }
    else:
        # Subsequent iterations: use LLM to decide new queries
        analysis = state.get("analysis")
        searched = [q.query_text for q in state.get("queries", [])]

        new_queries = _get_refinement_queries(topic, analysis, searched, iteration)
        all_queries = state.get("queries", []) + new_queries

        return {
            "queries": all_queries,
            "iteration": iteration + 1,
            "status": f"Refining search (iteration {iteration + 1})...",
            "log": [f"🔄 Iteration {iteration + 1}: Generated {len(new_queries)} new queries"],
        }


def execute_search(state: AgentState) -> dict:
    """Step 2: Execute search queries across all APIs."""
    queries = state["queries"]
    existing_papers = state.get("all_papers", [])

    # Only search new queries (from latest iteration)
    iteration = state.get("iteration", 1)
    # We just use all queries for simplicity; dedup handles repeats
    new_papers = search_with_queries(queries, limit_per_api=15)

    # Combine with existing
    combined = existing_papers + new_papers

    # Deduplicate
    unique_papers = deduplicate_papers(combined)

    return {
        "all_papers": unique_papers,
        "status": f"Found {len(unique_papers)} unique papers",
        "log": [f"📄 Collected {len(unique_papers)} unique papers (fetched {len(new_papers)} new)"],
    }


def score_and_filter(state: AgentState) -> dict:
    """Step 3: Score papers by relevance and filter."""
    topic = state["topic"]
    papers = state["all_papers"]

    scored = score_papers(papers, topic)

    # Generate embeddings
    embeddings = generate_embeddings(scored)

    return {
        "scored_papers": scored,
        "embeddings": embeddings,
        "status": f"Scored and filtered to {len(scored)} papers",
        "log": [f"⭐ Kept {len(scored)} papers after relevance scoring"],
    }


def build_graph(state: AgentState) -> dict:
    """Step 4: Build the knowledge graph."""
    papers = state["scored_papers"]
    embeddings = state.get("embeddings", {})

    # Get or create knowledge graph
    kg = state.get("knowledge_graph")
    if kg is None:
        kg = KnowledgeGraphBuilder()

    # Add papers
    kg.add_papers(papers)

    # Build edges
    kg.build_citation_edges()
    kg.build_similarity_edges(embeddings, threshold=SIMILARITY_THRESHOLD)
    kg.build_author_edges()

    # Extract and add topics
    paper_topics = extract_topics_batch(papers[:50], batch_size=5)  # Limit for speed
    kg.add_topic_nodes(paper_topics)

    stats = kg.get_stats()

    return {
        "knowledge_graph": kg,
        "status": f"Knowledge graph: {stats['paper_nodes']} papers, {stats['total_edges']} edges",
        "log": [
            f"🕸️ Knowledge graph built: {stats['paper_nodes']} papers, "
            f"{stats['citation_edges']} citation edges, "
            f"{stats['similarity_edges']} similarity edges, "
            f"{stats['topic_nodes']} topics"
        ],
    }


def analyze_graph(state: AgentState) -> dict:
    """Step 5: Run analysis algorithms on the graph."""
    kg = state["knowledge_graph"]
    topic = state["topic"]

    analysis = run_full_analysis(kg, topic)

    return {
        "analysis": analysis,
        "status": f"Analysis: {len(analysis.clusters)} clusters, {len(analysis.gaps)} gaps",
        "log": [
            f"📊 Analysis complete: {len(analysis.clusters)} clusters, "
            f"{len(analysis.key_papers)} key papers, "
            f"{len(analysis.gaps)} gaps, {len(analysis.trends)} trends. "
            f"Coverage: {analysis.coverage_score:.0%}"
        ],
    }


def generate_report_node(state: AgentState) -> dict:
    """Step 6: Generate the literature review report."""
    topic = state["topic"]
    papers = state["scored_papers"]
    analysis = state["analysis"]
    kg = state["knowledge_graph"]

    # Generate report
    report = generate_report(topic, papers, analysis)

    # Create interactive graph visualization
    partition = getattr(analysis, '_partition', {})
    pagerank = getattr(analysis, '_pagerank', {})
    graph_path = create_interactive_graph(
        kg.graph,
        partition=partition,
        pagerank=pagerank,
        paper_data=kg.papers,
    )

    return {
        "report": report,
        "status": "Report generated!",
        "log": [
            f"📝 Literature review generated: {len(report.sections)} sections",
            f"📊 Knowledge graph visualization saved",
        ],
    }


# ──────────────────────────────────────────────
# Router (Conditional Edge)
# ──────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["plan_search", "generate_report"]:
    """
    Decide whether to search more or generate the report.
    This is the core autonomous decision-making logic.
    """
    iteration = state.get("iteration", 0)
    papers = state.get("scored_papers", [])
    analysis = state.get("analysis")

    # Hard stop conditions
    if iteration >= MAX_ITERATIONS:
        console.print("[bold yellow]🛑 Max iterations reached. Generating report.[/bold yellow]")
        return "generate_report"

    if not analysis:
        console.print("[yellow]No analysis yet, need more data[/yellow]")
        return "plan_search"

    # Quality checks
    num_papers = len(papers)
    num_clusters = len(analysis.clusters)
    coverage = analysis.coverage_score

    console.print(f"\n[bold]🤔 Decision point:[/bold]")
    console.print(f"  Papers: {num_papers}/{MIN_PAPERS_FOR_ANALYSIS}")
    console.print(f"  Clusters: {num_clusters}/{MIN_CLUSTERS_REQUIRED}")
    console.print(f"  Coverage: {coverage:.0%}")

    if num_papers >= MIN_PAPERS_FOR_ANALYSIS and num_clusters >= MIN_CLUSTERS_REQUIRED and coverage >= 0.7:
        console.print("[bold green]✅ Sufficient data. Generating report.[/bold green]")
        return "generate_report"

    console.print("[bold cyan]🔄 Need more data. Searching again...[/bold cyan]")
    return "plan_search"


# ──────────────────────────────────────────────
# Helper: Refinement Queries
# ──────────────────────────────────────────────
def _get_refinement_queries(
    topic: str,
    analysis: AnalysisResult,
    searched: list[str],
    iteration: int,
) -> list[SearchQuery]:
    """Ask LLM what to search next based on current state."""
    prompt = STOPPING_DECISION_PROMPT.format(
        topic=topic,
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
        total_papers=len(analysis.key_papers) if analysis else 0,
        num_clusters=len(analysis.clusters) if analysis else 0,
        coverage_score=analysis.coverage_score if analysis else 0,
        new_papers_last=0,
        searched_queries=", ".join(searched[:10]),
        min_papers=MIN_PAPERS_FOR_ANALYSIS,
        min_clusters=MIN_CLUSTERS_REQUIRED,
    )

    try:
        text = call_gemini(prompt, max_tokens=1024, temperature=LLM_TEMPERATURE)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        new_queries = [
            SearchQuery(query_text=q, strategy="gap_driven")
            for q in data.get("new_queries", [])
        ]
        console.print(f"[cyan]Refinement: {data.get('reason', 'N/A')}[/cyan]")
        return new_queries

    except Exception as e:
        console.print(f"[yellow]Refinement query generation failed: {e}[/yellow]")
        return [
            SearchQuery(query_text=f"{topic} recent advances 2024", strategy="gap_driven"),
        ]


# ──────────────────────────────────────────────
# Build the LangGraph
# ──────────────────────────────────────────────
def build_agent_graph() -> StateGraph:
    """Construct and compile the agent state graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan_search", plan_search)
    workflow.add_node("execute_search", execute_search)
    workflow.add_node("score_and_filter", score_and_filter)
    workflow.add_node("build_graph", build_graph)
    workflow.add_node("analyze_graph", analyze_graph)
    workflow.add_node("generate_report", generate_report_node)

    # Define edges
    workflow.set_entry_point("plan_search")
    workflow.add_edge("plan_search", "execute_search")
    workflow.add_edge("execute_search", "score_and_filter")
    workflow.add_edge("score_and_filter", "build_graph")
    workflow.add_edge("build_graph", "analyze_graph")

    # Conditional: continue searching or generate report
    workflow.add_conditional_edges(
        "analyze_graph",
        should_continue,
        {
            "plan_search": "plan_search",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
def run_scholar(topic: str) -> AgentState:
    """
    Run the full ScholAR pipeline.

    Args:
        topic: Research topic from the user

    Returns:
        Final agent state with report and all data
    """
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  🎓 ScholAR - Autonomous Research Intelligence Agent[/bold green]")
    console.print(f"[bold green]  Topic: {topic}[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]\n")

    agent = build_agent_graph()

    initial_state: AgentState = {
        "topic": topic,
        "queries": [],
        "all_papers": [],
        "scored_papers": [],
        "embeddings": {},
        "knowledge_graph": None,
        "analysis": None,
        "report": None,
        "iteration": 0,
        "status": "Starting...",
        "log": [f"🚀 Starting ScholAR for topic: {topic}"],
    }

    # Run the agent
    final_state = agent.invoke(initial_state)

    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  ✅ ScholAR Complete![/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")

    return final_state


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "transformer attention mechanisms"
    result = run_scholar(topic)

    if result.get("report"):
        print(f"\nReport topic: {result['report'].topic}")
        print(f"Total sections: {len(result['report'].sections)}")
        print(f"Papers analyzed: {result['report'].total_papers_analyzed}")
