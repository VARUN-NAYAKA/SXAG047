"""
ScholAR - Streamlit Web Dashboard
🎓 Autonomous Research Intelligence Agent
"""

import streamlit as st
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from core.config import DATA_DIR

# Load API key: Streamlit Cloud secrets > .env > empty
def _get_api_key():
    """Get API key from Streamlit secrets (cloud) or .env (local)."""
    # 1. Try Streamlit Cloud secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            os.environ["GEMINI_API_KEY"] = key
            return key
    except Exception:
        pass
    # 2. Fall back to .env / environment variable
    return os.environ.get("GEMINI_API_KEY", "")

GEMINI_API_KEY = _get_api_key()

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ScholAR — Research Intelligence Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f8f9fc, #eef1f8, #e8ecf4);
        font-family: 'Inter', sans-serif;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #e94560, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #5a6785;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.15);
        border-color: #6C63FF;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6C63FF;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 3px solid #6C63FF;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem;
    }

    .gap-card {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-left: 4px solid #e94560;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #1e293b;
    }

    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6C63FF;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e2e8f0;
        margin-top: 10px;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6C63FF, #2ecc71);
    }

    .history-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        color: #334155;
    }

    .log-entry {
        padding: 8px 12px;
        margin: 4px 0;
        background: #f1f5f9;
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #334155;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None
if "logs" not in st.session_state:
    st.session_state.logs = []


# ──────────────────────────────────────────────
# Sidebar (collapsible)
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API key status indicator
    if GEMINI_API_KEY:
        st.success("✅ API Key loaded", icon="🔑")
    else:
        st.warning("⚠️ No default API Key found.")

    # Optional custom API key override
    custom_api_key = st.text_input(
        "🔑 Custom Gemini API Key (optional)",
        type="password",
        placeholder="Paste your own key to override default",
        help="If provided, this key will be used instead of the default one.",
    )

    # Resolve: custom > pre-provided
    ACTIVE_API_KEY = custom_api_key.strip() if custom_api_key and custom_api_key.strip() else GEMINI_API_KEY

    st.markdown("---")
    st.markdown("### 🔧 Advanced Settings")

    max_iterations = st.slider("Max Iterations", 1, 5, 3)
    min_papers = st.slider("Min Papers for Analysis", 10, 100, 30)
    similarity_threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.7, 0.05)

    # ── Search History ──
    st.markdown("---")
    st.markdown("### 📜 Search History")
    try:
        from core.history import get_history
        history = get_history(limit=10)
        if history:
            for item in history:
                ts = item['timestamp'][:16].replace('T', ' ')
                st.markdown(f"""
                <div class="history-item">
                    <strong>{item['topic']}</strong><br/>
                    <small>{ts} · {item['papers_count']} papers · {item['clusters_count']} clusters</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No searches yet.")
    except Exception:
        st.caption("No searches yet.")

    st.markdown("---")
    st.markdown(
        "### 📚 About ScholAR\n"
        "An autonomous research intelligence agent that:\n"
        "- 🔍 Searches multiple academic APIs\n"
        "- 🕸️ Builds knowledge graphs\n"
        "- 📊 Detects patterns & gaps\n"
        "- 📝 Generates literature reviews"
    )


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎓 ScholAR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Autonomous Research Intelligence Agent — '
    'From topic to literature review in minutes</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Search Mode Selection
# ──────────────────────────────────────────────
search_mode = st.radio(
    "Choose research mode:",
    ["🔬 Topic Search", "👤 Author Search", "📄 PDF Upload"],
    horizontal=True,
    label_visibility="collapsed",
)

topic = ""
uploaded_pdf = None
author_name = ""

if search_mode == "🔬 Topic Search":
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "🔬 Enter a research topic",
            placeholder="e.g., LLM hallucination mitigation, transformer attention mechanisms...",
            label_visibility="collapsed",
        )
    with col2:
        run_button = st.button("🚀 Run ScholAR", type="primary", use_container_width=True, disabled=st.session_state.running)

elif search_mode == "👤 Author Search":
    col1, col2 = st.columns([4, 1])
    with col1:
        author_name = st.text_input(
            "👤 Enter researcher name",
            placeholder="e.g., Geoffrey Hinton, Yann LeCun, Fei-Fei Li...",
            label_visibility="collapsed",
        )
    with col2:
        run_button = st.button("🔍 Search Author", type="primary", use_container_width=True, disabled=st.session_state.running)

elif search_mode == "📄 PDF Upload":
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_pdf = st.file_uploader(
            "Upload a research paper (PDF)",
            type=["pdf"],
            help="Upload a paper and ScholAR will analyze it, find related work, and generate a literature review.",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("📊 Analyze Paper", type="primary", use_container_width=True, disabled=st.session_state.running)
else:
    run_button = False


# ──────────────────────────────────────────────
# Run Agent
# ──────────────────────────────────────────────
should_run = run_button and (topic or author_name or uploaded_pdf)

if should_run:
    if not ACTIVE_API_KEY:
        st.error("⚠️ No Gemini API Key found. Add one in the sidebar, .env file, or Streamlit Cloud Secrets.")
    else:
        os.environ["GEMINI_API_KEY"] = ACTIVE_API_KEY

        st.session_state.running = True
        st.session_state.logs = []
        st.session_state.result = None

        # ── Resolve the effective topic based on mode ──
        effective_topic = topic

        # MODE 1: Author Search → list all papers by the author
        if search_mode == "👤 Author Search" and author_name:
            with st.spinner(f"🔍 Searching for papers by **{author_name}**..."):
                try:
                    from apis.semantic_scholar import search_author_papers
                    author_papers = search_author_papers(author_name, limit=100)

                    if not author_papers:
                        st.error(f"No papers found for author '{author_name}'. Try a different spelling.")
                        st.session_state.running = False
                        st.stop()

                    # Store author results directly
                    st.session_state.result = {
                        "mode": "author",
                        "author_name": author_name,
                        "author_papers": author_papers,
                    }
                    st.session_state.running = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Author search failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.running = False
                    st.stop()

        # MODE 2: PDF Upload → deep paper analysis + find similar work
        elif search_mode == "📄 PDF Upload" and uploaded_pdf:
            try:
                from core.pdf_utils import extract_text_from_pdf
                from core.paper_analyzer import analyze_paper_deeply, generate_comparison_report

                with st.spinner("📄 Reading PDF..."):
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    if not pdf_text:
                        st.error("Could not extract text from this PDF. It may be scanned/image-based.")
                        st.session_state.running = False
                        st.stop()

                with st.spinner("🧠 Analyzing paper in depth..."):
                    paper_analysis = analyze_paper_deeply(pdf_text)

                # Use the extracted search queries to find similar papers
                search_queries = paper_analysis.get("search_queries", [])
                if not search_queries:
                    search_queries = [paper_analysis.get("title", "research paper")]

                similar_papers = []
                with st.spinner("🔍 Finding similar papers in the literature..."):
                    from apis.semantic_scholar import search_papers as ss_search
                    for query in search_queries[:3]:
                        try:
                            results = ss_search(query, limit=10)
                            similar_papers.extend(results)
                        except Exception:
                            pass
                    # Deduplicate
                    seen = set()
                    unique_papers = []
                    for p in similar_papers:
                        if p.paper_id not in seen:
                            seen.add(p.paper_id)
                            unique_papers.append(p)
                    similar_papers = unique_papers[:20]

                comparison = ""
                if similar_papers:
                    with st.spinner("📊 Generating comparison report..."):
                        comparison = generate_comparison_report(pdf_text, similar_papers)

                st.session_state.result = {
                    "mode": "pdf",
                    "paper_analysis": paper_analysis,
                    "similar_papers": similar_papers,
                    "comparison_report": comparison,
                    "pdf_text": pdf_text,
                }
                st.session_state.running = False
                st.rerun()

            except Exception as e:
                st.error(f"PDF processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.running = False
                st.stop()

        # MODE 3: Topic Search → normal pipeline
        elif effective_topic:
            progress_bar = st.progress(0, text="Initializing ScholAR...")

            try:
                from agents.orchestrator import run_scholar
                import core.config as config

                config.MAX_ITERATIONS = max_iterations
                config.MIN_PAPERS_FOR_ANALYSIS = min_papers
                config.SIMILARITY_THRESHOLD = similarity_threshold

                with st.spinner("🧠 ScholAR is thinking autonomously..."):
                    result = run_scholar(effective_topic)

                # Save to history
                try:
                    from core.history import save_search
                    save_search(effective_topic, result)
                except Exception as e:
                    print(f"History save error: {e}")

                result["mode"] = "topic"
                st.session_state.result = result
                st.session_state.running = False
                progress_bar.progress(100, text="✅ Complete!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.running = False
                import traceback
                st.code(traceback.format_exc())
        else:
            st.error("Please enter a topic, author name, or upload a PDF.")
            st.session_state.running = False


# ──────────────────────────────────────────────
# Display Results
# ──────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    result_mode = result.get("mode", "topic")

    # ════════════════════════════════════════════
    # AUTHOR MODE DISPLAY
    # ════════════════════════════════════════════
    if result_mode == "author":
        author_papers = result.get("author_papers", [])
        author_name_display = result.get("author_name", "Unknown")

        st.success(f"✅ Found **{len(author_papers)}** papers by **{author_name_display}**")

        # Sort by year descending
        author_papers.sort(key=lambda p: p.year or 0, reverse=True)

        # Stats
        st.markdown("---")
        total_citations = sum(p.citation_count for p in author_papers)
        years = [p.year for p in author_papers if p.year]
        year_range = f"{min(years)}–{max(years)}" if years else "N/A"

        cols = st.columns(4)
        stat_data = [
            ("📄", "Total Papers", len(author_papers)),
            ("📊", "Total Citations", total_citations),
            ("📅", "Year Range", year_range),
            ("⭐", "Avg Citations", total_citations // max(len(author_papers), 1)),
        ]
        for col, (icon, label, value) in zip(cols, stat_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f'<div class="section-header">📄 Papers by {author_name_display}</div>', unsafe_allow_html=True)

        import pandas as pd
        paper_data = []
        for p in author_papers:
            url = p.get_download_url()
            paper_data.append({
                "Title": p.title,
                "Year": p.year or "N/A",
                "Citations": p.citation_count,
                "Venue": (p.venue or "N/A")[:40],
                "Link": url if url else "—",
            })

        df = pd.DataFrame(paper_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Link": st.column_config.LinkColumn("📥 Link", display_text="Open"),
            },
        )

    # ════════════════════════════════════════════
    # PDF MODE DISPLAY
    # ════════════════════════════════════════════
    elif result_mode == "pdf":
        paper_analysis = result.get("paper_analysis", {})
        similar_papers = result.get("similar_papers", [])
        comparison = result.get("comparison_report", "")

        st.success(f"✅ Paper analyzed: **{paper_analysis.get('title', 'Unknown')}**")

        tab_summary, tab_analysis_pc, tab_findings, tab_trends, tab_compare, tab_similar = st.tabs([
            "📝 Summary", "⚖️ Pros & Cons", "🔬 Key Findings",
            "📈 Trends", "🔄 Comparison", "📚 Similar Papers",
        ])

        with tab_summary:
            st.markdown(f'<div class="section-header">📝 Paper Summary</div>', unsafe_allow_html=True)
            st.markdown(paper_analysis.get("summary", "No summary available."))
            if paper_analysis.get("methodology"):
                st.markdown("### 🔧 Methodology")
                st.markdown(paper_analysis["methodology"])
            if paper_analysis.get("contributions"):
                st.markdown("### 🎯 Key Contributions")
                for c in paper_analysis["contributions"]:
                    st.markdown(f"- {c}")

        with tab_analysis_pc:
            st.markdown(f'<div class="section-header">⚖️ Strengths & Weaknesses</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Strengths")
                for pro in paper_analysis.get("pros", []):
                    st.markdown(f"""
                    <div style="background:#f0fdf4; border-left:4px solid #22c55e; padding:10px 14px; margin:6px 0; border-radius:0 8px 8px 0; color:#166534;">
                        ✓ {pro}
                    </div>
                    """, unsafe_allow_html=True)
                if not paper_analysis.get("pros"):
                    st.info("No specific strengths identified.")
            with col2:
                st.markdown("### ❌ Weaknesses")
                for con in paper_analysis.get("cons", []):
                    st.markdown(f"""
                    <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:10px 14px; margin:6px 0; border-radius:0 8px 8px 0; color:#991b1b;">
                        ✗ {con}
                    </div>
                    """, unsafe_allow_html=True)
                if not paper_analysis.get("cons"):
                    st.info("No specific weaknesses identified.")

        with tab_findings:
            st.markdown(f'<div class="section-header">🔬 Key Findings</div>', unsafe_allow_html=True)
            for i, finding in enumerate(paper_analysis.get("key_findings", []), 1):
                st.markdown(f"**{i}.** {finding}")
            if paper_analysis.get("implications"):
                st.markdown("### 🌍 Practical Implications")
                for imp in paper_analysis["implications"]:
                    st.markdown(f"- {imp}")

        with tab_trends:
            st.markdown(f'<div class="section-header">📈 Trends & Evolution</div>', unsafe_allow_html=True)
            for trend in paper_analysis.get("trends", []):
                st.markdown(f"- {trend}")
            if not paper_analysis.get("trends"):
                st.info("No specific trends mentioned in this paper.")

        with tab_compare:
            st.markdown(f'<div class="section-header">🔄 Comparison with Related Work</div>', unsafe_allow_html=True)
            if comparison:
                st.markdown(comparison)
            else:
                st.info("Comparison report not available.")

        with tab_similar:
            st.markdown(f'<div class="section-header">📚 Similar Papers Found</div>', unsafe_allow_html=True)
            if similar_papers:
                import pandas as pd
                similar_data = []
                for p in similar_papers[:20]:
                    url = p.get_download_url()
                    similar_data.append({
                        "Title": p.title[:80],
                        "Year": p.year or "N/A",
                        "Citations": p.citation_count,
                        "Venue": (p.venue or "N/A")[:30],
                        "Link": url if url else "—",
                    })
                df = pd.DataFrame(similar_data)
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Link": st.column_config.LinkColumn("📥 Link", display_text="Open"),
                    },
                )
            else:
                st.info("No similar papers found.")

    # ════════════════════════════════════════════
    # TOPIC MODE DISPLAY (existing)
    # ════════════════════════════════════════════
    elif result_mode == "topic":
        report = result.get("report")
        analysis = result.get("analysis")
        papers = result.get("scored_papers", [])
        kg = result.get("knowledge_graph")

    if result_mode == "topic" and result.get("report"):
        report = result["report"]
        analysis = result.get("analysis")
        papers = result.get("scored_papers", [])
        kg = result.get("knowledge_graph")
        st.success(f"✅ Analysis complete! Reviewed **{report.total_papers_analyzed}** papers.")

        # ── Metrics Row ──
        st.markdown("---")
        cols = st.columns(5)
        metrics = [
            ("📄", "Papers", report.total_papers_analyzed),
            ("🕸️", "Nodes", kg.num_nodes if kg else 0),
            ("🔗", "Edges", kg.num_edges if kg else 0),
            ("📊", "Clusters", len(analysis.clusters) if analysis else 0),
            ("🔍", "Gaps", len(analysis.gaps) if analysis else 0),
        ]
        for col, (icon, label, value) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Download Report Button ──
        pdf_path = DATA_DIR / "scholar_report.pdf"
        if not pdf_path.exists():
            try:
                from agents.report_agent import export_report_to_pdf
                export_report_to_pdf(report)
            except Exception:
                pass

        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Report",
                    data=f.read(),
                    file_name=f"ScholAR_{report.topic[:30].replace(' ', '_')}_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        # ── Tabs ──
        tab_report, tab_graph, tab_analysis, tab_gaps, tab_recs, tab_papers = st.tabs([
            "📝 Literature Review",
            "🕸️ Knowledge Graph",
            "📊 Analysis",
            "🔍 Research Gaps",
            "🏆 Top Picks",
            "📄 Papers",
        ])

        # ────── Tab 1: Literature Review ──────
        with tab_report:
            st.markdown('<div class="section-header">📝 Literature Review</div>', unsafe_allow_html=True)
            for section in report.sections:
                if section.title.startswith("Research Theme:"):
                    continue
                with st.expander(f"**{section.title}**", expanded=(section.order <= 2)):
                    if section.content:
                        st.markdown(section.content)
                    else:
                        st.info("Content generation in progress.")

        # ────── Tab 2: Knowledge Graph + Clusters ──────
        with tab_graph:
            st.markdown('<div class="section-header">🕸️ Interactive Knowledge Graph</div>', unsafe_allow_html=True)
            graph_file = DATA_DIR / "knowledge_graph.html"
            if graph_file.exists():
                st.components.v1.html(
                    graph_file.read_text(encoding="utf-8"),
                    height=700,
                    scrolling=True,
                )
            else:
                st.info("Knowledge graph visualization not available.")

            if analysis and analysis.clusters:
                st.markdown('<div class="section-header">🧩 Research Theme Clusters</div>', unsafe_allow_html=True)
                sorted_clusters = sorted(analysis.clusters, key=lambda c: c.cluster_id or 0)

                for cluster in sorted_clusters:
                    with st.expander(f"**{cluster.name}** — {len(cluster.paper_ids)} papers", expanded=False):
                        st.markdown(f"**Keywords:** {', '.join(cluster.keyword_list)}")
                        st.markdown(f"**Description:** {cluster.description}")

                        cluster_section = None
                        for section in report.sections:
                            if section.title == f"Research Theme: {cluster.name}":
                                cluster_section = section
                                break

                        if cluster_section and cluster_section.content:
                            st.markdown("---")
                            st.markdown(cluster_section.content)
                        else:
                            paper_dict = {p.paper_id: p for p in papers}
                            cluster_papers = [paper_dict[pid] for pid in cluster.paper_ids[:5] if pid in paper_dict]
                            if cluster_papers:
                                st.markdown("**Key papers:**")
                                for p in cluster_papers:
                                    st.markdown(f"- {p.title} ({p.year}) — {p.citation_count} citations")

        # ────── Tab 3: Analysis ──────
        with tab_analysis:
            st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                trend_file = DATA_DIR / "trend_chart.html"
                if trend_file.exists():
                    st.components.v1.html(trend_file.read_text(encoding="utf-8"), height=450)
                else:
                    st.info("No trend chart available — try increasing max iterations.")

            with col2:
                cluster_file = DATA_DIR / "cluster_chart.html"
                if cluster_file.exists():
                    st.components.v1.html(cluster_file.read_text(encoding="utf-8"), height=450)
                else:
                    st.info("No cluster chart available.")

            citation_file = DATA_DIR / "citation_chart.html"
            if citation_file.exists():
                st.components.v1.html(citation_file.read_text(encoding="utf-8"), height=500)

            if analysis and analysis.trends:
                st.markdown("### 📈 Detected Trends")
                for trend in analysis.trends:
                    st.markdown(
                        f"**{trend.topic}** ({trend.direction.upper()}) — "
                        f"Strength: {trend.strength:.0%}\n\n{trend.description}"
                    )

        # ────── Tab 4: Research Gaps ──────
        with tab_gaps:
            st.markdown('<div class="section-header">🔍 Research Gaps & Future Directions</div>', unsafe_allow_html=True)
            st.markdown("*Algorithmically detected gaps in the research landscape — areas where more work is needed.*")

            if analysis and analysis.gaps:
                for i, gap in enumerate(analysis.gaps, 1):
                    confidence_color = "#e94560" if gap.confidence > 0.7 else "#f59e0b" if gap.confidence > 0.5 else "#6C63FF"
                    st.markdown(f"""
                    <div class="gap-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <strong style="color:#1e293b; font-size:1.1rem;">Gap {i}</strong>
                                <span style="background:{confidence_color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; margin-left:8px;">
                                    {gap.confidence:.0%} confidence
                                </span>
                            </div>
                        </div>
                        <p style="color:#334155; margin:0.8rem 0; font-size:0.95rem; line-height:1.6;">
                            {gap.description}
                        </p>
                        <div style="margin-top:0.5rem;">
                            <strong style="color:#64748b; font-size:0.85rem;">Related Topics:</strong>
                            <span style="color:#6C63FF; font-size:0.85rem;">{', '.join(gap.related_topics[:4])}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if gap.suggested_directions:
                        with st.expander(f"💡 Suggested Research Directions for Gap {i}"):
                            for d in gap.suggested_directions:
                                st.markdown(f"→ {d}")
            else:
                st.info("No research gaps identified in this analysis.")

        # ────── Tab 5: Top 3 Recommended Papers ──────
        with tab_recs:
            st.markdown('<div class="section-header">🏆 ScholAR\'s Top Picks</div>', unsafe_allow_html=True)
            st.markdown(
                "*These papers are ranked using a composite score of **relevance** (topic match), "
                "**citations** (community impact), and **PageRank centrality** (structural importance in the knowledge graph).*"
            )

            if report.top_recommendations:
                for i, rec in enumerate(report.top_recommendations, 1):
                    medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "📄"
                    paper = rec.paper

                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:2rem;">{medal}</span>
                            <span style="color:#6C63FF; font-size:1.6rem; font-weight:bold;">{rec.confidence:.0f}%</span>
                        </div>
                        <h3 style="color:#1e293b; margin:0.5rem 0; font-size:1.15rem;">{paper.title}</h3>
                        <p style="color:#64748b; font-size:0.9rem;">
                            {', '.join(a.name for a in paper.authors[:3])} 
                            {'et al.' if len(paper.authors) > 3 else ''} 
                            · {paper.year or 'N/A'} · {paper.citation_count} citations
                            {' · ' + paper.venue if paper.venue else ''}
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{rec.confidence}%;"></div>
                        </div>
                        <p style="color:#334155; font-size:0.9rem; margin-top:0.8rem; line-height:1.5;">
                            <strong>Why ScholAR recommends this:</strong> {rec.reason}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Paper link
                    url = paper.get_download_url()
                    if url:
                        st.link_button(f"📎 Open Paper", url)
                    
                    # Abstract preview
                    if paper.abstract:
                        with st.expander("📖 Read Abstract"):
                            st.markdown(paper.abstract)
            else:
                st.info("No recommendations available.")

        # ────── Tab 6: Papers with Download Links ──────
        with tab_papers:
            st.markdown('<div class="section-header">📄 Collected Papers</div>', unsafe_allow_html=True)

            if papers:
                import pandas as pd

                paper_data = []
                for p in papers[:100]:
                    url = p.get_download_url()
                    paper_data.append({
                        "Title": p.title[:80],
                        "Year": p.year or "N/A",
                        "Citations": p.citation_count,
                        "Venue": (p.venue or "N/A")[:30],
                        "Relevance": p.relevance_score,
                        "Source": p.source_api,
                        "Link": url if url else "—",
                    })

                df = pd.DataFrame(paper_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Relevance": st.column_config.ProgressColumn(
                            "Relevance", format="%.3f", min_value=0, max_value=1,
                        ),
                        "Link": st.column_config.LinkColumn("📥 Link", display_text="Open"),
                    },
                )

        # ── Agent Log ──
        with st.expander("🤖 Agent Reasoning Log"):
            for log_entry in result.get("log", []):
                st.markdown(f'<div class="log-entry">{log_entry}</div>', unsafe_allow_html=True)

elif not st.session_state.running:
    st.markdown("---")
    st.markdown("### 💡 Try these example topics:")

    example_cols = st.columns(3)
    examples = [
        ("🧠", "LLM Hallucination Mitigation", "Techniques to reduce hallucinations in large language models"),
        ("🔬", "CRISPR Gene Therapy", "Clinical applications of CRISPR-Cas9 in genetic disease treatment"),
        ("🤖", "Multi-Agent Reinforcement Learning", "Cooperative and competitive multi-agent RL systems"),
    ]

    for col, (icon, title, desc) in zip(example_cols, examples):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: left; cursor: pointer;">
                <div style="font-size: 2rem;">{icon}</div>
                <strong style="color: #1e293b;">{title}</strong>
                <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
