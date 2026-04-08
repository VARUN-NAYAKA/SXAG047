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
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #533483, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #e94560;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #8892b0;
        margin-top: 0.5rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #2ecc71;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        display: inline-block;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .log-entry {
        padding: 8px 12px;
        margin: 4px 0;
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #533483;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #ccd6f6;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ccd6f6;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem;
    }

    .gap-card {
        background: rgba(233, 69, 96, 0.08);
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .rec-card {
        background: rgba(46, 204, 113, 0.08);
        border: 1px solid rgba(46, 204, 113, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        margin-top: 8px;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #e94560, #2ecc71);
    }

    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
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

    # API key — pre-loaded from .env, editable
    api_key = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY,
        type="password",
        help="Loaded from .env file. You can override it here.",
    )

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

col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "🔬 Enter a research topic or paper title",
        placeholder="e.g., LLM hallucination mitigation, transformer attention mechanisms...",
        label_visibility="collapsed",
    )
with col2:
    run_button = st.button(
        "🚀 Run ScholAR",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )


# ──────────────────────────────────────────────
# Run Agent
# ──────────────────────────────────────────────
if run_button and topic:
    active_key = api_key or GEMINI_API_KEY
    if not active_key:
        st.error("⚠️ Please provide your Gemini API Key in the sidebar or .env file.")
    else:
        os.environ["GEMINI_API_KEY"] = active_key

        st.session_state.running = True
        st.session_state.logs = []
        st.session_state.result = None

        progress_bar = st.progress(0, text="Initializing ScholAR...")

        try:
            from agents.orchestrator import run_scholar
            import core.config as config

            config.MAX_ITERATIONS = max_iterations
            config.MIN_PAPERS_FOR_ANALYSIS = min_papers
            config.SIMILARITY_THRESHOLD = similarity_threshold

            with st.spinner("🧠 ScholAR is thinking autonomously..."):
                result = run_scholar(topic)

            # Save to history
            try:
                from core.history import save_search
                save_search(topic, result)
            except Exception as e:
                print(f"History save error: {e}")

            st.session_state.result = result
            st.session_state.running = False
            progress_bar.progress(100, text="✅ Complete!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.running = False
            import traceback
            st.code(traceback.format_exc())


# ──────────────────────────────────────────────
# Display Results
# ──────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    report = result.get("report")
    analysis = result.get("analysis")
    papers = result.get("scored_papers", [])
    kg = result.get("knowledge_graph")

    if report:
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

            if analysis and analysis.gaps:
                for i, gap in enumerate(analysis.gaps, 1):
                    st.markdown(f"""
                    <div class="gap-card">
                        <strong>Gap {i}:</strong> {gap.description}<br/>
                        <small>Confidence: {gap.confidence:.0%} | 
                        Related: {', '.join(gap.related_topics[:3])}</small>
                    </div>
                    """, unsafe_allow_html=True)

                    if gap.suggested_directions:
                        with st.expander("Suggested Research Directions"):
                            for d in gap.suggested_directions:
                                st.markdown(f"- {d}")
            else:
                st.info("No research gaps identified in this analysis.")

        # ────── Tab 5: Top 3 Recommended Papers ──────
        with tab_recs:
            st.markdown('<div class="section-header">🏆 ScholAR\'s Top Picks</div>', unsafe_allow_html=True)
            st.markdown("*Based on relevance, citations, and graph centrality.*")

            if report.top_recommendations:
                for i, rec in enumerate(report.top_recommendations, 1):
                    medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "📄"
                    paper = rec.paper

                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.8rem;">{medal}</span>
                            <span style="color:#2ecc71; font-size:1.5rem; font-weight:bold;">{rec.confidence:.0f}%</span>
                        </div>
                        <h3 style="color:#ccd6f6; margin:0.5rem 0;">{paper.title}</h3>
                        <p style="color:#8892b0;">
                            {', '.join(a.name for a in paper.authors[:3])} 
                            {'et al.' if len(paper.authors) > 3 else ''} 
                            · {paper.year or 'N/A'} · {paper.citation_count} citations
                            {' · ' + paper.venue if paper.venue else ''}
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{rec.confidence}%;"></div>
                        </div>
                        <p style="color:#8892b0; font-size:0.85rem; margin-top:0.5rem;">
                            <strong>Why:</strong> {rec.reason}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Paper link
                    url = paper.get_download_url()
                    if url:
                        st.link_button(f"📎 Open Paper", url)
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
                <strong style="color: #ccd6f6;">{title}</strong>
                <p style="color: #8892b0; font-size: 0.85rem; margin-top: 0.5rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
