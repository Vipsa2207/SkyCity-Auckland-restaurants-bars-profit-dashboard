import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGO_PATH = os.path.join(BASE_DIR, "app", "static", "skycity_logo.jpg")

DARK_CSS = """
<style>
    .stApp { background-color: #0f1117; }

    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2a2d3e;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

    h1,h2,h3 { color: #e8eaf6 !important; }
    hr { border-color: #2a2d3e; margin: 0.6rem 0; }

    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }

     [data-testid="stVerticalBlock"] { gap: 0.6rem !important; }
    
    [data-testid="stPageLink"] { border-radius: 8px; padding: 2px 4px; margin-bottom: 2px; }
    [data-testid="stPageLink"]:hover { background-color: #242838; }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem !important;
        margin-top: 0 !important;
    }

    /* ── KPI Cards ─────────────────────────────────────────────── */
    .kpi-card {
        background: #161927;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #9aa0b4;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f4f5f9;
        line-height: 1.1;
    }
    .kpi-caption { font-size: 0.8rem; margin-top: 6px; font-weight: 500; }
    .kpi-caption.positive { color: #06d6a0; }
    .kpi-caption.negative { color: #ef476f; }
    .kpi-caption.neutral  { color: #ffd166; }
    .kpi-caption.info     { color: #7c83fd; }

    /* ── Section Header Bar ────────────────────────────────────── */
    .section-bar {
        border-left: 4px solid #7c83fd;
        padding-left: 12px;
        margin: 1.6rem 0 0.8rem 0;
    }
    .section-bar span {
        font-size: 0.78rem;
        color: #c9cce0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    /* ── Chart Container ───────────────────────────────────────── */
    .chart-card {
        background: #161927;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 14px 16px 6px 16px;
    }

    /* ── Insight Cards ─────────────────────────────────────────── */
    .insight-card {
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .insight-card.risk {
        background: rgba(239,71,111,0.08);
        border: 1px solid rgba(239,71,111,0.35);
    }
    .insight-card.positive {
        background: rgba(6,214,160,0.08);
        border: 1px solid rgba(6,214,160,0.35);
    }
    .insight-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 4px; }
    .insight-title.risk { color: #ef476f; }
    .insight-title.positive { color: #06d6a0; }
    .insight-body { color: #c9cce0; }
    .insight-body b { color: #ffffff; }
</style>
"""

def render_header():
    """Logo + title in header, vertically aligned with title baseline."""
    col_logo, col_title = st.columns([1.3, 7], vertical_alignment="bottom")
    with col_logo:
        st.image(LOGO_PATH, width=130)
    with col_title:
        st.markdown("""
        <div style= 'padding-top: 10px;'padding-bottom: 6px;'>
            <h1 style='font-size:1.9rem; color:#ffffff; margin:0; font-weight:700; line-height:1.3;'>
                SkyCity Auckland
            </h1>
            <p style='color:#7c83fd; font-size:0.9rem; margin:2px 0 0 0; font-weight:500;'>
                Restaurant & Bar · Profit Intelligence Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

def render_sidebar_filters(df, get_cuisines, get_segments, get_subregions):
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.image(LOGO_PATH, use_container_width=True)

        st.markdown("---")
        st.markdown("""
<div style='margin-bottom: 0.5rem;'>
    <p style='color:#6b7280; font-size:0.72rem; text-transform:uppercase;
              letter-spacing:0.08em; font-weight:700; margin-bottom:0.6rem;'>
        Navigation
    </p>
</div>
""", unsafe_allow_html=True)

        nav_style = """
<style>
[data-testid="stPageLink"] > a {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #c9cce0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    transition: background 0.15s !important;
}
[data-testid="stPageLink"] > a:hover {
    background: #242838 !important;
    color: #ffffff !important;
}
[data-testid="stPageLink"][aria-current="page"] > a {
    background: #242838 !important;
    color: #7c83fd !important;
    font-weight: 600 !important;
}
</style>
"""
        st.markdown(nav_style, unsafe_allow_html=True)

        st.page_link("main.py",                     label="Home",              icon=":material/home:")
        st.page_link("pages/1_Overview.py",         label="Overview",          icon=":material/bar_chart:")
        st.page_link("pages/2_Profit_Predictor.py", label="Profit Predictor",  icon=":material/model_training:")
        st.page_link("pages/3_WhatIf_Simulator.py", label="What-If Simulator", icon=":material/tune:")
        st.page_link("pages/4_Optimization.py",     label="Optimization",      icon=":material/target:")
        st.page_link("pages/5_Sensitivity.py",      label="Sensitivity",       icon=":material/show_chart:")
        st.markdown("---")
        st.markdown("### Filters")
        cuisines   = st.multiselect("Cuisine Type", get_cuisines(df))
        segments   = st.multiselect("Segment",      get_segments(df))
        subregions = st.multiselect("Subregion",    get_subregions(df))

    return cuisines, segments, subregions

# ── Reusable Components ───────────────────────────────────────────────────

def kpi_card(label, value, caption=None, caption_type="info"):
    """Renders a single styled KPI card. caption_type: positive/negative/neutral/info"""
    caption_html = f"<div class='kpi-caption {caption_type}'>{caption}</div>" if caption else ""
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        {caption_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(text):
    """Section header with colored left bar, like 'CHURNED VS RETAINED'."""
    st.markdown(f"<div class='section-bar'><span>{text}</span></div>", unsafe_allow_html=True)

def page_title(title, subtitle=None):
    """Bold page title with optional subtitle — sits below the header."""
    sub_html = f"<p style='color:#9aa0b4; font-size:0.95rem; margin:4px 0 0 0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom: 1rem;'>
        <h2 style='font-size:1.6rem; color:#ffffff; font-weight:700; margin:0; line-height:1.3;'>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def insight_card(title, body, kind="risk"):
    """Insight callout box. kind: 'risk' (red) or 'positive' (green)."""
    icon = "⚠️" if kind == "risk" else "✅"
    st.markdown(f"""
    <div class='insight-card {kind}'>
        <div class='insight-title {kind}'>{icon} {title}</div>
        <div class='insight-body'>{body}</div>
    </div>
    """, unsafe_allow_html=True)