"""
Customer Segmentation Dashboard — Animated UI, Fixed Dark Mode & Sidebar
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# SESSION STATE
# ======================================================================
if "dark" not in st.session_state:
    st.session_state.dark = False
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"

# ======================================================================
# THEME VARIABLES
# ======================================================================
if st.session_state.dark:
    BG_APP       = "linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)"
    BG_APP_SOLID = "#0f172a"
    BG_CARD      = "#1e293b"
    BG_CARD_2    = "#334155"
    TEXT_MAIN    = "#f1f5f9"
    TEXT_MUTED   = "#94a3b8"
    BORDER       = "#334155"
    SIDEBAR_BG   = "#0f172a"
    HOVER_ACCENT = "rgba(59,130,246,0.18)"
    CHART_BG     = "rgba(0,0,0,0)"
else:
    BG_APP       = "linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)"
    BG_APP_SOLID = "#f8fafc"
    BG_CARD      = "#ffffff"
    BG_CARD_2    = "#f8fafc"
    TEXT_MAIN    = "#0f172a"
    TEXT_MUTED   = "#64748b"
    BORDER       = "#e2e8f0"
    SIDEBAR_BG   = "#ffffff"
    HOVER_ACCENT = "rgba(59,130,246,0.08)"
    CHART_BG     = "rgba(0,0,0,0)"

# ======================================================================
# GLOBAL STYLES
# ======================================================================
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    }}

    /* ---------- Force backgrounds everywhere ---------- */
    .stApp, .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > section,
    .main, .block-container {{
        background: {BG_APP} !important;
        background-attachment: fixed !important;
        color: {TEXT_MAIN} !important;
    }}
    [data-testid="stHeader"], header[data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
    }}
    .block-container {{
        padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1350px;
        animation: pageIn 0.6s ease-out;
    }}
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p, .stText,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {{
        color: {TEXT_MAIN};
    }}
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .stCaption, small {{
        color: {TEXT_MUTED} !important;
    }}

    /* ---------- Keyframes ---------- */
    @keyframes pageIn {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }}
        50%      {{ box-shadow: 0 0 0 12px rgba(59,130,246,0); }}
    }}
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50%      {{ background-position: 100% 50%; }}
    }}
    @keyframes floatOrb {{
        from {{ transform: rotate(0deg); }}
        to   {{ transform: rotate(360deg); }}
    }}
    @keyframes blink {{ 50% {{ opacity: 0; }} }}

    h1, h2, h3, h4, h5, h6 {{ color: {TEXT_MAIN} !important; }}

    /* ---------- METRIC CARDS ---------- */
    div[data-testid="stMetric"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 16px; padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden;
        animation: fadeUp 0.5s ease-out backwards;
    }}
    div[data-testid="stMetric"]::before {{
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        transform: scaleX(0); transform-origin: left;
        transition: transform 0.4s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -8px rgba(15,23,42,0.15);
        border-color: #cbd5e1;
    }}
    div[data-testid="stMetric"]:hover::before {{ transform: scaleX(1); }}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{
        color: {TEXT_MUTED} !important;
        font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
        color: {TEXT_MAIN} !important;
        font-weight: 700; font-size: 1.75rem;
    }}

    /* ---------- SIDEBAR BASE ---------- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        background: {SIDEBAR_BG} !important;
        color: {TEXT_MAIN} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN}; }}
    section[data-testid="stSidebar"] hr {{ border-color: {BORDER} !important; }}
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        color: {TEXT_MAIN} !important;
    }}

    /* ============================================================
       SIDEBAR RADIO — FIXED COMPACT LAYOUT
       ============================================================ */
    section[data-testid="stSidebar"] .stRadio > div,
    section[data-testid="stSidebar"] div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label,
    section[data-testid="stSidebar"] .stRadio label {{
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 6px 10px !important;
        margin: 0 !important;
        border-radius: 8px !important;
        gap: 8px !important;
        min-height: 36px !important;
        height: auto !important;
        line-height: 1.2 !important;
        transition: background 0.2s ease !important;
        background: transparent !important;
        width: 100% !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child,
    section[data-testid="stSidebar"] .stRadio label > div:first-child {{
        transform: scale(0.75) !important;
        margin: 0 !important;
        flex-shrink: 0 !important;
        flex-grow: 0 !important;
        width: auto !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child,
    section[data-testid="stSidebar"] .stRadio label > div:last-child {{
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.9rem !important;
        white-space: nowrap !important;
        color: {TEXT_MAIN} !important;
        flex-grow: 1 !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: {HOVER_ACCENT} !important;
    }}
    /* Kill the tall spacing between radio items */
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* ---------- BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
        border-radius: 10px; font-weight: 600;
        border: 1px solid {BORDER} !important;
        background: {BG_CARD} !important;
        color: {TEXT_MAIN} !important;
        padding: 10px 22px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {{
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(59,130,246,0.5);
    }}

    /* ---------- INPUTS ---------- */
    .stTextInput input, .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"] {{
        background: {BG_CARD} !important;
        color: {TEXT_MAIN} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    .stTextInput input::placeholder, .stNumberInput input::placeholder {{
        color: {TEXT_MUTED} !important;
        opacity: 0.7;
    }}
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {{
        background: {BG_CARD} !important;
        color: {TEXT_MAIN} !important;
    }}

    /* ---------- DATAFRAMES / ALERTS / SPINNERS ---------- */
    .stDataFrame {{
        border-radius: 12px; overflow: hidden;
        border: 1px solid {BORDER} !important;
        animation: fadeUp 0.5s ease-out;
        background: {BG_CARD} !important;
    }}
    .stAlert {{
        border-radius: 12px; animation: fadeUp 0.4s ease-out;
        color: {TEXT_MAIN} !important;
    }}
    [data-testid="stSpinner"] {{ color: {TEXT_MAIN} !important; }}

    /* ---------- HERO ---------- */
    .hero {{
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #1e40af 100%);
        background-size: 200% 200%;
        color: white; padding: 44px 48px;
        border-radius: 20px; margin-bottom: 28px;
        position: relative; overflow: hidden;
        animation: heroSlide 0.8s cubic-bezier(0.4, 0, 0.2, 1),
                   gradientShift 12s ease infinite;
    }}
    @keyframes heroSlide {{
        from {{ opacity: 0; transform: translateY(-20px) scale(0.98); }}
        to   {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    .hero::after {{
        content: ''; position: absolute;
        top: -50%; right: -50%; width: 100%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
        animation: floatOrb 20s linear infinite;
    }}
    .hero h1 {{
        color: white !important; margin: 0;
        font-size: 2.2rem; font-weight: 800;
        letter-spacing: -0.02em;
        position: relative; z-index: 1; min-height: 2.6rem;
    }}
    .hero .cursor {{
        display: inline-block; width: 3px; height: 2rem;
        background: #60a5fa; margin-left: 4px;
        vertical-align: -4px;
        animation: blink 1s step-end infinite;
    }}
    .hero p {{
        color: #cbd5e1 !important; margin-top: 10px;
        font-size: 1.1rem;
        position: relative; z-index: 1;
    }}
    .hero .badge {{
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        padding: 6px 14px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
        margin-top: 16px;
        position: relative; z-index: 1;
        border: 1px solid rgba(255,255,255,0.2);
        color: white !important;
    }}

    /* ---------- CHIP ---------- */
    .chip {{
        display: inline-block; padding: 6px 16px;
        border-radius: 999px; font-size: 0.9rem;
        font-weight: 700; color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: glowPulse 2s ease-in-out infinite;
    }}

    /* ---------- SECTION HEADERS ---------- */
    .section-anchor {{ scroll-margin-top: 90px; }}
    .section-header {{
        display: flex; align-items: center; gap: 10px;
        margin: 28px 0 16px 0;
        animation: fadeUp 0.5s ease-out;
    }}
    .section-header .icon {{ font-size: 1.5rem; }}
    .section-header .title {{
        font-size: 1.35rem; font-weight: 700;
        color: {TEXT_MAIN} !important;
        letter-spacing: -0.02em;
    }}
    .section-header .line {{
        flex: 1; height: 1px;
        background: linear-gradient(90deg, {BORDER}, transparent);
    }}

    /* ---------- TOP PROGRESS BAR ---------- */
    .top-progress {{
        position: fixed; top: 0; left: 0;
        height: 3px; width: 0%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        z-index: 9999;
        transition: width 0.4s ease, opacity 0.4s ease;
        box-shadow: 0 0 12px rgba(59,130,246,0.6);
    }}

    html {{ scroll-behavior: smooth; }}

    /* ---------- DARK MODE OVERRIDES ---------- */
    {'[data-testid="stAppViewContainer"] > .main { background: #0f172a !important; }' if st.session_state.dark else ''}
    {'section[data-testid="stSidebar"] { background: #0f172a !important; }' if st.session_state.dark else ''}
    {'div[data-testid="stMetric"] { background: #1e293b !important; border-color: #334155 !important; }' if st.session_state.dark else ''}
    {'div[data-testid="stMetricLabel"] * { color: #94a3b8 !important; }' if st.session_state.dark else ''}
    {'div[data-testid="stMetricValue"] * { color: #f1f5f9 !important; }' if st.session_state.dark else ''}
    {'section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child { color: #f1f5f9 !important; }' if st.session_state.dark else ''}
    {'h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }' if st.session_state.dark else ''}
    {'p, span, label, div { color: #f1f5f9 !important; }' if st.session_state.dark else ''}
    {'[data-testid="stCaptionContainer"] * { color: #94a3b8 !important; }' if st.session_state.dark else ''}
    {'[data-testid="stMarkdownContainer"] * { color: #f1f5f9 !important; }' if st.session_state.dark else ''}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# TOP PROGRESS BAR
# ======================================================================
components.html("""
<div id="topProgress" class="top-progress"></div>
<script>
  (function () {
    const bar = document.getElementById('topProgress');
    if (!bar) return;
    bar.style.width = '0%';
    bar.style.opacity = '1';
    let w = 0;
    const tick = setInterval(() => {
      w += 12 + Math.random() * 20;
      if (w >= 100) {
        w = 100;
        clearInterval(tick);
        setTimeout(() => { bar.style.opacity = '0'; }, 250);
      }
      bar.style.width = w + '%';
    }, 90);
  })();
</script>
<style>
.top-progress {
  position: fixed; top: 0; left: 0; height: 3px; width: 0%;
  background: linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899);
  z-index: 99999; transition: width .4s ease, opacity .4s ease;
  box-shadow: 0 0 12px rgba(59,130,246,.6);
}
</style>
""", height=0)

# ======================================================================
# HELPERS
# ======================================================================
def section_header(icon, title, anchor=None):
    anchor_attr = f'id="{anchor}"' if anchor else ""
    st.markdown(f"""
    <div class="section-anchor" {anchor_attr}></div>
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
        <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

def confetti():
    components.html("""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.0/dist/confetti.browser.min.js"></script>
    </head><body>
    <script>
      (function(){
        const duration = 1800;
        const end = Date.now() + duration;
        (function frame(){
          confetti({ particleCount: 4, angle: 60, spread: 55, origin: { x: 0 },
            colors: ['#3b82f6','#8b5cf6','#ec4899','#f59e0b'] });
          confetti({ particleCount: 4, angle: 120, spread: 55, origin: { x: 1 },
            colors: ['#3b82f6','#8b5cf6','#ec4899','#f59e0b'] });
          if (Date.now() < end) requestAnimationFrame(frame);
        })();
      })();
    </script>
    </body></html>
    """, height=0)

def typed_hero(title, subtitle, badge):
    components.html(f"""
    <html><head>
    <style>
      body {{ margin: 0; font-family: 'Inter','Segoe UI',sans-serif; }}
      .hero {{
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #1e40af 100%);
        background-size: 200% 200%; color: white;
        padding: 44px 48px; border-radius: 20px;
        position: relative; overflow: hidden;
        animation: gradientShift 12s ease infinite;
      }}
      @keyframes gradientShift {{
        0%,100% {{ background-position: 0% 50%; }}
        50%     {{ background-position: 100% 50%; }}
      }}
      .hero::after {{
        content:''; position:absolute; top:-50%; right:-50%;
        width:100%; height:200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
        animation: floatOrb 20s linear infinite;
      }}
      @keyframes floatOrb {{ from {{transform:rotate(0)}} to {{transform:rotate(360deg)}} }}
      .hero h1 {{
        margin: 0; font-size: 2.2rem; font-weight: 800;
        letter-spacing: -0.02em; color: white;
        position: relative; z-index: 1; min-height: 2.8rem;
      }}
      .cursor {{
        display: inline-block; width: 3px; height: 2rem;
        background: #60a5fa; margin-left: 4px;
        vertical-align: -4px;
        animation: blink 1s step-end infinite;
      }}
      @keyframes blink {{ 50% {{ opacity: 0; }} }}
      .hero p {{
        color: #cbd5e1; margin-top: 10px; font-size: 1.1rem;
        position: relative; z-index: 1; opacity: 0;
        animation: fadeUp 0.6s ease 2.4s forwards;
      }}
      .hero .badge {{
        display:inline-block; background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px); padding: 6px 14px;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        margin-top: 16px; position: relative; z-index: 1;
        border: 1px solid rgba(255,255,255,0.2); color: white;
        opacity: 0; animation: fadeUp 0.6s ease 2.8s forwards;
      }}
      @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}
    </style></head>
    <body>
    <div class="hero">
      <h1><span id="typed"></span><span class="cursor"></span></h1>
      <p>{subtitle}</p>
      <span class="badge">{badge}</span>
    </div>
    <script>
      const text = "{title}";
      let i = 0;
      const el = document.getElementById("typed");
      function type() {{
        if (i < text.length) {{
          el.textContent += text.charAt(i);
          i++;
          setTimeout(type, 45);
        }}
      }}
      type();
    </script>
    </body></html>
    """, height=210)

def animated_metric(label, value_str, help_text=None):
    import re
    m = re.search(r"([\d,]+\.?\d*)", str(value_str))
    if not m:
        st.metric(label, value_str, help=help_text)
        return

    numeric = float(m.group(1).replace(",", ""))
    prefix = value_str[:m.start(1)]
    suffix = value_str[m.end(1):]
    has_dec = "true" if "." in m.group(1) else "false"

    components.html(f"""
    <html><head>
    <style>
      body {{ margin: 0; font-family: 'Inter','Segoe UI',sans-serif; }}
      .card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 16px; padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
        position: relative; overflow: hidden;
        transition: all 0.35s ease;
      }}
      .card::before {{
        content:''; position:absolute; top:0;left:0;right:0;height:3px;
        background: linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899);
        transform: scaleX(0); transform-origin:left;
        transition: transform 0.4s ease;
      }}
      .card:hover {{ transform: translateY(-4px);
        box-shadow: 0 12px 24px -8px rgba(15,23,42,0.15);
        border-color:#cbd5e1; }}
      .card:hover::before {{ transform: scaleX(1); }}
      .lbl {{
        color: {TEXT_MUTED};
        font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 6px;
      }}
      .val {{ color: {TEXT_MAIN}; font-weight: 700; font-size: 1.75rem; }}
    </style></head>
    <body>
    <div class="card">
      <div class="lbl">{label}</div>
      <div class="val"><span id="n"></span></div>
    </div>
    <script>
      const target = {numeric};
      const prefix = "{prefix}";
      const suffix = "{suffix}";
      const hasDecimals = {has_dec};
      const duration = 900;
      const start = performance.now();
      function fmt(v) {{
        if (hasDecimals) return v.toFixed(1);
        return Math.round(v).toLocaleString('en-US');
      }}
      function step(now) {{
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        const v = target * eased;
        document.getElementById('n').textContent = prefix + fmt(v) + suffix;
        if (t < 1) requestAnimationFrame(step);
        else document.getElementById('n').textContent = prefix + fmt(target) + suffix;
      }}
      requestAnimationFrame(step);
    </script>
    </body></html>
    """, height=115)

# ======================================================================
# LOAD DATA
# ======================================================================
ART = Path("model_artifacts")

@st.cache_data(show_spinner=False)
def load_metadata():
    with open(ART / "metadata.json") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_customers():
    return pd.read_csv(ART / "customer_clusters.csv")

@st.cache_data(show_spinner=False)
def load_kmeans_profile():
    df = pd.read_csv(ART / "kmeans_profile.csv")
    return df.set_index(df.columns[0])

@st.cache_data(show_spinner=False)
def load_agg_profile():
    df = pd.read_csv(ART / "agg_profile.csv")
    return df.set_index(df.columns[0])

@st.cache_data(show_spinner=False)
def load_evaluation():
    return pd.read_csv(ART / "evaluation_table.csv")

@st.cache_data(show_spinner=False)
def load_kmeans_tuning():
    return pd.read_csv(ART / "kmeans_tuning.csv")

@st.cache_data(show_spinner=False)
def load_linkage():
    return pd.read_csv(ART / "linkage_comparison.csv")

@st.cache_data(show_spinner=False)
def load_pca():
    return pd.read_csv(ART / "pca_coords.csv")

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(ART / "scaler.pkl"), joblib.load(ART / "kmeans.pkl")

if not ART.exists() or not (ART / "metadata.json").exists():
    st.error("⚠️ Model artifacts not found. Please run the export cell in the notebook first.")
    st.stop()

with st.spinner("🔄 Loading model metadata..."):
    meta = load_metadata()
with st.spinner("📦 Loading customer clusters..."):
    customers = load_customers()
with st.spinner("📊 Building cluster profiles..."):
    kmeans_profile = load_kmeans_profile()
    agg_profile = load_agg_profile()
with st.spinner("📈 Loading evaluation metrics..."):
    evaluation = load_evaluation()
    kmeans_tuning = load_kmeans_tuning()
    linkage = load_linkage()
with st.spinner("🎨 Preparing PCA coordinates..."):
    pca_df = load_pca()
with st.spinner("🤖 Restoring trained model..."):
    scaler, kmeans_model = load_model()

FEATURES = meta["feature_columns"]
CLUSTER_COLORS = px.colors.qualitative.Set2

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 8px 0 16px 0;'>
        <div style='font-size: 2.2rem; margin-bottom: 4px;'>🛍️</div>
        <div style='font-weight: 800; font-size: 1.15rem; color:{TEXT_MAIN}; letter-spacing:-0.02em;'>
            Customer Segmentation
        </div>
        <div style='color:{TEXT_MUTED}; font-size: 0.8rem; margin-top: 4px;'>
            UCI Online Retail · ML Capstone
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    dark_on = st.toggle("🌙 Dark mode", value=st.session_state.dark, key="dark_toggle")
    if dark_on != st.session_state.dark:
        st.session_state.dark = dark_on
        st.rerun()

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Data & EDA",
            "🎯 Clustering Results",
            "📈 Model Comparison",
            "🗺️ Visualizations",
            "🧑‍💼 Customer Profiler",
            "🔮 Predict New Customer",
        ],
        label_visibility="collapsed",
        key="page_radio",
    )
    st.session_state.page = page

    st.markdown("---")
    st.markdown(f"""
    <div style='padding: 8px 0;'>
        <div style='color:{TEXT_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>
            Quick stats
        </div>
        <div style='display:flex; justify-content:space-between; padding: 4px 0; font-size: 0.85rem;'>
            <span style='color:{TEXT_MUTED};'>Customers</span>
            <span style='color:{TEXT_MAIN}; font-weight:600;'>{meta['unique_customers']:,}</span>
        </div>
        <div style='display:flex; justify-content:space-between; padding: 4px 0; font-size: 0.85rem;'>
            <span style='color:{TEXT_MUTED};'>Features</span>
            <span style='color:{TEXT_MAIN}; font-weight:600;'>{len(FEATURES)}</span>
        </div>
        <div style='display:flex; justify-content:space-between; padding: 4px 0; font-size: 0.85rem;'>
            <span style='color:{TEXT_MUTED};'>Best k</span>
            <span style='color:{TEXT_MAIN}; font-weight:600;'>{meta['selected_k']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def cluster_chip(cluster_id, palette=CLUSTER_COLORS):
    color = palette[int(cluster_id) % len(palette)]
    return f'<span class="chip" style="background:{color}">Cluster {cluster_id}</span>'

# ======================================================================
# PAGE: OVERVIEW
# ======================================================================
if page == "🏠 Overview":
    typed_hero(
        "Customer Segmentation Dashboard",
        "Discovering customer groups with K-Means and Agglomerative Clustering",
        "🚀 Powered by scikit-learn · Plotly · Streamlit",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: animated_metric("Customers Analysed", f"{meta['n_customers']:,}")
    with c2: animated_metric("Transaction Lines", f"{meta['transaction_rows_after_cleaning']:,}")
    with c3: animated_metric("Features Engineered", f"{len(FEATURES)}")
    with c4: animated_metric("Clusters (k)", f"{meta['selected_k']}")

    section_header("🎯", "Project Objectives", anchor="objectives")
    st.markdown("""
    - Identify groups of customers with **similar purchasing behaviour**.
    - Compare **K-Means** and **Agglomerative Hierarchical Clustering**.
    - Evaluate cluster quality with **Silhouette**, **Davies–Bouldin**, **Calinski–Harabasz**.
    - Visualise segments with **PCA**.
    """)

    section_header("📦", "Pipeline at a glance", anchor="pipeline")
    st.dataframe(pd.DataFrame({
        "Stage": ["1. Clean", "2. Feature Eng.", "3. Transform", "4. Cluster", "5. Evaluate"],
        "What": [
            "Removed nulls, duplicates, cancellations",
            "Built RFM + behavioural features per customer",
            "log1p + StandardScaler",
            "K-Means & Agglomerative",
            "Silhouette, DB, CH + PCA",
        ],
    }), use_container_width=True, hide_index=True)

# ======================================================================
# PAGE: DATA
# ======================================================================
elif page == "📊 Data & EDA":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>📊 Data & Exploratory Analysis</h2>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: animated_metric("Rows kept after cleaning", f"{meta['transaction_rows_after_cleaning']:,}")
    with c2: animated_metric("Unique customers", f"{meta['unique_customers']:,}")
    with c3: animated_metric("Engineered features", f"{len(FEATURES)}")

    section_header("🧬", "Customer-level features", anchor="features")
    st.dataframe(pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Days since last purchase (lower = more recent)",
            "Number of distinct invoices",
            "Total positive purchase value (£)",
            "Mean quantity per line",
            "Mean unit price (£)",
        ][:len(FEATURES)],
    }), use_container_width=True, hide_index=True)

    section_header("📐", "Feature distributions", anchor="distributions")
    selected = st.multiselect(
        "Choose features", FEATURES,
        default=[f for f in ["Monetary", "Frequency", "Recency"] if f in FEATURES],
    )
    if selected:
        cols = st.columns(min(len(selected), 3))
        for i, f in enumerate(selected):
            with cols[i % len(cols)]:
                with st.spinner(f"Plotting {f}..."):
                    fig = px.histogram(customers, x=f, nbins=50,
                                       color_discrete_sequence=["#3b82f6"])
                    fig.update_layout(
                        height=280, margin=dict(l=10, r=10, t=30, b=10),
                        showlegend=False, title=f,
                        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                        font=dict(color=TEXT_MAIN),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    section_header("🔍", "Raw data (first 100)", anchor="raw")
    st.dataframe(customers.head(100), use_container_width=True, hide_index=True)

# ======================================================================
# PAGE: CLUSTERING
# ======================================================================
elif page == "🎯 Clustering Results":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>🎯 Clustering Results</h2>",
                unsafe_allow_html=True)
    algo = st.radio("Algorithm", ["K-Means", "Agglomerative"], horizontal=True)

    if algo == "K-Means":
        profile = kmeans_profile.copy()
        sizes = {int(k): v for k, v in meta["kmeans_cluster_sizes"].items()}
    else:
        profile = agg_profile.copy()
        sizes = {int(k): v for k, v in meta["agg_cluster_sizes"].items()}

    section_header("📊", f"Cluster sizes — {algo}", anchor="sizes")
    size_df = pd.DataFrame({
        "Cluster": [f"Cluster {k}" for k in sizes.keys()],
        "Customers": list(sizes.values()),
    })
    with st.spinner("Rendering cluster sizes..."):
        fig = px.bar(size_df, x="Cluster", y="Customers", color="Cluster",
                     color_discrete_sequence=CLUSTER_COLORS, text="Customers")
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=350, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color=TEXT_MAIN),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_header("📋", f"Profiles — {algo}", anchor="profiles")
    st.dataframe(profile, use_container_width=True)

    section_header("🕸️", "Cluster profile radar", anchor="radar")
    radar_feats = [c for c in ["Recency_mean","Frequency_mean","Monetary_mean",
                                "AverageQuantity_mean","AverageUnitPrice_mean","TotalItems_mean"]
                   if c in profile.columns]
    if radar_feats and len(profile) > 1:
        with st.spinner("Building radar chart..."):
            norm = profile[radar_feats].copy()
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
            fig = go.Figure()
            for cid, row in norm.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row.values.tolist() + [row.values[0]],
                    theta=[f.replace("_mean","") for f in radar_feats] + [radar_feats[0].replace("_mean","")],
                    fill="toself", name=f"Cluster {cid}"))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                            gridcolor=BORDER)),
                height=500, margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor=CHART_BG, font=dict(color=TEXT_MAIN),
            )
            st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# PAGE: COMPARISON
# ======================================================================
elif page == "📈 Model Comparison":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>📈 Model Comparison & Evaluation</h2>",
                unsafe_allow_html=True)

    section_header("🏆", "Final metrics", anchor="metrics")
    st.dataframe(evaluation.round(4), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Silhouette** ↑")
        st.caption("Higher is better. Cluster separation quality.")
    with c2:
        st.markdown("**Davies–Bouldin** ↓")
        st.caption("Lower is better. Within vs between-cluster scatter.")
    with c3:
        st.markdown("**Calinski–Harabasz** ↑")
        st.caption("Higher is better. Between vs within dispersion.")

    section_header("📉", "Elbow & Silhouette", anchor="elbow")
    with st.spinner("Rendering tuning curve..."):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kmeans_tuning["k"], y=kmeans_tuning["Inertia"],
                                 mode="lines+markers", name="Inertia",
                                 line=dict(color="#3b82f6", width=3), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=kmeans_tuning["k"], y=kmeans_tuning["Silhouette"],
                                 mode="lines+markers", name="Silhouette",
                                 line=dict(color="#ef4444", width=3), marker=dict(size=9),
                                 yaxis="y2"))
        fig.update_layout(
            xaxis=dict(title="k (number of clusters)", dtick=1),
            yaxis=dict(title="Inertia"),
            yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
            height=440,
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color=TEXT_MAIN),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_header("🔗", "Linkage comparison", anchor="linkage")
    st.dataframe(linkage.round(4), use_container_width=True, hide_index=True)

# ======================================================================
# PAGE: VISUALIZATIONS
# ======================================================================
elif page == "🗺️ Visualizations":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>🗺️ Cluster Visualizations</h2>",
                unsafe_allow_html=True)
    st.caption(f"PCA explains **{meta['total_variance_explained']*100:.1f}%** of variance.")

    algo = st.radio("Colour by", ["KMeansCluster", "AgglomerativeCluster"], horizontal=True)
    if algo in pca_df.columns:
        d = pca_df.copy()
        d[algo] = d[algo].astype(str)
        with st.spinner("Rendering PCA scatter..."):
            fig = px.scatter(
                d, x="PC1", y="PC2", color=algo,
                color_discrete_sequence=CLUSTER_COLORS,
                opacity=0.65, height=600,
            )
            fig.update_traces(marker=dict(size=7, line=dict(width=0)))
            fig.update_layout(
                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                margin=dict(l=10, r=10, t=30, b=10),
                font=dict(color=TEXT_MAIN),
            )
            st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# PAGE: PROFILER
# ======================================================================
elif page == "🧑‍💼 Customer Profiler":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>🧑‍💼 Customer Profiler</h2>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        cf = st.multiselect("Filter by cluster",
                            sorted(customers["KMeansCluster"].unique()),
                            default=sorted(customers["KMeansCluster"].unique()))
    with c2:
        country_opts = sorted(customers["Country"].dropna().unique()) if "Country" in customers.columns else []
        country_filter = st.multiselect("Filter by Country", country_opts, default=[])
    with c3:
        rr = st.slider("Max recency (days)",
                       int(customers["Recency"].min()),
                       int(customers["Recency"].max()),
                       int(customers["Recency"].max()))

    with st.spinner("Applying filters..."):
        filtered = customers[customers["KMeansCluster"].isin(cf) & (customers["Recency"] <= rr)]
        if country_filter:
            filtered = filtered[filtered["Country"].isin(country_filter)]

    c1, c2, c3, c4 = st.columns(4)
    with c1: animated_metric("Customers matched", f"{len(filtered):,}")
    with c2: animated_metric("Avg Monetary", f"£{filtered['Monetary'].mean():,.2f}")
    with c3: animated_metric("Avg Frequency", f"{filtered['Frequency'].mean():.2f}")
    with c4: animated_metric("Avg Recency", f"{filtered['Recency'].mean():.1f} days")

    section_header("📋", "Matched customers", anchor="matched")
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)

    st.download_button("⬇️ Download filtered CSV",
                       filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_customers.csv", mime="text/csv")

# ======================================================================
# PAGE: PREDICT
# ======================================================================
elif page == "🔮 Predict New Customer":
    st.markdown(f"<h2 style='margin-top:0; color:{TEXT_MAIN};'>🔮 Predict Segment for a New Customer</h2>",
                unsafe_allow_html=True)
    st.caption("Enter a customer's purchasing summary; the model assigns them to a segment.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input("Recency (days)", 0, 500, 30)
            frequency = st.number_input("Frequency", 0, 500, 5)
        with c2:
            monetary = st.number_input("Monetary (£)", 0.0, 500000.0, 1500.0)
            avg_qty = st.number_input("Avg quantity", 0.0, 100000.0, 10.0)
        with c3:
            avg_price = st.number_input("Avg unit price (£)", 0.0, 10000.0, 3.0)
            total_items = st.number_input("Total items", 0, 500000, 500)
        submitted = st.form_submit_button("Analyse customer")

    if submitted:
        with st.spinner("🤖 Running model inference..."):
            row = pd.DataFrame([{
                "Recency": recency, "Frequency": frequency, "Monetary": monetary,
                "AverageQuantity": avg_qty, "AverageUnitPrice": avg_price, "TotalItems": total_items,
            }]).reindex(columns=FEATURES, fill_value=0)
            row_scaled = scaler.transform(np.log1p(row))
            cluster = int(kmeans_model.predict(row_scaled)[0])

        confetti()
       

        profile = kmeans_profile.copy()
        profile.index = profile.index.astype(int)
        if cluster in profile.index:
            rp = profile.loc[cluster]
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1: animated_metric("Avg Monetary", f"£{rp.get('Monetary_mean', 0):,.0f}")
            with pc2: animated_metric("Avg Frequency", f"{rp.get('Frequency_mean', 0):.1f}")
            with pc3: animated_metric("Avg Recency", f"{rp.get('Recency_mean', 0):.0f} days")
            with pc4: animated_metric("Segment size", f"{int(rp.get('CustomerCount', 0)):,}")

            section_header("📊", "Percentile within segment", anchor="percentile")
            pct = {}
            for f in FEATURES:
                seg_vals = customers[customers["KMeansCluster"] == cluster][f]
                pct[f] = (seg_vals <= row[f].values[0]).mean() * 100 if len(seg_vals) else 0
            pct_df = pd.DataFrame({
                "Feature": list(pct.keys()),
                "Percentile within segment": [f"{v:.0f}%" for v in pct.values()],
            })
            st.dataframe(pct_df, use_container_width=True, hide_index=True)

            st.caption(
                "Interpretation: lower Recency percentile = more recent purchase. "
                "Higher Frequency / Monetary percentile = more active and higher value customer."
            )