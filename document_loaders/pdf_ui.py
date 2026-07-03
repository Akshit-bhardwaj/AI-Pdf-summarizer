"""
app.py — Streamlit Frontend for PDF AI Assistant
=================================================
Consumes the same LangChain components used in pdf.py:
  - PyPDFLoader        (langchain_community.document_loaders)
  - ChatMistralAI      (langchain_mistralai)
  - ChatPromptTemplate (langchain_core.prompts)

Backend logic (pdf.py) is NOT modified or imported directly because it
is a script (runs top-level code on import with a hardcoded PDF path).
This app reproduces the exact same pipeline in a frontend-friendly,
session-aware, and interactive way.

Design: MeetIQ-inspired — deep dark background, warm orange/amber
radial glows, glassmorphism cards, bold italic hero typography.
"""

import os
import io
import time
import tempfile
import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# 0. Environment & Page Configuration
# ─────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "PDF AI Assistant powered by Mistral AI & LangChain",
    },
)

# ─────────────────────────────────────────────
# 1. Custom CSS — MeetIQ-Inspired Dark Theme
#    Orange/Amber radial glow, glassmorphism,
#    bold italic typography
# ─────────────────────────────────────────────
def inject_css() -> None:
    """Inject premium MeetIQ-inspired CSS styles."""
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,700;1,800&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-base:        #07080f;
        --bg-deep:        #0b0c17;
        --bg-card:        rgba(255,255,255,0.04);
        --bg-card-hover:  rgba(255,255,255,0.07);
        --bg-input:       rgba(255,255,255,0.05);
        --bg-sidebar:     #090a14;

        --orange-core:    #f97316;
        --orange-bright:  #fb923c;
        --orange-dim:     #ea6a0a;
        --orange-glow:    rgba(249,115,22,0.35);
        --orange-subtle:  rgba(249,115,22,0.12);
        --orange-border:  rgba(249,115,22,0.30);

        --amber:          #fbbf24;
        --amber-glow:     rgba(251,191,36,0.20);

        --text-primary:   #f0ece8;
        --text-secondary: #a89e95;
        --text-muted:     #5c554e;

        --success:        #22c55e;
        --success-bg:     rgba(34,197,94,0.12);
        --warning:        #f59e0b;
        --warning-bg:     rgba(245,158,11,0.12);
        --error:          #ef4444;
        --error-bg:       rgba(239,68,68,0.12);

        --radius-card:    18px;
        --radius-btn:     12px;
        --radius-pill:    100px;

        --shadow-card:    0 8px 40px rgba(0,0,0,0.5);
        --shadow-glow:    0 0 60px rgba(249,115,22,0.18);
        --transition:     all 0.22s cubic-bezier(.4,0,.2,1);
    }

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Deep dark app background with subtle radial glow */
    .stApp {
        background-color: var(--bg-base);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -5%,
                rgba(249,115,22,0.18) 0%,
                transparent 65%),
            radial-gradient(ellipse 40% 30% at 80% 90%,
                rgba(251,191,36,0.06) 0%,
                transparent 60%);
        color: var(--text-primary);
        min-height: 100vh;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Main content padding ── */
    .block-container {
        max-width: 860px;
        padding: 2rem 2rem 5rem !important;
    }

    /* ═══════════════════════════════
       SIDEBAR
    ═══════════════════════════════ */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid rgba(249,115,22,0.12);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.2rem;
    }

    /* Sidebar logo/brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .sidebar-brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--orange-dim), var(--orange-bright));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px var(--orange-glow);
        flex-shrink: 0;
    }
    .sidebar-brand-text {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.3px;
    }
    .sidebar-brand-sub {
        font-size: 0.68rem;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }

    /* Sidebar section labels */
    .sidebar-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--orange-core);
        margin: 1.6rem 0 0.6rem;
        padding-left: 0.1rem;
    }

    /* Sidebar divider */
    .s-divider {
        border: none;
        border-top: 1px solid rgba(249,115,22,0.10);
        margin: 1rem 0;
    }

    /* Sidebar doc info card */
    .doc-info-card {
        background: var(--bg-card);
        border: 1px solid var(--orange-border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        font-size: 0.8rem;
        line-height: 2;
        color: var(--text-secondary);
    }
    .doc-info-card b { color: var(--text-primary); }

    /* ═══════════════════════════════
       HERO SECTION — MeetIQ style
    ═══════════════════════════════ */
    .hero-wrap {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        margin-bottom: 2rem;
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(249,115,22,0.15);
        background: linear-gradient(180deg,
            rgba(249,115,22,0.08) 0%,
            rgba(249,115,22,0.02) 50%,
            transparent 100%);
    }
    /* Glowing orb behind hero — matches MeetIQ's central orange burst */
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -30px;
        left: 50%;
        transform: translateX(-50%);
        width: 500px;
        height: 200px;
        background: radial-gradient(ellipse,
            rgba(249,115,22,0.40) 0%,
            rgba(251,191,36,0.12) 40%,
            transparent 70%);
        filter: blur(30px);
        pointer-events: none;
        z-index: 0;
    }
    /* Subtle star dots */
    .hero-wrap::after {
        content: '✦  ✦   ✦    ✦  ✦';
        position: absolute;
        top: 1.5rem;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 0.4rem;
        color: rgba(249,115,22,0.3);
        letter-spacing: 20px;
        pointer-events: none;
    }
    .hero-badge {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(249,115,22,0.15);
        border: 1px solid rgba(249,115,22,0.35);
        border-radius: var(--radius-pill);
        padding: 0.28rem 0.85rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--orange-bright);
        letter-spacing: 0.3px;
        margin-bottom: 1.2rem;
    }
    .hero-badge span { color: var(--amber); }
    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -1.5px;
        color: var(--text-primary);
        margin: 0 0 0.5rem;
    }
    /* The italic orange accent word — like "Understands" in MeetIQ */
    .hero-title em {
        font-style: italic;
        background: linear-gradient(90deg, var(--orange-core), var(--amber));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        position: relative;
        z-index: 1;
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0 auto;
        max-width: 480px;
        line-height: 1.6;
    }

    /* ═══════════════════════════════
       GLASSMORPHISM CARDS
    ═══════════════════════════════ */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid rgba(249,115,22,0.18);
        border-radius: var(--radius-card);
        padding: 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: var(--transition);
    }
    .glass-card:hover {
        border-color: rgba(249,115,22,0.35);
        box-shadow: var(--shadow-glow);
    }

    /* Section headings inside cards */
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-sub {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0 0 1.2rem;
    }

    /* ═══════════════════════════════
       STAT CARDS (like 95%, 98%, 92%)
    ═══════════════════════════════ */
    .stat-cards-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 1.25rem 0;
    }
    .stat-card {
        background: rgba(249,115,22,0.08);
        border: 1px solid rgba(249,115,22,0.22);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .stat-card-icon { font-size: 1.1rem; margin-bottom: 0.2rem; }
    .stat-card-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--orange-bright);
        line-height: 1.1;
    }
    .stat-card-label {
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* ═══════════════════════════════
       FILE BADGE
    ═══════════════════════════════ */
    .file-badge {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        background: rgba(249,115,22,0.07);
        border: 1px solid rgba(249,115,22,0.20);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .file-badge-icon {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--orange-dim), var(--orange-bright));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 16px var(--orange-glow);
        flex-shrink: 0;
    }
    .file-badge-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    .file-badge-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 0.1rem 0 0;
    }

    /* ═══════════════════════════════
       STATUS BANNERS
    ═══════════════════════════════ */
    .status-banner {
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.84rem;
        font-weight: 500;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-success {
        background: var(--success-bg);
        border: 1px solid rgba(34,197,94,0.30);
        color: var(--success);
    }
    .status-warning {
        background: var(--warning-bg);
        border: 1px solid rgba(245,158,11,0.30);
        color: var(--warning);
    }
    .status-error {
        background: var(--error-bg);
        border: 1px solid rgba(239,68,68,0.30);
        color: var(--error);
    }

    /* ═══════════════════════════════
       CHAT BUBBLES
    ═══════════════════════════════ */
    .chat-wrap {
        display: flex;
        flex-direction: column;
        gap: 1.4rem;
        padding: 0.25rem 0;
    }
    .chat-row {
        display: flex;
        align-items: flex-end;
        gap: 0.7rem;
    }
    .chat-row.user-row { flex-direction: row-reverse; }

    .chat-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .av-user {
        background: linear-gradient(135deg, var(--orange-core), var(--orange-bright));
        color: #fff;
        box-shadow: 0 4px 12px var(--orange-glow);
    }
    .av-ai {
        background: rgba(249,115,22,0.10);
        border: 1px solid var(--orange-border);
        color: var(--orange-bright);
    }
    .bubble {
        max-width: 76%;
        border-radius: 18px;
        padding: 0.85rem 1.1rem;
        font-size: 0.875rem;
        line-height: 1.65;
    }
    .bubble-user {
        background: linear-gradient(135deg, var(--orange-dim) 0%, var(--orange-bright) 100%);
        color: #fff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 20px rgba(249,115,22,0.35);
    }
    .bubble-ai {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(249,115,22,0.18);
        color: var(--text-primary);
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(8px);
    }
    .chat-ts {
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
        padding: 0 0.2rem;
    }
    .user-row .chat-ts { text-align: right; }

    /* Empty chat state */
    .chat-empty {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-muted);
    }
    .chat-empty-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
        opacity: 0.5;
    }
    .chat-empty-text {
        font-size: 0.875rem;
        opacity: 0.7;
    }

    /* ═══════════════════════════════
       DIVIDER
    ═══════════════════════════════ */
    .divider {
        border: none;
        border-top: 1px solid rgba(249,115,22,0.12);
        margin: 1.5rem 0;
    }

    /* ═══════════════════════════════
       STREAMLIT WIDGET OVERRIDES
    ═══════════════════════════════ */

    /* Text Input */
    .stTextInput > div > div > input {
        background: var(--bg-input) !important;
        border: 1.5px solid rgba(249,115,22,0.22) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 1rem !important;
        transition: var(--transition) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--orange-core) !important;
        box-shadow: 0 0 0 3px var(--orange-subtle) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* Primary Buttons — orange gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--orange-dim), var(--orange-bright)) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-btn) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
        padding: 0.6rem 1.4rem !important;
        cursor: pointer !important;
        transition: var(--transition) !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(249,115,22,0.30) !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 28px rgba(249,115,22,0.50) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* Secondary button */
    .secondary-btn > button {
        background: transparent !important;
        border: 1.5px solid var(--orange-border) !important;
        color: var(--orange-bright) !important;
        box-shadow: none !important;
    }
    .secondary-btn > button:hover {
        background: var(--orange-subtle) !important;
        box-shadow: none !important;
    }

    /* Danger button */
    .danger-btn > button {
        background: var(--error-bg) !important;
        border: 1.5px solid rgba(239,68,68,0.35) !important;
        color: var(--error) !important;
        box-shadow: none !important;
    }
    .danger-btn > button:hover {
        background: rgba(239,68,68,0.20) !important;
        box-shadow: none !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--orange-dim), var(--orange-bright)) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-btn) !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(249,115,22,0.30) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(249,115,22,0.04) !important;
        border: 2px dashed rgba(249,115,22,0.28) !important;
        border-radius: 16px !important;
        transition: var(--transition) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--orange-core) !important;
        background: rgba(249,115,22,0.07) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-input) !important;
        border-color: rgba(249,115,22,0.22) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* Slider */
    [data-testid="stSlider"] .stSlider > div > div {
        background: rgba(249,115,22,0.15) !important;
    }

    /* Toggle */
    [data-testid="stToggle"] span {
        background-color: rgba(249,115,22,0.20) !important;
    }
    [data-testid="stToggle"] [aria-checked="true"] span {
        background-color: var(--orange-core) !important;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--orange-dim), var(--amber)) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: rgba(249,115,22,0.15) !important;
        border-radius: 4px !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid rgba(249,115,22,0.18) !important;
        border-radius: 14px !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(249,115,22,0.32) !important;
    }
    [data-testid="stExpanderToggleIcon"] { color: var(--orange-bright) !important; }

    /* Spinner */
    [data-testid="stSpinner"] > div {
        border-top-color: var(--orange-core) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(249,115,22,0.25);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(249,115,22,0.45);
    }

    /* Markdown headings */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        letter-spacing: -0.5px !important;
    }

    /* Caption */
    .stCaption { color: var(--text-muted) !important; }

    /* Selectbox text color */
    [data-testid="stSelectbox"] span {
        color: var(--text-primary) !important;
    }

    /* Slider labels */
    [data-testid="stSlider"] p {
        color: var(--text-secondary) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. Session State Initialisation
# ─────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all session-state keys on first run."""
    defaults = {
        "chat_history": [],     # list of {"role": "user"|"ai", "content": str, "ts": str}
        "docs":         None,   # loaded LangChain Document objects
        "doc_name":     None,   # uploaded filename
        "doc_pages":    0,      # total page count
        "doc_size_kb":  0.0,    # file size in KB
        "pdf_ready":    False,  # True once PDF is loaded & ready
        "model_name":   "mistral-small-latest",
        "temperature":  0.3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
# 3. Backend Helpers — mirrors pdf.py exactly
# ─────────────────────────────────────────────
def load_pdf(tmp_path: str) -> list:
    """
    Load PDF using PyPDFLoader — exact same class used in pdf.py.
    Returns a list of LangChain Document objects (one per page).
    """
    loader = PyPDFLoader(tmp_path)
    return loader.load()


def build_prompt_template() -> ChatPromptTemplate:
    """
    Builds the ChatPromptTemplate from pdf.py.
    System: 'You are the best model to summarize the user data'
    Human:  '{data}'
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are the best model to summarize the user data"),
        ("human", "{data}"),
    ])


def build_llm(model_name: str, temperature: float) -> ChatMistralAI:
    """
    Instantiates ChatMistralAI — same class and model as pdf.py.
    """
    return ChatMistralAI(model=model_name, temperature=temperature)


def query_llm(question: str, docs: list, model_name: str, temperature: float) -> str:
    """
    Core inference pipeline (mirrors pdf.py):
      1. Concatenate all page content into one context string.
      2. Format the ChatPromptTemplate with context + question.
      3. Invoke ChatMistralAI and return response.content.
    """
    full_context = "\n\n".join(
        f"[Page {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    combined_input = f"Question: {question}\n\nDocument Content:\n{full_context}"

    template = build_prompt_template()
    llm      = build_llm(model_name, temperature)
    prompt   = template.format_messages(data=combined_input)
    response = llm.invoke(prompt)
    return response.content


# ─────────────────────────────────────────────
# 4. UI Component Renderers
# ─────────────────────────────────────────────
def render_hero() -> None:
    """MeetIQ-inspired hero with orange radial glow and italic accent."""
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">✦ <span>New</span> &nbsp;Powered by Mistral AI</div>
        <h1 class="hero-title">
            AI That Reads, <em>Understands,</em><br>And Answers.
        </h1>
        <p class="hero-sub">
            Upload any PDF and ask questions in plain English —
            instant answers powered by Mistral AI &amp; LangChain.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render sidebar with branding, settings, and document info."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">📄</div>
            <div>
                <div class="sidebar-brand-text">PDF AI</div>
                <div class="sidebar-brand-sub">ASSISTANT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

        # ── Model Settings ───────────────────
        st.markdown('<p class="sidebar-label">🧠 Model</p>', unsafe_allow_html=True)
        model_options = [
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
        ]
        st.session_state.model_name = st.selectbox(
            "Mistral Model",
            model_options,
            index=model_options.index(st.session_state.model_name),
            help="Larger models = more capable but slower.",
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Lower = focused · Higher = creative",
        )

        # ── Document Info ────────────────────
        if st.session_state.pdf_ready:
            st.markdown('<p class="sidebar-label">📋 Document</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="doc-info-card">
                📁 <b>{st.session_state.doc_name}</b><br>
                📄 <b>Pages:</b> {st.session_state.doc_pages}<br>
                💾 <b>Size:</b> {st.session_state.doc_size_kb:.1f} KB<br>
                💬 <b>Messages:</b> {len(st.session_state.chat_history)}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="sidebar-label">⚙️ Actions</p>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button("🗑️ Clear", key="sb_clear"):
                    st.session_state.chat_history = []
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
                if st.button("📂 New", key="sb_new"):
                    _reset_state()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Footer ───────────────────────────
        st.markdown('<hr class="s-divider">', unsafe_allow_html=True)
        st.caption("🔗 [Mistral AI](https://mistral.ai) · [LangChain](https://langchain.com)")


def render_stat_cards(pages: int, size_kb: float, chats: int) -> None:
    """Render MeetIQ-style floating stat cards."""
    st.markdown(f"""
    <div class="stat-cards-row">
        <div class="stat-card">
            <div class="stat-card-icon">📄</div>
            <div class="stat-card-val">{pages}</div>
            <div class="stat-card-label">Pages Loaded</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-icon">💾</div>
            <div class="stat-card-val">{size_kb:.0f}<span style="font-size:0.8rem;font-weight:600"> KB</span></div>
            <div class="stat-card-label">File Size</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-icon">💬</div>
            <div class="stat-card-val">{chats}</div>
            <div class="stat-card-label">Messages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_file_badge(name: str, size_kb: float) -> None:
    """Render the uploaded file info badge."""
    st.markdown(f"""
    <div class="file-badge">
        <div class="file-badge-icon">📄</div>
        <div>
            <p class="file-badge-name">{name}</p>
            <p class="file-badge-meta">{size_kb:.1f} KB &nbsp;·&nbsp; PDF Document &nbsp;·&nbsp; Ready to process</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status(kind: str, message: str) -> None:
    """Render a coloured status banner (success / warning / error)."""
    icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    st.markdown(
        f'<div class="status-banner status-{kind}">'
        f'{icons.get(kind, "ℹ️")} {message}</div>',
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Render full conversation as orange-themed chat bubbles."""
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-empty">
            <div class="chat-empty-icon">💬</div>
            <div class="chat-empty-text">
                No conversation yet.<br>Ask your first question below!
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-row user-row">
                <div>
                    <div class="bubble bubble-user">{msg["content"]}</div>
                    <div class="chat-ts">{msg["ts"]}</div>
                </div>
                <div class="chat-avatar av-user">U</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            ai_text = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(f"""
            <div class="chat-row">
                <div class="chat-avatar av-ai">🤖</div>
                <div>
                    <div class="bubble bubble-ai">{ai_text}</div>
                    <div class="chat-ts">{msg["ts"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_document_preview(docs: list) -> None:
    """Expander with page-by-page document preview."""
    with st.expander("📖 Document Preview", expanded=False):
        selected_page = st.selectbox(
            "Page",
            list(range(1, len(docs) + 1)),
            format_func=lambda x: f"Page {x}",
            key="preview_page",
        )
        content = docs[selected_page - 1].page_content
        st.markdown(f"""
        <div style="
            background: rgba(249,115,22,0.04);
            border: 1px solid rgba(249,115,22,0.15);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            font-size: 0.81rem;
            line-height: 1.75;
            white-space: pre-wrap;
            max-height: 360px;
            overflow-y: auto;
            color: #b8ad9e;
            font-family: 'Inter', monospace;
        ">{content[:3000]}{"…" if len(content) > 3000 else ""}</div>
        """, unsafe_allow_html=True)
        st.caption(f"Page {selected_page} of {len(docs)}  ·  {len(content):,} characters")


def _reset_state() -> None:
    """Reset all document and chat state."""
    st.session_state.update({
        "chat_history": [],
        "docs":         None,
        "doc_name":     None,
        "doc_pages":    0,
        "doc_size_kb":  0.0,
        "pdf_ready":    False,
    })


def _ts() -> str:
    """Current time as HH:MM for chat timestamps."""
    return datetime.datetime.now().strftime("%H:%M")


# ─────────────────────────────────────────────
# 5. Main Application
# ─────────────────────────────────────────────
def main() -> None:
    init_session_state()
    inject_css()
    render_sidebar()
    render_hero()

    # ══════════════════════════════════════════
    # PHASE 1 — Upload & Process PDF
    # ══════════════════════════════════════════
    if not st.session_state.pdf_ready:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-title">📤 Upload Your PDF</div>
        <div class="card-sub">Drag &amp; drop or browse — we'll extract every page instantly.</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="PDF upload",
            type=["pdf"],
            accept_multiple_files=False,
            key="pdf_uploader",
            help="Only PDF files are supported. Recommended max size: 50 MB.",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            size_kb    = len(file_bytes) / 1024

            # Validate non-empty
            if size_kb < 1:
                render_status("error", "File appears empty or corrupt. Please try another PDF.")
                return

            # File info card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_file_badge(uploaded_file.name, size_kb)

            if st.button("⚡ Load & Process PDF", key="process_btn"):
                bar = st.progress(0, text="Initialising…")
                try:
                    bar.progress(15, text="📥 Saving file to disk…")
                    time.sleep(0.15)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    bar.progress(42, text="📄 Loading PDF with PyPDFLoader…")
                    docs = load_pdf(tmp_path)

                    if not docs:
                        render_status("error",
                            "No text could be extracted. The PDF may be scanned / image-based.")
                        bar.empty()
                        return

                    bar.progress(78, text="🔗 Preparing Mistral AI pipeline…")
                    time.sleep(0.25)

                    st.session_state.docs        = docs
                    st.session_state.doc_name    = uploaded_file.name
                    st.session_state.doc_pages   = len(docs)
                    st.session_state.doc_size_kb = size_kb
                    st.session_state.pdf_ready   = True

                    os.unlink(tmp_path)

                    bar.progress(100, text="✅ Ready!")
                    time.sleep(0.35)
                    bar.empty()
                    st.rerun()

                except FileNotFoundError:
                    render_status("error", "Could not create temp file. Check disk permissions.")
                    bar.empty()
                except Exception as exc:
                    render_status("error", f"Failed to load PDF: {exc}")
                    bar.empty()

            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # PHASE 2 — Chat Interface
    # ══════════════════════════════════════════
    else:
        docs     = st.session_state.docs
        doc_name = st.session_state.doc_name

        # ── Success status ───────────────────
        render_status(
            "success",
            f"<b>{doc_name}</b> loaded — {st.session_state.doc_pages} pages ready for Q&A."
        )

        # ── MeetIQ-style stat cards ──────────
        render_stat_cards(
            st.session_state.doc_pages,
            st.session_state.doc_size_kb,
            len(st.session_state.chat_history),
        )

        # ── Document preview expander ────────
        render_document_preview(docs)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Conversation ─────────────────────
        st.markdown("""
        <div class="card-title" style="margin-bottom:1rem;">💬 Conversation</div>
        """, unsafe_allow_html=True)

        with st.container():
            render_chat_history()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Question input form ──────────────
        with st.form(key="question_form", clear_on_submit=True):
            col_q, col_s = st.columns([5, 1])
            with col_q:
                question = st.text_input(
                    label="question",
                    placeholder="Ask anything about this PDF…",
                    label_visibility="collapsed",
                    key="q_input",
                )
            with col_s:
                submitted = st.form_submit_button("Send 🚀")

        # Process question
        if submitted:
            q = question.strip()
            if not q:
                render_status("warning", "Please type a question before sending.")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": q, "ts": _ts()}
                )
                with st.spinner("🤖 Mistral AI is thinking…"):
                    try:
                        answer = query_llm(
                            question=q,
                            docs=docs,
                            model_name=st.session_state.model_name,
                            temperature=st.session_state.temperature,
                        )
                        st.session_state.chat_history.append(
                            {"role": "ai", "content": answer, "ts": _ts()}
                        )
                    except ValueError as ve:
                        render_status("error", f"Configuration error: {ve}")
                    except ConnectionError:
                        render_status("error", "Network error — check your internet connection.")
                    except Exception as exc:
                        render_status("error", f"AI error: {exc}")
                st.rerun()

        # ── Bottom action bar ────────────────
        if st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            col_dl, col_cl, col_nw = st.columns([3, 1, 1])

            # Download last AI response
            last_ai = next(
                (m["content"] for m in reversed(st.session_state.chat_history)
                 if m["role"] == "ai"),
                None,
            )
            with col_dl:
                if last_ai:
                    report = io.StringIO()
                    report.write("PDF AI Assistant — Response Export\n")
                    report.write(f"Document : {doc_name}\n")
                    report.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    report.write("=" * 60 + "\n\n")
                    report.write(last_ai)
                    st.download_button(
                        label="⬇️ Download Last Response",
                        data=report.getvalue(),
                        file_name=f"ai_response_{datetime.date.today()}.txt",
                        mime="text/plain",
                        key="dl_resp",
                    )

            with col_cl:
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button("🗑️ Clear", key="main_clear"):
                    st.session_state.chat_history = []
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with col_nw:
                st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
                if st.button("📂 New PDF", key="main_new"):
                    _reset_state()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
