"""app.py — Streamlit frontend for the AI HR Recruitment Assistant.
Talks to the FastAPI backend (main.py) running the 4-agent CrewAI pipeline:
Screening -> Ranking -> Interview Questions -> Hiring Recommendation.
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Page config

st.set_page_config(page_title="AI HR Recruitment Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

DEFAULT_API_URL = "https://mandar-borhade-777-ai-hr-recruitment-backend.hf.space"
MAX_CANDIDATES = 5

# Color scheme

PRIMARY = "#6C2BD9"
PRIMARY_DARK = "#5522B0"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"
TEXT_MUTED = "#94A3B8"

VERDICT_STYLE = {
    "Strong Hire": {"color": "#FFFFFF", "bg": "#10B981", "emoji": "🌟"},
    "Hire":        {"color": "#FFFFFF", "bg": "#059669", "emoji": "✅"},
    "Maybe":       {"color": "#FFFFFF", "bg": "#F59E0B", "emoji": "🤔"},
    "No Hire":     {"color": "#FFFFFF", "bg": "#EF4444", "emoji": "⛔"},
}

RANK_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉"}

TECH_STACK = [
    {"icon": "⚡", "name": "Groq · Llama 3.3 70B", "detail": "High-performance LLM inference"},
    {"icon": "🤖", "name": "CrewAI", "detail": "Multi-agent orchestration"},
    {"icon": "🔍", "name": "LlamaIndex + ChromaDB", "detail": "RAG & vector search"},
    {"icon": "🚀", "name": "FastAPI", "detail": "High-speed backend API"},
    {"icon": "🎨", "name": "Streamlit", "detail": "Interactive frontend"},
]

AGENT_PIPELINE = [
    {"num": "01", "icon": "🔍", "name": "Screening Agent", "detail": "Extracts skills, experience, education"},
    {"num": "02", "icon": "📊", "name": "Ranking Agent", "detail": "Scores & ranks candidates (0-100)"},
    {"num": "03", "icon": "🎤", "name": "Interview Agent", "detail": "Generates questions for top 3"},
    {"num": "04", "icon": "✅", "name": "Recommendation Agent", "detail": "Final hire / no-hire verdict"},
]

# HTML helper — dedents multi-line HTML so Streamlit's Markdown parser
# doesn't mistake indented lines for a fenced code block (CommonMark rule).
def html_block(s: str) -> str:
    return "\n".join(line.strip() for line in s.strip("\n").splitlines())

# CSS

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .block-container {{ padding-top: 4rem; max-width: 1080px; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E2E8F0; box-shadow: 2px 0 8px rgba(15, 23, 42, 0.03); }}
    .sidebar-header {{ background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 1.1rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05); }}
    .sidebar-header .title {{ font-size: 1.25rem; font-weight: 800; color: {TEXT_PRIMARY}; }}
    .sidebar-header .subtitle {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 0.15rem; }}
    .sidebar-section-title {{ font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #000000; margin: 0.4rem 0 0.6rem 0; }}
    .tech-card {{ background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 12px; padding: 0.65rem 0.8rem; margin-bottom: 0.55rem; transition: transform 0.15s ease, box-shadow 0.15s ease; }}
    .tech-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(108, 43, 217, 0.12); border-color: {PRIMARY}; }}
    .tech-card .t-name {{ font-weight: 700; font-size: 0.85rem; color: {TEXT_PRIMARY}; }}
    .tech-card .t-detail {{ font-size: 0.73rem; color: {TEXT_SECONDARY}; margin-top: 0.1rem; }}
    .agent-card {{ background: #FAF7FF; border: 1.5px solid #D8BEF5; border-radius: 12px; padding: 0.65rem 0.8rem; margin-bottom: 0.55rem; display: flex; gap: 0.6rem; align-items: flex-start; }}
    .agent-card .a-num {{ font-weight: 800; font-size: 0.78rem; color: {PRIMARY}; background: #EFE4FC; border-radius: 6px; padding: 0.1rem 0.4rem; min-width: 26px; text-align: center; }}
    .agent-card .a-name {{ font-weight: 700; font-size: 0.85rem; color: {TEXT_PRIMARY}; }}
    .agent-card .a-detail {{ font-size: 0.73rem; color: {TEXT_SECONDARY}; margin-top: 0.1rem; }}
    .hero {{ background: linear-gradient(135deg, #F3EEFC 0%, #EDE4FA 100%); border: 1.5px solid #D8BEF5; border-radius: 32px; padding: 2.1rem 2.3rem; margin-bottom: 1.2rem; width: 100%; box-sizing: border-box; overflow: hidden; }}
    .hero h1 {{ font-size: 2rem; font-weight: 800; color: {TEXT_PRIMARY}; margin-bottom: 0.5rem; }}
    .hero .tagline {{ font-size: 1rem; color: {TEXT_SECONDARY}; margin: 0; }}
    .tech-box {{ display: inline-flex; align-items: center; gap: 0.4rem; background: linear-gradient(120deg, {PRIMARY}, {PRIMARY_DARK}); border: 1.5px solid {PRIMARY_DARK}; border-radius: 999px; padding: 0.35rem 0.85rem; margin: 0.2rem 0.3rem 0.2rem 0; font-size: 0.82rem; font-weight: 600; color: #FFFFFF; }}
    .section-label {{ font-size: 0.95rem; font-weight: 700; color: {TEXT_PRIMARY}; margin: 1.1rem 0 0.5rem 0; }}
    .card {{ background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 16px; padding: 1.3rem 1.5rem; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04); margin-bottom: 1.1rem; }}
    .metric-card {{ background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 16px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04); }}
    .metric-card .m-icon {{ font-size: 1.5rem; }}
    .metric-card .m-value {{ font-size: 1.7rem; font-weight: 800; color: {PRIMARY}; margin-top: 0.2rem; }}
    .metric-card .m-label {{ font-size: 0.78rem; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }}
    .cmp-table {{ width: 100%; border-collapse: collapse; background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 14px; overflow: hidden; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05); }}
    .cmp-table th {{ background: {PRIMARY}; color: #FFFFFF; text-align: left; padding: 0.7rem 0.9rem; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    .cmp-table td {{ padding: 0.7rem 0.9rem; font-size: 0.9rem; color: {TEXT_PRIMARY}; border-bottom: 1px solid #EEF2F7; }}
    .cmp-table tr:nth-child(even) {{ background: #FAFBFD; }}
    .cmp-table tr:hover {{ background: #F3EEFC; }}
    .cmp-table a.cand-link {{ color: {TEXT_PRIMARY}; font-weight: 700; text-decoration: none; }}
    .cmp-table a.cand-link:hover {{ color: {PRIMARY}; text-decoration: underline; }}
    .pill {{ display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px; font-weight: 700; font-size: 0.78rem; }}
    .cand-card {{ background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 18px; padding: 1.6rem 1.7rem; margin-bottom: 1.3rem; box-shadow: 0 3px 14px rgba(15, 23, 42, 0.05); }}
    .cand-header {{ display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #EEF2F7; padding-bottom: 0.9rem; margin-bottom: 0.9rem; }}
    .cand-name {{ font-size: 1.35rem; font-weight: 800; color: {TEXT_PRIMARY}; }}
    .cand-file {{ font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 0.15rem; }}
    .ring {{ width: 68px; height: 68px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }}
    .ring-inner {{ width: 54px; height: 54px; border-radius: 50%; background: #FFFFFF; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem; }}
    .field-label {{ font-weight: 700; font-size: 0.88rem; color: {TEXT_PRIMARY}; margin: 0.85rem 0 0.35rem 0; }}
    .field-text {{ font-size: 0.88rem; color: {TEXT_SECONDARY}; line-height: 1.5; }}
    .field-block {{ padding-bottom: 0.9rem; margin-bottom: 0.9rem; border-bottom: 1px solid #E2E8F0; }}
    .field-block:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
    .field-block.interview {{ margin-top: 0.4rem; }}
    .tag {{ display: inline-block; padding: 0.2rem 0.6rem; margin: 0.15rem 0.3rem 0.15rem 0; border-radius: 8px; font-size: 0.78rem; font-weight: 600; }}
    .tag.match {{ background: #D1FAE5; color: #047857; }}
    .tag.missing {{ background: #FEE2E2; color: #B91C1C; }}
    .tag.cert {{ background: #E0E7FF; color: #4338CA; }}
    .tag.strength {{ background: #D1FAE5; color: #047857; }}
    .tag.risk {{ background: #FEF3C7; color: #92400E; }}
    div.stButton > button {{ border-radius: 12px; font-weight: 700; padding: 0.65rem 1.2rem; border: none; }}
    div.stButton > button[kind="primary"] {{ background: linear-gradient(120deg, {PRIMARY}, {PRIMARY_DARK}); color: white; }}
    hr {{ margin: 1.4rem 0; border-color: #E2E8F0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state

if "report" not in st.session_state:
    st.session_state.report = None
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

# Sidebar

with st.sidebar:
    st.markdown(
        '<div class="sidebar-header"><div class="title">🤖 AI HR Assistant</div>'
        '<div class="subtitle">AI-Powered Recruitment Intelligence</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">Technology Stack</div>', unsafe_allow_html=True)
    for tech in TECH_STACK:
        st.markdown(
            f'<div class="tech-card"><div class="t-name">{tech["icon"]} {tech["name"]}</div>'
            f'<div class="t-detail">{tech["detail"]}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-title">Agent Pipeline</div>', unsafe_allow_html=True)
    for agent in AGENT_PIPELINE:
        st.markdown(
            f'<div class="agent-card"><div class="a-num">{agent["num"]}</div><div>'
            f'<div class="a-name">{agent["icon"]} {agent["name"]}</div>'
            f'<div class="a-detail">{agent["detail"]}</div></div></div>',
            unsafe_allow_html=True,
        )

# Hero

st.markdown(
    '<div class="hero"><h1>🤖 AI HR Recruitment Assistant</h1>'
    '<p class="tagline">A multi-agent AI pipeline that screens, ranks, and evaluates candidate resumes end to end.</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    "".join(f'<span class="tech-box">{t["icon"]} {t["name"]}</span>' for t in TECH_STACK),
    unsafe_allow_html=True,
)
st.write("")

# Input section

st.markdown('<div class="section-label">📝 Job Description</div>', unsafe_allow_html=True)
job_description = st.text_area(
    "Job description", height=180, label_visibility="collapsed",
    placeholder="Paste the job description here — e.g. We're hiring a Senior Backend "
    "Engineer with 5+ years of experience in Python, distributed systems, and AWS...",
)

st.markdown('<div class="section-label">📁 Upload Resumes (up to 5)</div>', unsafe_allow_html=True)
max_slider = st.slider("Number of candidates to process", min_value=1, max_value=MAX_CANDIDATES, value=MAX_CANDIDATES)
uploaded_files = st.file_uploader("Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")
if uploaded_files and len(uploaded_files) > max_slider:
    st.warning(f"Only the first {max_slider} file(s) will be analyzed based on the slider above.")
    uploaded_files = uploaded_files[:max_slider]
if uploaded_files:
    for f in uploaded_files:
        st.caption(f"✅ {f.name} · {f.size / 1024:.1f} KB")

st.write("")
run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not (job_description and uploaded_files))
if not job_description or not uploaded_files:
    st.caption("Add a job description and at least one resume to enable analysis.")
st.markdown("---")

# Backend call

def run_analysis(api_url: str, job_desc: str, files: List[Any]) -> Optional[Dict[str, Any]]:
    files_payload = [("resumes", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    data = {"job_description": job_desc}
    progress = st.progress(0, text="Screening resumes...")
    stages = ["Screening resumes", "Ranking candidates", "Drafting interview questions", "Finalizing recommendations"]

    start = time.time()
    try:
        resp = requests.post(api_url.rstrip("/") + "/analyze", data=data, files=files_payload, timeout=310)
    except requests.exceptions.ConnectionError:
        progress.empty()
        st.error(f"Couldn't connect to the backend at `{api_url}`. Make sure `uvicorn main:app` is running.")
        return None
    except requests.exceptions.Timeout:
        progress.empty()
        st.error("The request timed out. The pipeline may still be running on the server.")
        return None

    for i, stage in enumerate(stages):
        progress.progress(int((i + 1) / len(stages) * 100), text=stage)
        time.sleep(0.1)
    progress.empty()

    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        st.error(f"Analysis failed ({resp.status_code}): {detail}")
        return None

    result = resp.json()
    result["_client_elapsed_seconds"] = time.time() - start
    return result

if run_clicked:
    with st.spinner("Running the 4-agent pipeline — this can take a couple of minutes..."):
        result = run_analysis(st.session_state.api_url, job_description, uploaded_files)
    if result:
        st.session_state.report = result
        st.toast("Analysis complete!", icon="✅")

# Derived metric helpers

def skill_match_pct(c: Dict[str, Any]) -> float:
    matched, missing = len(c.get("matched_skills") or []), len(c.get("missing_skills") or [])
    total = matched + missing
    return round((matched / total) * 100) if total else 0.0

def skill_match_color(pct: float) -> str:
    return SUCCESS if pct > 70 else WARNING if pct >= 40 else DANGER

def rank_label(rank: Optional[int]) -> str:
    return "—" if rank is None else RANK_EMOJI.get(rank, f"#{rank}")

def processing_time_str(report: Dict[str, Any]) -> str:
    durations = [s.get("duration_seconds") for s in report.get("agent_trace", []) if s.get("duration_seconds")]
    total = sum(durations) if durations else report.get("_client_elapsed_seconds", 0)
    if not total:
        return "—"
    return f"{total:.0f}s" if total < 60 else f"{total / 60:.1f}m"

# Result renderers

def render_dashboard(report: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
    scores = [c["score"] for c in candidates if c.get("score") is not None]
    avg_score = f"{(sum(scores) / len(scores)):.0f}" if scores else "—"
    strong_hires = sum(1 for c in candidates if c.get("verdict") == "Strong Hire")

    metrics = [
        ("👥", len(candidates), "Total Candidates"),
        ("🏆", strong_hires, "Strong Hires"),
        ("📊", avg_score, "Average Score"),
        ("⚡", processing_time_str(report), "Processing Time"),
    ]
    cols = st.columns(4)
    for col, (icon, value, label) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card"><div class="m-icon">{icon}</div>'
            f'<div class="m-value">{value}</div><div class="m-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

def render_comparison_table(candidates: List[Dict[str, Any]]) -> None:
    rows = []
    for idx, c in enumerate(candidates):
        rank = c.get("rank")
        score = c.get("score")
        verdict = c.get("verdict") or "Pending"
        vstyle = VERDICT_STYLE.get(verdict, {"color": TEXT_PRIMARY, "bg": "#E2E8F0", "emoji": "•"})
        pct = skill_match_pct(c)
        pct_color = skill_match_color(pct)
        score_txt = f"{score:.0f}/100" if score is not None else "—"

        rows.append(
            f'<tr><td>{rank_label(rank)}</td>'
            f'<td><a class="cand-link" href="#cand-{idx}">{c["candidate_name"]}</a></td>'
            f'<td>{score_txt}</td>'
            f'<td><span class="pill" style="background:{vstyle["bg"]}; color:{vstyle["color"]};">'
            f'{vstyle["emoji"]} {verdict}</span></td>'
            f'<td><span style="color:{pct_color}; font-weight:700;">{pct}%</span></td></tr>'
        )

    table_html = (
        '<table class="cmp-table"><thead><tr><th>Rank</th><th>Candidate</th><th>Score</th>'
        f'<th>Verdict</th><th>Skill Match %</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

def _tags(items: List[str], css_class: str, empty_text: str = "None recorded") -> str:
    if not items:
        return f'<span class="field-text">{empty_text}</span>'
    return "".join(f'<span class="tag {css_class}">{i}</span>' for i in items)

def _field_block(icon: str, label: str, content: str, extra_class: str = "") -> str:
    cls = f"field-block {extra_class}".strip()
    return f'<div class="{cls}"><div class="field-label">{icon} {label}</div><div class="field-text">{content}</div></div>'

def _candidate_fields_html(c: Dict[str, Any]) -> str:
    fields = [
        ("📝", "Recommendation Summary", c.get("recommendation_summary") or "No recommendation summary available."),
        ("✅", "Strengths", _tags(c.get("key_strengths", []), "strength")),
        ("⚠️", "Risks", _tags(c.get("key_risks", []), "risk")),
        ("🛠️", "Matched Skills", _tags(c.get("matched_skills", []), "match")),
        ("❌", "Missing Skills", _tags(c.get("missing_skills", []), "missing")),
        ("🎓", "Certifications", _tags(c.get("certifications", []), "cert")),
        ("🏗️", "Notable Projects", _tags(c.get("notable_projects", []), "cert")),
        ("💼", "Experience", c.get("experience_summary") or "Not available."),
        ("🎓", "Education", c.get("education_summary") or "Not available."),
        ("💭", "Overall Impression", c.get("overall_impression") or "Not available."),
        ("⚖️", "Ranking Justification", c.get("ranking_justification") or "Not available."),
    ]
    html = "".join(_field_block(icon, label, content) for icon, label, content in fields)
    questions = "".join(f"{i}. {q}<br/><br/>" for i, q in enumerate(c.get("interview_questions", []), 1)) \
        or "Not generated for this candidate."
    html += _field_block("🎤", "Interview Questions", questions, "interview")
    return html

def render_candidate_profiles(candidates: List[Dict[str, Any]]) -> None:
    for idx, c in enumerate(candidates):
        rank, score = c.get("rank"), c.get("score")
        verdict = c.get("verdict") or "Pending"
        vstyle = VERDICT_STYLE.get(verdict, {"color": TEXT_PRIMARY, "bg": "#E2E8F0", "emoji": "•"})
        ring_color = skill_match_color(score) if score is not None else TEXT_MUTED
        score_display = f"{score:.0f}" if score is not None else "—"
        deg = (score or 0) * 3.6

        st.markdown(
            html_block(
                f"""
                <div class="cand-card" id="cand-{idx}">
                    <div class="cand-header">
                        <div>
                            <div class="cand-name">{rank_label(rank)} {c['candidate_name']}</div>
                            <div class="cand-file">{c['file_name']}</div>
                            <span class="pill" style="background:{vstyle['bg']}; color:{vstyle['color']}; margin-top:0.5rem; display:inline-block;">
                                {vstyle['emoji']} {verdict}</span>
                        </div>
                        <div class="ring" style="background: conic-gradient({ring_color} {deg}deg, #E5E7EB 0deg);">
                            <div class="ring-inner" style="color:{ring_color};">{score_display}</div>
                        </div>
                    </div>
                    {_candidate_fields_html(c)}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

# PDF export

def generate_pdf(report: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bytes:
    def pdf_rank(rank: Optional[int]) -> str:
        return f"#{rank}" if rank is not None else "—"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=rl_colors.HexColor(PRIMARY))
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=rl_colors.HexColor(TEXT_PRIMARY), spaceBefore=10)
    h3_style = ParagraphStyle("H3X", parent=styles["Heading3"], textColor=rl_colors.HexColor(PRIMARY), spaceBefore=8)
    body_style = ParagraphStyle("BodyX", parent=styles["BodyText"], textColor=rl_colors.HexColor(TEXT_SECONDARY), leading=14)

    story = [
        Paragraph("AI HR Recruitment Assistant — Report", title_style),
        Paragraph("The right candidate isn't just found — they're understood.", body_style),
        Spacer(1, 0.5 * cm),
    ]
    if report.get("job_description_excerpt"):
        story += [Paragraph("Job Description (excerpt)", h2_style), Paragraph(report["job_description_excerpt"], body_style), Spacer(1, 0.4 * cm)]

    story.append(Paragraph("Candidate Comparison", h2_style))
    table_data = [["Rank", "Candidate", "Score", "Verdict", "Skill Match %"]]
    for c in candidates:
        score_txt = f"{c['score']:.0f}/100" if c.get("score") is not None else "—"
        table_data.append([pdf_rank(c.get("rank")), c.get("candidate_name", ""), score_txt, c.get("verdict") or "Pending", f"{skill_match_pct(c)}%"])
    tbl = Table(table_data, hAlign="LEFT", colWidths=[2 * cm, 5.5 * cm, 2.5 * cm, 3 * cm, 3 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(PRIMARY)), ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#FAFBFD")]), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    sections_map = [
        ("Recommendation Summary", "recommendation_summary", False), ("Strengths", "key_strengths", True),
        ("Risks", "key_risks", True), ("Matched Skills", "matched_skills", True), ("Missing Skills", "missing_skills", True),
        ("Certifications", "certifications", True), ("Notable Projects", "notable_projects", True),
        ("Experience", "experience_summary", False), ("Education", "education_summary", False),
        ("Overall Impression", "overall_impression", False), ("Ranking Justification", "ranking_justification", False),
    ]
    for c in candidates:
        block = [Paragraph(f"{pdf_rank(c.get('rank'))} {c.get('candidate_name', '')}", h2_style)]
        score_line = (
            f"File: {c.get('file_name', '')} | Score: {c['score']:.0f}/100"
            if c.get("score") is not None else f"File: {c.get('file_name', '')}"
        )
        block.append(Paragraph(score_line, body_style))
        block.append(Paragraph(f"Verdict: {c.get('verdict') or 'Pending'}", body_style))

        for label, key, is_list in sections_map:
            value = ", ".join(c.get(key, [])) if is_list else c.get(key)
            if value:
                block.append(Paragraph(label, h3_style))
                block.append(Paragraph(value, body_style))

        if c.get("interview_questions"):
            block.append(Paragraph("Interview Questions", h3_style))
            for i, q in enumerate(c["interview_questions"], 1):
                block.append(Paragraph(f"{i}. {q}", body_style))
        block.append(Spacer(1, 0.5 * cm))
        story.append(KeepTogether(block))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# Results

report = st.session_state.report

if report is None:
    st.markdown(
        '<div class="card" style="text-align:center; padding:2.6rem 1.5rem;">'
        '<div style="font-size:2.2rem;">📥</div>'
        '<div style="font-weight:700; font-size:1.1rem; margin-top:0.4rem;">No analysis yet</div>'
        '<div style="color:#94A3B8; margin-top:0.3rem;">Fill in the job description and upload resumes '
        'above, then click <strong>Run Analysis</strong>.</div></div>',
        unsafe_allow_html=True,
    )
else:
    candidates = sorted(report.get("candidates", []), key=lambda c: (c.get("rank") is None, c.get("rank") if c.get("rank") is not None else 0))

    st.markdown('<div class="section-label">📊 Dashboard</div>', unsafe_allow_html=True)
    render_dashboard(report, candidates)
    if report.get("skipped_files"):
        st.warning("Skipped files: " + ", ".join(report["skipped_files"]))
    for err in report.get("errors", []):
        st.error(err)

    st.markdown('<div class="section-label">📋 Candidate Comparison</div>', unsafe_allow_html=True)
    if candidates:
        render_comparison_table(candidates)
    else:
        st.info("No candidates to compare.")

    st.markdown('<div class="section-label">👤 Candidate Profiles</div>', unsafe_allow_html=True)
    if candidates:
        render_candidate_profiles(candidates)
    else:
        st.info("No candidate profiles available.")

    st.markdown("---")
    pdf_bytes = generate_pdf(report, candidates)
    st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name="hr_recruitment_report.pdf", mime="application/pdf", use_container_width=True)
