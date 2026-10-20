"""Clausula web app. Talks to the FastAPI backend.

Run (with the API already running): streamlit run app/streamlit_app.py
"""

import html
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader

from clausula.labels import EXPLANATIONS, HEADLINES, SEVERITY, grade
from clausula.text import split_sentences

API_URL = os.getenv("CLAUSULA_API_URL", "http://localhost:8000")
SAMPLE = (Path(__file__).parent / "sample_tos.txt").read_text()
CHUNK = 40  # sentences per API call, so we can show progress on long documents

GRADES = {
    "A": ("#16a34a", "Looks fair", "We didn't find potentially unfair clauses."),
    "B": ("#65a30d", "Mostly fair", "A few minor points worth knowing."),
    "C": ("#d97706", "Read with care", "Some clauses could work against you."),
    "D": ("#ea580c", "Several red flags", "Many clauses favor the provider over you."),
    "E": ("#dc2626", "Serious red flags", "This contract strongly favors the provider."),
}
SEVERITY_STYLE = {
    "high": ("#dc2626", "#fee2e2", "High impact"),
    "medium": ("#ea580c", "#ffedd5", "Medium impact"),
    "low": ("#ca8a04", "#fef9c3", "Low impact"),
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

st.set_page_config(page_title="Clausula: read the fine print", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1100px; padding-top: 2.5rem;}
  .hero {margin-bottom: 1.5rem;}
  .hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;}
  .hero p {font-size: 1.15rem; color: #475569; margin-top: .3rem;}
  .card {background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 22px 26px;
         box-shadow: 0 1px 3px rgba(15,23,42,.06);}
  .grade {display: flex; gap: 26px; align-items: center;}
  .grade-letter {min-width: 104px; height: 104px; border-radius: 22px; color: #fff; font-size: 64px;
                 font-weight: 800; display: flex; align-items: center; justify-content: center;}
  .grade h2 {margin: 0 0 4px 0; font-size: 1.7rem;}
  .grade p {margin: 0; color: #475569;}
  .stats {display: flex; gap: 28px; margin-top: 12px; color: #334155; font-size: .95rem;}
  .stats b {font-size: 1.25rem; color: #0f172a;}
  .point {background: #fff; border: 1px solid #e2e8f0; border-left-width: 6px; border-radius: 12px;
          padding: 14px 18px; margin: 14px 0 4px 0;}
  .point h4 {margin: 0 0 4px 0; font-size: 1.08rem;}
  .point p {margin: 0; color: #475569; font-size: .95rem;}
  .pill {display: inline-block; font-size: .75rem; font-weight: 600; padding: 2px 10px; border-radius: 999px;
         margin-right: 6px;}
  .quote {border-left: 3px solid #cbd5e1; padding: 6px 12px; margin: 8px 0; color: #334155; font-style: italic;}
  .doc {line-height: 1.75; color: #334155; max-height: 600px; overflow-y: auto;}
  .doc mark {padding: 2px 3px; border-radius: 4px; color: #0f172a;}
  .feature h4 {margin-bottom: 4px;}
  .feature p {color: #475569; font-size: .95rem;}
</style>
""", unsafe_allow_html=True)


def analyze(text: str, progress) -> dict | None:
    """Send the document to the API in chunks of sentences and merge the results."""
    sentences = split_sentences(text)
    if not sentences:
        return None
    clauses = []
    for i in range(0, len(sentences), CHUNK):
        resp = requests.post(f"{API_URL}/analyze", json={"text": "\n".join(sentences[i:i + CHUNK])}, timeout=300)
        resp.raise_for_status()
        clauses += resp.json()["clauses"]
        done = min(i + CHUNK, len(sentences))
        progress.progress(done / len(sentences), text=f"Reading the fine print… {done} / {len(sentences)} sentences")
    counts = Counter(lab["category"] for c in clauses for lab in c["labels"])
    return {"sentences": sentences, "clauses": clauses, "categories": dict(counts), "grade": grade(counts)}


def pdf_text(file) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(file.getvalue())).pages)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚖️ About Clausula")
    st.markdown(
        "Clausula reads Terms of Service and flags clauses that are often considered **unfair to consumers**, "
        "in 8 categories studied by legal researchers (CLAUDETTE project)."
    )
    st.markdown("#### How it works")
    st.markdown(
        "1. The text is split into sentences.\n"
        "2. A fast **TF-IDF model** screens every sentence.\n"
        "3. Suspicious ones are read by **legal-bert**, a transformer fine-tuned on 9k labeled ToS sentences.\n"
        "4. Findings are weighted by impact into a **grade from A to E**."
    )
    st.markdown("#### Model quality")
    st.markdown("macro-F1 **0.78** on the LexGLUE `unfair_tos` test set.")
    st.markdown(
        "[💻 Source code](https://github.com/DaviiSouza/clausula) · "
        "[🤗 Model](https://huggingface.co/DaviiiiS/clausula-legal-bert)"
    )
    st.caption("English only. Not legal advice: a starting point for reading, not a legal assessment.")

# ---------- Hero ----------
st.markdown(
    '<div class="hero"><h1>⚖️ Clausula</h1>'
    "<p>Know what you're agreeing to. Paste any Terms of Service and see the clauses that could work against you.</p>"
    "</div>",
    unsafe_allow_html=True,
)

if "text" not in st.session_state:
    st.session_state.text = ""

tab_text, tab_pdf = st.tabs(["📋 Paste text", "📄 Upload PDF"])
with tab_text:
    text = st.text_area("Terms of Service", key="text", height=220, label_visibility="collapsed",
                        placeholder="Paste the full Terms of Service here (English)…")
    c1, c2, _ = st.columns([1.2, 1.2, 4])
    run_text = c1.button("Analyze", type="primary", disabled=not text.strip(), use_container_width=True)
    # A callback runs before the text area is created, so it can change its value.
    c2.button("Try an example", use_container_width=True, on_click=lambda: st.session_state.update(text=SAMPLE))
with tab_pdf:
    pdf = st.file_uploader("Terms of Service (PDF)", type="pdf", label_visibility="collapsed")
    run_pdf = st.button("Analyze PDF", type="primary", disabled=pdf is None)

if run_text or run_pdf:
    progress = st.progress(0.0, text="Reading the fine print…")
    try:
        document = text if run_text else pdf_text(pdf)
        st.session_state.result = analyze(document, progress)
        if st.session_state.result is None:
            st.warning("We couldn't find any sentences in this text.")
    except requests.ConnectionError:
        st.error("The analysis service is starting up. Please try again in a few seconds.")
    except requests.RequestException:
        st.error("The analysis took too long or failed. Try a shorter document.")
    except Exception:
        st.error("We couldn't read this file. Is it a text PDF (not a scanned image)?")
    finally:
        progress.empty()

result = st.session_state.get("result")

# ---------- Empty state ----------
if not result:
    st.write("")
    cols = st.columns(3)
    features = [
        ("🔎", "Finds the fine print", "Arbitration, account termination, liability limits and 5 more kinds of clause."),
        ("💬", "Explains it plainly", "Every finding comes with a short explanation of what it means for you."),
        ("🅰️", "Grades the contract", "From A (fair) to E (serious red flags), weighted by how much each clause can hurt."),
    ]
    for col, (icon, title, desc) in zip(cols, features, strict=True):
        col.markdown(f'<div class="card feature"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
    st.stop()

# ---------- Results ----------
color, title, subtitle = GRADES[result["grade"]]
n_kinds = len(result["categories"])
st.write("")
st.markdown(
    f'<div class="card grade"><div class="grade-letter" style="background:{color}">{result["grade"]}</div>'
    f"<div><h2>{title}</h2><p>{subtitle}</p>"
    f'<div class="stats"><span><b>{len(result["clauses"])}</b> clauses flagged</span>'
    f"<span><b>{n_kinds}</b> of 8 kinds found</span>"
    f'<span><b>{len(result["sentences"])}</b> sentences read</span></div></div></div>',
    unsafe_allow_html=True,
)

tab_points, tab_doc = st.tabs(["⚠️ What you should know", "📜 Full document"])

with tab_points:
    if not result["categories"]:
        st.success("No potentially unfair clauses found. Still, nothing replaces reading it yourself!")
    by_category = {}
    for clause in result["clauses"]:
        for lab in clause["labels"]:
            by_category.setdefault(lab["category"], []).append((lab["score"], clause["text"]))
    for category in sorted(by_category, key=lambda c: (SEVERITY_ORDER[SEVERITY[c]], -len(by_category[c]))):
        sev_color, sev_bg, sev_label = SEVERITY_STYLE[SEVERITY[category]]
        items = sorted(by_category[category], reverse=True)
        st.markdown(
            f'<div class="point" style="border-left-color:{sev_color}">'
            f'<span class="pill" style="background:{sev_bg};color:{sev_color}">{sev_label}</span>'
            f'<span class="pill" style="background:#f1f5f9;color:#475569">{category}</span>'
            f"<h4>{HEADLINES[category]}</h4><p>{EXPLANATIONS[category]}</p></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Show {len(items)} clause{'s' if len(items) > 1 else ''}"):
            for score, sentence in items:
                st.markdown(f'<div class="quote">“{html.escape(sentence)}”</div>', unsafe_allow_html=True)
                st.caption(f"Confidence: {score:.0%}")

with tab_doc:
    flagged = {c["text"]: c["labels"][0]["category"] for c in result["clauses"]}
    parts = []
    for sentence in result["sentences"]:
        safe = html.escape(sentence)
        if sentence in flagged:
            sev_color, sev_bg, _ = SEVERITY_STYLE[SEVERITY[flagged[sentence]]]
            parts.append(f'<mark style="background:{sev_bg};box-shadow:inset 0 -2px 0 {sev_color}" '
                         f'title="{flagged[sentence]}">{safe}</mark>')
        else:
            parts.append(safe)
    st.markdown(f'<div class="card doc">{" ".join(parts)}</div>', unsafe_allow_html=True)
    st.caption("Highlighted sentences were flagged. Hover over them to see the category.")
