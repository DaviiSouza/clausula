"""Clausula web app. Talks to the FastAPI backend.

Run (with the API already running): streamlit run app/streamlit_app.py
"""

import html
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("CLAUSULA_API_URL", "http://localhost:8000")
SAMPLE = (Path(__file__).parent / "sample_tos.txt").read_text()

COLORS = {
    "Limitation of liability": "#e45756",
    "Unilateral termination": "#f58518",
    "Unilateral change": "#eeca3b",
    "Content removal": "#b279a2",
    "Contract by using": "#4c78a8",
    "Choice of law": "#72b7b2",
    "Jurisdiction": "#54a24b",
    "Arbitration": "#9d755d",
}
RISK = {"low": ("🟢", "Low"), "medium": ("🟡", "Medium"), "high": ("🔴", "High")}

st.set_page_config(page_title="Clausula", page_icon="⚖️", layout="wide")
st.title("⚖️ Clausula")
st.caption("Find potentially unfair clauses in Terms of Service before you click *I agree*. "
           "Not legal advice.")

if "text" not in st.session_state:
    st.session_state.text = ""

input_col, _ = st.columns([3, 1])
with input_col:
    tab_text, tab_pdf = st.tabs(["Paste text", "Upload PDF"])
    with tab_text:
        if st.button("Load example"):
            st.session_state.text = SAMPLE
        text = st.text_area("Terms of Service", key="text", height=250,
                            placeholder="Paste the full Terms of Service here…")
        run_text = st.button("Analyze text", type="primary", disabled=not text.strip())
    with tab_pdf:
        pdf = st.file_uploader("Terms of Service (PDF)", type="pdf")
        run_pdf = st.button("Analyze PDF", type="primary", disabled=pdf is None)

result = None
try:
    with st.spinner("Reading the fine print…"):
        if run_text:
            resp = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=120)
        elif run_pdf:
            resp = requests.post(f"{API_URL}/analyze/pdf", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
                                 timeout=120)
        else:
            resp = None
    if resp is not None:
        if resp.ok:
            result = resp.json()
        else:
            st.error(resp.json().get("detail", "Something went wrong"))
except requests.ConnectionError:
    st.error(f"Could not reach the API at {API_URL}. Is it running?")

if result:
    st.divider()
    emoji, label = RISK[result["risk_level"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk level", f"{emoji} {label}")
    c2.metric("Suspicious clauses", result["n_flagged"])
    c3.metric("Sentences analyzed", result["n_sentences"])

    if result["categories"]:
        st.subheader("What we found")
        chips = " ".join(
            f'<span style="background:{COLORS[cat]};color:white;padding:4px 10px;border-radius:12px;'
            f'margin:2px;display:inline-block">{cat} · {n}</span>'
            for cat, n in result["categories"].items()
        )
        st.markdown(chips, unsafe_allow_html=True)

        st.subheader("Clauses to read carefully")
        for clause in result["clauses"]:
            main = clause["labels"][0]["category"]
            st.markdown(
                f'<div style="border-left:5px solid {COLORS[main]};padding:8px 14px;margin:10px 0;'
                f'background:rgba(127,127,127,0.08);border-radius:4px">{html.escape(clause["text"])}</div>',
                unsafe_allow_html=True,
            )
            for lab in clause["labels"]:
                st.markdown(f"**{lab['category']}** ({lab['score']:.0%}): {lab['explanation']}")
    else:
        st.success("No potentially unfair clauses found.")

    st.caption(f"Model: {result['model']}")
