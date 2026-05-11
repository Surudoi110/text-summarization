import os
import time
import re

import nltk
import streamlit as st
import requests

# TextRank via sumy — needs NLTK punkt tokenizer data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .summary-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.4rem;
        font-size: 0.97rem;
        line-height: 1.75;
        white-space: pre-wrap;
    }
    .method-badge {
        display: inline-block;
        background: #4a6cf7;
        color: white;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    method = st.radio(
        "Method",
        ["TextRank (local)", "BART", "Pegasus", "Compare All"],
        index=0,
        help=(
            "**TextRank** — extractive, runs locally, no API key needed.\n\n"
            "**BART / Pegasus** — abstractive, require a free HuggingFace token.\n\n"
            "**Compare All** — run all three and compare results."
        ),
    )

    if "TextRank" in method or method == "Compare All":
        num_sentences = st.slider("TextRank — sentences to extract", 1, 10, 3)
    else:
        num_sentences = 3

    st.divider()

    hf_token = st.text_input(
        "HuggingFace API Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Required for BART and Pegasus. Free token at huggingface.co/settings/tokens (read scope is enough).",
    )

    st.divider()

    show_rouge = st.checkbox("Show ROUGE scores", value=False)

    st.divider()
    st.caption(
        "BART: `facebook/bart-large-cnn`  \n"
        "Pegasus: `google/pegasus-cnn_dailymail`  \n"
        "TextRank: `sumy` (local)"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📝 Text Summarizer")
st.caption(
    "Compare extractive (TextRank) and abstractive (BART, Pegasus) summarization. "
    "TextRank runs locally; BART and Pegasus use the Hugging Face Inference API."
)

# ── Input area ────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Input")
    article = st.text_area(
        label="article",
        height=340,
        placeholder="Paste a news article or any text here…",
        label_visibility="collapsed",
    )

    ref_summary = ""
    if show_rouge:
        ref_summary = st.text_area(
            "Reference summary (for ROUGE scoring):",
            height=100,
            placeholder="Paste the ground-truth / reference summary here…",
        )

    word_count = len(article.split()) if article.strip() else 0
    char_count = len(article)
    st.caption(f"**{word_count}** words · **{char_count}** characters")

    summarize_btn = st.button("✨ Summarize", type="primary", use_container_width=True)

# ── Core functions ────────────────────────────────────────────────────────────

def trim_article(text: str, max_words: int = 800) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def textrank_summarize(text: str, n_sentences: int = 3) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


BART_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
PEGASUS_URL = "https://router.huggingface.co/hf-inference/models/google/pegasus-cnn_dailymail"


def hf_summarize(text: str, api_url: str, token: str, max_retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text}
    for attempt in range(max_retries):
        timeout = 60 + 40 * attempt
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list):
                return result[0].get("summary_text", "")
            return result.get("summary_text", "")
        except requests.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(3)
    raise RuntimeError("Max retries exceeded.")


def clean_summary(text: str) -> str:
    return text.replace("<n>", "\n").strip()


def compute_rouge(hypothesis: str, reference: str) -> dict:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 3),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 3),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 3),
    }


def compression_ratio(original: str, summary: str) -> float:
    orig_words = len(original.split())
    summ_words = len(summary.split())
    if orig_words == 0:
        return 0.0
    return round(summ_words / orig_words * 100, 1)


def render_summary(
    label: str,
    summary: str,
    original: str,
    ref_summary: str = "",
    show_rouge: bool = False,
) -> None:
    summ_words = len(summary.split())
    comp = compression_ratio(original, summary)

    st.markdown(f'<span class="method-badge">{label}</span>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Summary words", summ_words)
    m2.metric("Compression", f"{comp}%", help="Summary words ÷ input words × 100")

    if show_rouge and ref_summary.strip():
        rouge = compute_rouge(summary, ref_summary)
        m3.metric("ROUGE-L", rouge["ROUGE-L"])
        with st.expander("Full ROUGE scores"):
            col_r1, col_r2, col_rl = st.columns(3)
            col_r1.metric("ROUGE-1", rouge["ROUGE-1"])
            col_r2.metric("ROUGE-2", rouge["ROUGE-2"])
            col_rl.metric("ROUGE-L", rouge["ROUGE-L"])
    else:
        m3.metric("Input words", len(original.split()))

    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇ Download summary",
        data=summary,
        file_name=f"summary_{label.lower().replace(' ', '_')}.txt",
        mime="text/plain",
        key=f"dl_{label}_{int(time.time())}",
    )


# ── Output area ───────────────────────────────────────────────────────────────
with col_out:
    st.subheader("Summary")

    if summarize_btn:
        if not article.strip():
            st.warning("Please paste some text on the left to summarize.")
            st.stop()

        needs_token = method in ("BART", "Pegasus", "Compare All")
        if needs_token and not hf_token.strip():
            st.error(
                "A HuggingFace API token is required for BART and Pegasus. "
                "Add it in the sidebar (free at huggingface.co)."
            )
            st.stop()

        trimmed = trim_article(article)

        # ── Compare All: tabs ──────────────────────────────────────────────
        if method == "Compare All":
            tab_tr, tab_bart, tab_peg = st.tabs(["TextRank", "BART", "Pegasus"])

            with tab_tr:
                with st.spinner("Running TextRank locally…"):
                    try:
                        s = textrank_summarize(trimmed, num_sentences)
                        render_summary("TextRank", s, article, ref_summary, show_rouge)
                    except Exception as e:
                        st.error(f"TextRank failed: {e}")

            with tab_bart:
                with st.spinner("Calling BART (may take ~30 s on first run)…"):
                    try:
                        s = clean_summary(hf_summarize(trimmed, BART_URL, hf_token))
                        render_summary("BART", s, article, ref_summary, show_rouge)
                    except Exception as e:
                        st.error(f"BART failed: {e}")

            with tab_peg:
                with st.spinner("Calling Pegasus (may take ~30 s on first run)…"):
                    try:
                        s = clean_summary(hf_summarize(trimmed, PEGASUS_URL, hf_token))
                        render_summary("Pegasus", s, article, ref_summary, show_rouge)
                    except Exception as e:
                        st.error(f"Pegasus failed: {e}")

        # ── Single method ──────────────────────────────────────────────────
        else:
            spinner_msgs = {
                "TextRank (local)": "Running TextRank locally…",
                "BART": "Calling BART API (may take ~30 s on first run)…",
                "Pegasus": "Calling Pegasus API (may take ~30 s on first run)…",
            }

            with st.spinner(spinner_msgs[method]):
                try:
                    if method == "TextRank (local)":
                        summary = textrank_summarize(trimmed, num_sentences)
                        label = "TextRank"
                    elif method == "BART":
                        summary = clean_summary(hf_summarize(trimmed, BART_URL, hf_token))
                        label = "BART"
                    else:
                        summary = clean_summary(hf_summarize(trimmed, PEGASUS_URL, hf_token))
                        label = "Pegasus"

                    render_summary(label, summary, article, ref_summary, show_rouge)

                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    st.exception(e)

    else:
        st.info(
            "1. Choose a method in the **sidebar**  \n"
            "2. Paste your text on the **left**  \n"
            "3. Click **✨ Summarize**"
        )
