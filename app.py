import os

import streamlit as st
import requests

st.set_page_config(
    page_title="News Article Summarizer",
    layout="centered"
)

st.title("News Article Summarizer")
st.write("summarize your news articles here!")

article = st.text_area(
    "Paste a news article here:",
    height=300,
    placeholder="Enter your news article here!"
)

summarize_button = st.button("Summarize")

def trim_article(text, max_words=800):
    words = text.split()
    return " ".join(words[:max_words])

API_URL = "https://router.huggingface.co/hf-inference/models/google/pegasus-cnn_dailymail"
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

def pegasus_summarize(text, timeout=120):
    payload = {"inputs": text}

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()[0]["summary_text"]

def clean_pegasus_summary(summary: str) -> str:
    return summary.replace("<n>", "\n").strip()

if summarize_button:
    if not article.strip():
        st.warning("Please enter an article to summarize.")
    else:
        with st.spinner("Generating summary using Pegasus..."):
            try:
                trimmed_article = trim_article(article)
                raw_summary = pegasus_summarize(trimmed_article)
                summary = clean_pegasus_summary(raw_summary)
                st.subheader("Generated Summary")
                st.write(summary)
            except Exception as e:
                st.error("Something went wrong while generating the summary.")
                st.exception(e)

# streamlit run app.py