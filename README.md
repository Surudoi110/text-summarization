# Automatic Text Summarization

Comparing extractive vs. abstractive summarization on the CNN/DailyMail dataset, with a small Streamlit demo that summarizes a pasted news article using Pegasus.

The project explores three approaches and evaluates them with ROUGE-1 / ROUGE-2 / ROUGE-L against reference highlights:

| Approach     | Type         | Model                       |
|--------------|--------------|-----------------------------|
| TextRank     | Extractive   | `sumy` TextRankSummarizer   |
| BART         | Abstractive  | `facebook/bart-large-cnn`   |
| Pegasus      | Abstractive  | `google/pegasus-cnn_dailymail` |

## Repo layout

```
text-summarization/
├── Code.ipynb        Full pipeline: load CNN/DailyMail sample → preprocess → 3 summarizers → ROUGE eval
├── app.py            Streamlit demo (paste an article → get a Pegasus summary)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
# 1. Clone + install
git clone https://github.com/Surudoi110/text-summarization.git
cd text-summarization

python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

# 2. Add your HuggingFace token (Read scope is enough)
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
# then edit .env and paste your token after HF_TOKEN=
```

Generate a token at https://huggingface.co/settings/tokens.

## Run the Streamlit demo

```bash
# Make sure HF_TOKEN is exported into the shell first
# Windows PowerShell:
$env:HF_TOKEN="hf_your_token_here"

# macOS / Linux:
# export HF_TOKEN=hf_your_token_here

streamlit run app.py
```

Open http://localhost:8501, paste a news article into the textarea, click **Summarize**. The app trims the article to 800 words, calls the Pegasus model via HuggingFace's hosted inference API, and renders the summary.

## Re-run the notebook evaluation

The notebook expects a `data/test.csv` file with the standard CNN/DailyMail columns (`article`, `highlights`). The script samples 100 rows with a fixed seed for reproducibility.

```bash
jupyter notebook Code.ipynb
```

Steps inside the notebook:

1. Load + sample CNN/DailyMail
2. Preprocess each article (whitespace collapse + safe truncation to ~3500 chars at the last sentence boundary)
3. Run TextRank, BART, and Pegasus on each sample
4. Compute ROUGE-1 / ROUGE-2 / ROUGE-L against the reference highlights
5. Aggregate the results

## Approach details

### Extractive (TextRank)
Builds a sentence graph weighted by similarity, then ranks sentences by a PageRank-style score and picks the top-N. Output is verbatim sentences from the source — fast, deterministic, but constrained to whatever's literally in the article.

### Abstractive (BART, Pegasus)
Encoder-decoder transformers pre-trained on summarization data — they generate new text rather than picking from the source. BART (`facebook/bart-large-cnn`) is the more general-purpose option; Pegasus (`google/pegasus-cnn_dailymail`) is specifically pre-trained for news summarization.

Both are called over HuggingFace's hosted inference API rather than running locally — keeps the project light and reproducible without a GPU.

### Why ROUGE
ROUGE measures n-gram overlap between the generated summary and a reference. ROUGE-1 (unigrams) catches content overlap; ROUGE-2 (bigrams) penalizes choppy or wrong-word output; ROUGE-L (longest common subsequence) measures sentence-level structure.

## Notes

- **Free-tier HF inference rate limits apply.** Heavy use of the notebook may hit rate limits; it has basic retry/print-on-error handling.
- The Pegasus output uses `<n>` as a sentence separator instead of newlines — the demo replaces those before displaying.

## License

MIT
