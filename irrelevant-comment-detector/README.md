# Irrelevant Comment Detection Web Application

![CI](../../actions/workflows/ci.yml/badge.svg)

A full-stack web app that takes a post (a video description, blog post, article or product page) plus its comments, and flags the comments that don't belong.

Each comment gets one of four labels:

| Label | Meaning | Example (on a pasta recipe) |
|---|---|---|
| **Relevant** | On-topic, adds to the discussion | "Can I use milk instead of cream for this pasta?" |
| **Off-topic** | Not about the post | "Who won the cricket match yesterday?" |
| **Spam** | Promotion, links, contact handles, money bait | "Earn $500 daily, DM me on WhatsApp" |
| **Low-effort** | Generic filler with no content | "nice video", "first!!", "🔥🔥🔥" |

Everything except *Relevant* counts as irrelevant in the summary.

## How it works

```
             ┌──────────── post + title ────────────┐
comment ──►  │ 1. Relevance                          │
             │    TF-IDF cosine similarity (60%)     │
             │  + keyword overlap with the post (40%)│
             ├───────────────────────────────────────┤
             │ 2. Spam score                         │
             │    rule signals (links, phone/email,  │
             │    promo phrases, money bait, CAPS)   │
             │  + trained ML classifier (Python only)│
             ├───────────────────────────────────────┤
             │ 3. Low-effort check (generic words)   │
             └──────────────┬────────────────────────┘
                            ▼
     spam ≥ 0.5 → Spam │ filler → Low-effort │ relevance ≥ 0.10 → Relevant │ else Off-topic
```

- **Text processing:** lowercase, stop-word removal, light stemming (`recipes → recipe`, `cooking → cook`). The title is counted twice so it carries more weight.
- **ML model:** `ml/train.py` trains a scikit-learn pipeline (word 1–2-gram TF-IDF + character 3–5-gram TF-IDF → Logistic Regression). The model on its own only marks a comment as spam when it is at least 90% sure; below that it only reinforces rule hits. This keeps it from flagging genuine questions.
- **Threshold:** the on-topic threshold (default `0.10`) is adjustable in the UI and the API.
- **Explanations:** every result includes the reasons and the keywords it shared with the post.

## Project structure

```
irrelevant-comment-detector/
├── app/
│   ├── detector.py        # detection engine (relevance, spam, low-effort)
│   └── main.py            # FastAPI server + static UI
├── static/index.html      # web UI (calls the API; falls back to an in-browser engine)
├── ml/
│   ├── build_dataset.py   # generates data/comments.csv
│   └── train.py           # trains + evaluates the spam classifier
├── data/comments.csv      # labelled training data (694 rows)
├── model/                 # spam_model.joblib + metrics.json
├── tests/test_detector.py # pytest suite (engine + API)
├── .github/workflows/ci.yml  # runs the tests on every push
├── Dockerfile, render.yaml, railway.json
├── requirements.txt
└── LICENSE (MIT)
```

## Run locally

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ml/build_dataset.py && python ml/train.py      # optional: model is already included
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the app and http://localhost:8000/docs for interactive API docs. The badge in the top-right of the UI shows **Engine: server + ML model** when the backend is in use. The page starts empty; paste a post and comments, or pick an example from the list.

Run the tests:

```bash
pytest -q
```

## API

`POST /api/analyze`

```json
{
  "title": "Creamy Garlic Pasta",
  "post": "Boil the spaghetti, saute garlic in butter, add cream and parmesan...",
  "comments": ["Can I use milk instead of cream?", "Check out my channel!", "nice video"],
  "threshold": 0.10
}
```

Response (trimmed):

```json
{
  "summary": {"total": 3, "relevant": 1, "off_topic": 0, "spam": 1, "low_effort": 1,
              "irrelevant": 2, "irrelevant_pct": 66.7, "model_loaded": true},
  "results": [
    {"index": 0, "text": "Can I use milk instead of cream?", "label": "relevant",
     "is_irrelevant": false, "relevance": 0.287, "spam_score": 0.066, "confidence": 0.83,
     "reasons": ["Shares topic words with the post: cream"], "matched_keywords": ["cream"]}
  ]
}
```

Other endpoints:

- `POST /api/analyze-csv` (multipart: `post`, `file`, optional `title`, `threshold`, `column`) reads comments from a CSV column named `comment` (or the first column).
- `GET /api/health` returns status, whether the model loaded, and its evaluation metrics.

## Deploy

**Render (free tier):** push this folder to a GitHub repo, then on Render choose *New → Blueprint* and pick the repo. `render.yaml` builds the Dockerfile and uses `/api/health` as the health check.

**Railway:** *New Project → Deploy from GitHub repo*. `railway.json` tells it to build the Dockerfile. Then *Settings → Networking → Generate Domain*.

**Any Docker host:**

```bash
docker build -t comment-detector .
docker run -p 8000:8000 comment-detector
```

The Docker build regenerates the dataset and retrains the model, so the model always matches the installed scikit-learn version.

**Frontend only:** `static/index.html` works on its own (GitHub Pages, Netlify, or any static host). Without a backend it runs the same rules in the browser, without the ML model.

## Limitations and next steps

- The training data is generated from templates, so the reported 100% test accuracy is optimistic. For real-world use, append real labelled comments (for example the UCI *YouTube Spam Collection* dataset) to `data/comments.csv` and re-run `python ml/train.py`.
- Relevance is lexical. A relevant comment that uses completely different words from the post ("this saved my dinner" on a recipe) can be marked off-topic. Swapping TF-IDF for sentence embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2`) would handle synonyms and paraphrases.
- English only.
