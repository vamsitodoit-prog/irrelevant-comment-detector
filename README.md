# Irrelevant Comment Detector

A web app that checks the comments under a post and flags the ones that don't belong. Paste a post and its comments, and every comment is labelled as relevant, off-topic, spam or low-effort, with scores and a reason.

**Live app:** https://irrelevant-comment-detector.onrender.com/

> The app runs on Render's free plan. It sleeps after about 15 minutes without visitors, so the first load after a pause can take around 30 seconds.

## What it does

| Label | Meaning | Example on a pasta recipe |
|---|---|---|
| **Relevant** | On-topic and adds to the discussion | "Can I use milk instead of cream for this pasta?" |
| **Off-topic** | Not about the post | "Who won the cricket match yesterday?" |
| **Spam** | Promotion, links, contact handles, money bait | "Earn $500 daily, DM me on WhatsApp" |
| **Low-effort** | Generic filler with no content | "nice video", "first!!", "🔥🔥🔥" |

Everything except Relevant counts as irrelevant in the summary at the top of the results.

Features:
- Paste comments one per line, or upload a CSV
- Adjustable on-topic threshold
- Filter results by label
- Relevance and spam scores, plus the reason for each label
- Copy the results as CSV
- REST API with interactive docs at `/docs`

## How it works

1. **Relevance:** TF-IDF cosine similarity between the comment and the post (the title counts double), blended with keyword overlap after removing stop-words and stemming.
2. **Spam:** rule signals (links, phone numbers, email addresses, messaging handles, promotion phrases, money bait, all caps, repeated characters) combined with a trained scikit-learn classifier (word and character TF-IDF into logistic regression).
3. **Low-effort:** short comments made only of generic praise or filler words.

The model on its own only marks a comment as spam when it is at least 90% sure. Below that it only reinforces the rule signals, so genuine questions aren't flagged.

## Tech stack

Python, FastAPI, scikit-learn, and a plain HTML, CSS and JavaScript front end. Deployed with Docker on Render.

## Project layout

The project lives in the `irrelevant-comment-detector/` folder of this repo.

```
irrelevant-comment-detector/
├── app/
│   ├── detector.py          # detection engine
│   └── main.py              # FastAPI server and static UI
├── static/index.html        # web interface
├── ml/
│   ├── build_dataset.py     # generates the training data
│   └── train.py             # trains and evaluates the spam model
├── data/comments.csv        # labelled training data
├── model/                   # trained model and metrics
├── tests/test_detector.py   # automated tests
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Run it on your computer (Windows)

You need Python 3.10 or newer from python.org. During install, tick **Add python.exe to PATH**.

```powershell
cd irrelevant-comment-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser.

If `venv\Scripts\activate` is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first and try again.

Run the tests:

```powershell
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

The response contains a `summary` with counts and a `results` list. Each result has `label`, `relevance`, `spam_score`, `confidence`, `reasons` and `matched_keywords`.

Other endpoints:
- `POST /api/analyze-csv` reads comments from a CSV column named `comment` (or the first column)
- `GET /api/health` returns status and model metrics

## Deployment

Render rebuilds and redeploys the app automatically on every push to `main`, using the `Dockerfile` and `render.yaml` in this repo.

## Limitations

- The training data is generated from templates, so the reported 100% test accuracy is optimistic. Adding real labelled comments to `data/comments.csv` and re-running `python ml/train.py` would give a more honest model.
- Relevance is based on shared words, not meaning, so a relevant comment that paraphrases the post can be marked off-topic. Sentence embeddings would handle this better.
- English only.
