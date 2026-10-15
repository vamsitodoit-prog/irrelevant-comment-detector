from fastapi.testclient import TestClient

from app.detector import IrrelevantCommentDetector
from app.main import app

POST = ("How to make a creamy garlic pasta in 15 minutes. Boil the spaghetti, saute garlic in butter, "
        "add cream and parmesan cheese, then toss the pasta.")
TITLE = "Creamy Garlic Pasta"

CASES = {
    "Can I use milk instead of cream for this pasta?": "relevant",
    "My parmesan burned, what heat should the butter be at?": "relevant",
    "Who won the cricket match yesterday?": "off_topic",
    "I hate Mondays so much": "off_topic",
    "Earn $500 daily from home, DM me on WhatsApp +91 98765 43210": "spam",
    "Check out my channel for gaming videos": "spam",
    "FREE iPhone giveaway click here bit.ly/abc": "spam",
    "nice video": "low_effort",
    "🔥🔥🔥": "low_effort",
}


def test_labels_rules_only():
    d = IrrelevantCommentDetector(use_model=False)
    res = d.analyze(POST, list(CASES), TITLE)["results"]
    for r in res:
        assert r["label"] == CASES[r["text"]], (r["text"], r["label"], r["relevance"], r["spam_score"])


def test_labels_with_model():
    d = IrrelevantCommentDetector()
    res = d.analyze(POST, list(CASES), TITLE)["results"]
    for r in res:
        assert r["label"] == CASES[r["text"]], (r["text"], r["label"])


def test_summary_counts():
    s = IrrelevantCommentDetector().analyze(POST, list(CASES), TITLE)["summary"]
    assert s["total"] == len(CASES)
    assert s["relevant"] + s["irrelevant"] == s["total"]
    assert s["spam"] == 3 and s["low_effort"] == 2


def test_api_roundtrip():
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    r = c.post("/api/analyze", json={"post": POST, "title": TITLE, "comments": list(CASES)})
    assert r.status_code == 200
    assert len(r.json()["results"]) == len(CASES)


def test_api_validation():
    c = TestClient(app)
    assert c.post("/api/analyze", json={"post": POST, "comments": ["  "]}).status_code == 400
    assert c.post("/api/analyze", json={"post": POST, "comments": []}).status_code == 422


def test_api_csv():
    c = TestClient(app)
    csv_body = "comment\nCan I use milk instead of cream?\nnice video\n"
    r = c.post("/api/analyze-csv", data={"post": POST}, files={"file": ("c.csv", csv_body, "text/csv")})
    assert r.status_code == 200 and r.json()["summary"]["total"] == 2


def test_model_does_not_flag_genuine_questions_alone():
    d = IrrelevantCommentDetector()
    post = "The useEffect hook in React: dependency array, cleanup functions and avoiding render loops."
    r = d.analyze(post, ["Why does my effect run twice in development with StrictMode?", "good news"], "React useEffect")
    assert r["results"][0]["label"] != "spam"
    assert r["results"][1]["label"] == "low_effort"
