"""
Irrelevant comment detection engine.

Each comment is scored against the post it belongs to and given one of four labels:

  relevant     - on-topic, adds to the discussion
  off_topic    - not about the post (low semantic/keyword similarity)
  spam         - promotional, links, contact info, scams, money-making
  low_effort   - generic filler such as "nice", "first!!", emoji-only

Signals
-------
1. Relevance  : TF-IDF cosine similarity between the comment and the post,
                blended with keyword overlap (how many of the comment's
                content words appear in the post).
2. Spam score : weighted rule-based signals, combined (max) with the
                probability from a trained scikit-learn classifier when the
                model file is available.
3. Low effort : short comments made only of generic praise / filler words.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "spam_model.joblib"

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves just also really get got like will one would much even still well im dont ive
youre thats its u ur ya yeah yes ok okay oh
""".split())

GENERIC_WORDS = set("""
nice good great awesome cool wow amazing love loved lovely first second third lol lmao haha hahaha
thanks thank thx ty video post content keep it up bro sis dude man fire lit superb best wonderful
beautiful fantastic excellent perfect op epic sweet hi hello hey hii yo agree same true facts
news info update updates interesting informative helpful useful
""".split())

PROMO_PATTERNS = [
    r"\bsubscribe\b", r"\bsub\s*4\s*sub\b", r"\bcheck\s+(out\s+)?my\b", r"\bvisit\s+my\b",
    r"\bfollow\s+(me|my|back)\b", r"\bmy\s+(channel|page|profile|store|shop|website|blog)\b",
    r"\bclick\s+(here|the\s+link|link)\b", r"\blink\s+in\s+(my\s+)?bio\b", r"\bbuy\s+now\b",
    r"\bfree\s+(gift|followers|likes|iphone|money|giveaway|trial)\b", r"\bgiveaway\b",
    r"\bdiscount\s+code\b", r"\bpromo\s+code\b", r"\buse\s+code\b", r"\bdm\s+me\b",
    r"\binbox\s+me\b", r"\bwork\s+from\s+home\b", r"\bpromote\b", r"\bsponsor",
    r"\blimited\s+offer\b", r"\bact\s+now\b", r"\border\s+now\b", r"\bshop\s+now\b",
]
MONEY_PATTERNS = [
    r"\$\s?\d", r"\b\d+\s?(k|usd|dollars|rs|inr)\b", r"\bearn(ed|ing)?\b", r"\bprofit",
    r"\bincome\b", r"\bcrypto\b", r"\bbitcoin\b", r"\bbtc\b", r"\bforex\b", r"\binvest(ing|ment|or)?\b",
    r"\bpassive\s+income\b", r"\bget\s+rich\b", r"\bmake\s+money\b", r"\bcash\b", r"\btrading\b",
    r"\bloan\b", r"\bcasino\b", r"\bbetting\b", r"\blottery\b",
]
URL_RE = re.compile(r"(https?://|www\.)\S+|\b\S+\.(com|net|org|io|ly|xyz|in|co|me|link|shop)\b(/\S*)?", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(\+?\d[\d\s-]{8,}\d)")
CONTACT_RE = re.compile(r"\b(whats\s?app|telegram|wa\.me|t\.me|signal|snapchat|snap|kik)\b", re.I)
REPEAT_RE = re.compile(r"(.)\1{3,}")
WORD_RE = re.compile(r"[a-z0-9']+")


NO_STEM = {"news", "series", "species", "bus", "gas", "this", "always", "perhaps", "less", "focus", "status", "analysis"}


def stem(word: str) -> str:
    """Very light suffix stripper so 'recipes'/'recipe', 'cooking'/'cook' match."""
    if word in NO_STEM or word.endswith(("ss", "us", "is")):
        return word
    for suf in ("ingly", "edly", "ing", "ies", "ied", "ers", "est", "ed", "es", "er", "ly", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            return base
    return word


def tokenize(text: str) -> list[str]:
    words = WORD_RE.findall(text.lower().replace("’", "'"))
    return [stem(w.strip("'")) for w in words if w.strip("'") and w not in STOPWORDS and len(w) > 1]


def raw_words(text: str) -> list[str]:
    return [w.strip("'") for w in WORD_RE.findall(text.lower()) if w.strip("'")]


@dataclass
class CommentResult:
    index: int
    text: str
    label: str
    is_irrelevant: bool
    relevance: float
    spam_score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


class IrrelevantCommentDetector:
    def __init__(self, threshold: float = 0.10, spam_threshold: float = 0.5, use_model: bool = True):
        self.threshold = threshold
        self.spam_threshold = spam_threshold
        self.model = None
        if use_model and MODEL_PATH.exists():
            try:
                import joblib
                self.model = joblib.load(MODEL_PATH)
            except Exception:  # model is optional; rules still work
                self.model = None

    # ---------- relevance ----------
    @staticmethod
    def _tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
        n = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        vecs = []
        for d in docs:
            tf = Counter(d)
            vec = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vecs.append({t: v / norm for t, v in vec.items()})
        return vecs

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    # ---------- spam ----------
    def _spam_signals(self, text: str) -> tuple[float, list[str]]:
        reasons, score = [], 0.0
        low = text.lower()
        if URL_RE.search(text):
            score += 0.45; reasons.append("Contains a link")
        if EMAIL_RE.search(text) or PHONE_RE.search(text) or CONTACT_RE.search(text):
            score += 0.4; reasons.append("Shares contact info / messaging handle")
        promo = sum(1 for p in PROMO_PATTERNS if re.search(p, low))
        if promo:
            score += min(0.35 + 0.15 * (promo - 1), 0.65); reasons.append("Self-promotion language")
        money = sum(1 for p in MONEY_PATTERNS if re.search(p, low))
        if money:
            score += min(0.2 + 0.15 * (money - 1), 0.5); reasons.append("Money / investment bait")
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            score += 0.15; reasons.append("Mostly CAPS")
        if REPEAT_RE.search(text) or text.count("!") >= 4:
            score += 0.1; reasons.append("Excessive repetition / punctuation")
        return min(score, 1.0), reasons

    def _model_spam_prob(self, texts: list[str]) -> list[Optional[float]]:
        if self.model is None or not texts:
            return [None] * len(texts)
        try:
            return [float(p) for p in self.model.predict_proba(texts)[:, 1]]
        except Exception:
            return [None] * len(texts)

    # ---------- low effort ----------
    @staticmethod
    def _is_low_effort(text: str) -> bool:
        words = raw_words(text)
        if not words:
            return True  # emoji / punctuation only
        if len(words) <= 6 and all(w in GENERIC_WORDS or w in STOPWORDS for w in words):
            return True
        return False

    # ---------- main ----------
    def analyze(self, post: str, comments: list[str], title: str = "") -> dict:
        post_text = f"{title} {title} {post}".strip()  # title weighted x2
        post_tokens = tokenize(post_text)
        comment_tokens = [tokenize(c) for c in comments]
        vecs = self._tfidf_vectors([post_tokens] + comment_tokens)
        post_vec, post_set = vecs[0], set(post_tokens)
        model_probs = self._model_spam_prob(comments)

        results: list[CommentResult] = []
        for i, text in enumerate(comments):
            toks = comment_tokens[i]
            cos = self._cosine(post_vec, vecs[i + 1])
            matched = sorted({t for t in toks if t in post_set})
            overlap = len([t for t in toks if t in post_set]) / len(toks) if toks else 0.0
            relevance = round(min(1.0, 0.6 * cos + 0.4 * overlap), 3)

            rule_spam, reasons = self._spam_signals(text)
            mp = model_probs[i]
            spam = rule_spam
            if mp is not None:
                # model alone needs to be fairly sure; rules + model reinforce
                # The model alone must be very sure (>= 0.9); otherwise it only reinforces rule hits.
                model_alone = mp * 0.9 if mp >= 0.9 else mp * 0.45
                spam = max(rule_spam, model_alone, (rule_spam + mp) / 2 if rule_spam > 0 else 0.0)
                if mp >= 0.9:
                    reasons.append(f"ML model flags as spam ({mp:.0%})")
            spam = round(min(spam, 1.0), 3)

            if spam >= self.spam_threshold:
                label = "spam"
                conf = spam
            elif self._is_low_effort(text):
                label = "low_effort"
                reasons = ["Generic / filler comment with no substance"]
                conf = 0.8
            elif relevance >= self.threshold:
                label = "relevant"
                reasons = [f"Shares topic words with the post: {', '.join(matched[:5])}"] if matched else ["Semantically close to the post"]
                conf = min(0.99, 0.55 + (relevance - self.threshold) * 1.5)
            else:
                label = "off_topic"
                reasons = (["No meaningful overlap with the post's topic"] if not matched
                           else [f"Only weak overlap with the post ({', '.join(matched[:3])})"]) + reasons
                conf = min(0.99, 0.55 + (self.threshold - relevance) * 3)

            results.append(CommentResult(
                index=i, text=text, label=label, is_irrelevant=label != "relevant",
                relevance=relevance, spam_score=spam, confidence=round(conf, 3),
                reasons=reasons, matched_keywords=matched[:8],
            ))

        counts = Counter(r.label for r in results)
        total = len(results)
        return {
            "summary": {
                "total": total,
                "relevant": counts.get("relevant", 0),
                "off_topic": counts.get("off_topic", 0),
                "spam": counts.get("spam", 0),
                "low_effort": counts.get("low_effort", 0),
                "irrelevant": total - counts.get("relevant", 0),
                "irrelevant_pct": round(100 * (total - counts.get("relevant", 0)) / total, 1) if total else 0.0,
                "model_loaded": self.model is not None,
                "threshold": self.threshold,
            },
            "results": [asdict(r) for r in results],
        }
