"""
Builds a labelled comment dataset (data/comments.csv) for the spam classifier.

label = 1 -> spam / promotional / scam
label = 0 -> genuine comment (on-topic or casual, but not spam)

The dataset is generated from templates so the project is fully reproducible
without downloading anything. To improve real-world accuracy, append real
labelled comments (e.g. the UCI "YouTube Spam Collection") to the CSV and
re-run `python ml/train.py`.
"""
import csv
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "comments.csv"

SPAM_TEMPLATES = [
    "Check out my channel {link} for more {thing}!",
    "Subscribe to my channel please, I post {thing} every day",
    "Earn ${n} per day working from home, DM me for details",
    "I made ${n}k last month trading {coin}, message me on WhatsApp {phone}",
    "FREE {prize} giveaway!! click here {link}",
    "Follow me on instagram @{handle} for daily {thing}",
    "Visit my website {link} best {thing} at lowest price",
    "Use code {code} for {n}% off, link in bio",
    "Contact {email} for genuine {coin} investment with guaranteed profit",
    "sub4sub anyone? I will subscribe back",
    "Want free followers and likes? Go to {link}",
    "My mentor helped me earn passive income, text her on Telegram @{handle}",
    "BUY NOW!!! LIMITED OFFER {link}",
    "Hey guys please check my new {thing} video, would mean a lot",
    "Get rich quick with {coin}, {n}x returns guaranteed",
    "Promote your business with us, inbox me",
    "Win a {prize} by joining {link}",
    "I am a certified forex trader, DM for signals",
    "Order now and get {prize} free {link}",
    "Best online casino bonus {link} sign up today",
]
HAM_TEMPLATES = [
    "This explanation of {topic} finally made it click for me",
    "Could you do a follow-up on {topic}? I'm still confused about one part",
    "I tried this {topic} method yesterday and it worked really well",
    "At {time} you say the opposite of what the docs say about {topic}",
    "Great breakdown, the part about {topic} was especially useful",
    "I disagree slightly, I think {topic} depends on the context",
    "Does this also apply to {topic} or only the basic case?",
    "My {person} and I watched this together and learned a lot about {topic}",
    "The audio is a bit low but the content on {topic} is solid",
    "Anyone else here because of the exam on {topic} tomorrow?",
    "Thanks for sharing, I had never heard of {topic} before",
    "I think there's a small mistake in the {topic} example",
    "What camera do you use? The footage looks great",
    "The weather here has been crazy all week",
    "I had pasta for dinner tonight",
    "Who else is watching this in {year}?",
    "My cat just walked across my keyboard lol",
    "nice video",
    "first!",
    "love this",
    "Where did you buy that jacket, it looks good",
    "Can't wait for the next part about {topic}",
    "I've been following since the beginning, so happy to see the growth",
    "This reminds me of my grandmother's kitchen",
]
FILL = {
    "link": ["bit.ly/x9a2", "www.cheapdeals.xyz", "https://free-gifts.link/claim", "tinyurl.com/promo22", "shopnow.in/sale"],
    "thing": ["gaming videos", "vlogs", "music covers", "makeup tutorials", "crypto tips", "memes"],
    "n": ["500", "300", "50", "20", "10", "1000"],
    "coin": ["bitcoin", "crypto", "forex", "BTC", "ethereum"],
    "phone": ["+1 555 201 9988", "+91 98765 43210", "+44 7700 900123"],
    "prize": ["iPhone 17", "PS5", "gift card", "Amazon voucher", "Netflix account"],
    "handle": ["deals_daily", "rich_mindset", "anna.trades", "promo_king"],
    "code": ["SAVE20", "FREE50", "VIP10"],
    "email": ["profitteam@gmail.com", "invest.now@mail.com"],
    "topic": ["recursion", "photosynthesis", "the sourdough starter", "React hooks", "compound interest",
              "the French revolution", "gradient descent", "knife sharpening", "the chorus chords", "SQL joins"],
    "time": ["2:15", "5:40", "10:03", "the halfway mark"],
    "person": ["brother", "mom", "roommate", "study group"],
    "year": ["2025", "2026"],
}


def fill(t: str) -> str:
    while "{" in t:
        key = t[t.index("{") + 1 : t.index("}")]
        t = t.replace("{" + key + "}", random.choice(FILL[key]), 1)
    return t


def vary(s: str) -> str:
    r = random.random()
    if r < 0.15:
        return s.upper()
    if r < 0.3:
        return s.lower()
    if r < 0.4:
        return s + " " + random.choice(["!!", "🔥", "😂", "🙏", "💯"])
    return s


def main():
    rows = set()
    for _ in range(700):
        rows.add((vary(fill(random.choice(SPAM_TEMPLATES))), 1))
        rows.add((vary(fill(random.choice(HAM_TEMPLATES))), 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for text, label in sorted(rows):
            w.writerow([text, label])
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
