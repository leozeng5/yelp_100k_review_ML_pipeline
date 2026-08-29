import csv, random, re
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
SAMPLE_SIZE = 100_000
SEED = 42
random.seed(SEED)

# 1) Load user & business lookups
user_dict = {}
with open("yelp_users.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        user_dict[r["user_id"]] = {
            "user_review_count": int(r["user_review_count"]),
            "user_avg_stars":    float(r["user_avg_stars"]),
            "useful":            int(r["useful"]),
            "funny":             int(r["funny"]),
            "cool":              int(r["cool"]),
            "fans":              int(r["fans"])
        }

biz_dict = {}
with open("yelp_business.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        biz_dict[r["business_id"]] = {
            "biz_review_count": int(r["biz_review_count"]),
            "biz_avg_stars":    float(r["biz_avg_stars"])
        }

# 2) Reservoir sample from reviews
reservoir = []
with open("yelp_reviews.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(row)
        else:
            j = random.randint(1, i)
            if j <= SAMPLE_SIZE:
                reservoir[j-1] = row

# 3) Feature engineering
upper_re    = re.compile(r"\b[A-Z]{2,}\b")
question_re = re.compile(r"\?")
exclaim_re  = re.compile(r"!")

data = defaultdict(list)
for row in reservoir:
    uid, bid = row["user_id"], row["business_id"]
    txt, stars = row["text"], float(row["review_stars"])
    u, b = user_dict.get(uid), biz_dict.get(bid)
    if u is None or b is None:
        continue

    # text features
    rl   = len(txt)
    wc   = len(txt.split())
    eq   = len(exclaim_re.findall(txt))
    qq   = len(question_re.findall(txt))
    up   = len(upper_re.findall(txt))
    sent = 1 if stars >= 4 else 0

    data["review_length"].append(rl)
    data["word_count"].append(wc)
    data["exclamation_count"].append(eq)
    data["question_count"].append(qq)
    data["upper_word_count"].append(up)
    data["sentiment"].append(sent)
    data["review_stars"].append(stars)

    # user/business features
    for k,v in u.items(): data[k].append(v)
    for k,v in b.items(): data[k].append(v)

# 4) Build DataFrame
df = pd.DataFrame(data)

# 5) Plot 1: Before/After log1p on review_length
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].hist(df["review_length"], bins=50)
axes[0].set_title("Before: review_length")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")
axes[1].hist(np.log1p(df["review_length"]), bins=50)
axes[1].set_title("After log1p: review_length")
axes[1].set_xlabel("log1p(Characters)")
axes[1].set_ylabel("Count")
plt.suptitle("Review Length: Before vs After Transformation")
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

# 6) Plot 2: useful vs funny correlation
plt.figure(figsize=(6,4))
plt.scatter(df["useful"], df["funny"], alpha=0.3)
r = df["useful"].corr(df["funny"])
plt.xlabel("useful")
plt.ylabel("funny")
plt.title(f"useful vs funny (r = {r:.2f})")
plt.tight_layout()
plt.show()

# 7) Plot 3: bin review_stars into neg/pos
df["star_bin"] = pd.cut(df["review_stars"], bins=[0,3,5], labels=["neg","pos"])
counts = df["star_bin"].value_counts().loc[["neg","pos"]]
plt.figure(figsize=(4,3))
counts.plot.bar(color=["#d62728","#2ca02c"])
plt.xlabel("Sentiment Bin")
plt.ylabel("Count")
plt.title("review_stars binned into neg vs pos")
plt.tight_layout()
plt.show()
