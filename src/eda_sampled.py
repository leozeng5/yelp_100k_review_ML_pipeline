import csv, random, re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PARAMETERS
SAMPLE_SIZE = 100_000
SEED = 42
random.seed(SEED)

# 1) Load small tables into dicts (use correct column names)
user_dict = {}
with open("yelp_users.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        user_dict[row["user_id"]] = {
            "user_review_count": int(row["user_review_count"]),
            "user_avg_stars":    float(row["user_avg_stars"]),
            "useful":            int(row["useful"]),
            "funny":             int(row["funny"]),
            "cool":              int(row["cool"]),
            "fans":              int(row["fans"])
        }

biz_dict = {}
with open("yelp_business.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        biz_dict[row["business_id"]] = {
            "biz_review_count": int(row["biz_review_count"]),
            "biz_avg_stars":    float(row["biz_avg_stars"])
        }

# 2) Reservoir sampling of reviews
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

    # Text features
    rl   = len(txt)
    wc   = len(txt.split())
    eq   = len(exclaim_re.findall(txt))
    qq   = len(question_re.findall(txt))
    up   = len(upper_re.findall(txt))
    sent = 1 if stars >= 4 else 0

    # Append features
    data["review_length"].append(rl)
    data["word_count"].append(wc)
    data["exclamation_count"].append(eq)
    data["question_count"].append(qq)
    data["upper_word_count"].append(up)
    data["sentiment"].append(sent)
    data["review_stars"].append(stars)
    data["user_id"].append(uid)

    for k,v in u.items(): data[k].append(v)
    for k,v in b.items(): data[k].append(v)

# 4) Build DataFrame
df = pd.DataFrame(data)

# 5) Summary statistics
print(df.describe().T)

# 6) Plots

# Figure 1: Positive vs Negative
plt.figure(figsize=(4,3))
df["sentiment"].value_counts(normalize=True).plot.bar()
plt.xticks([0,1], ["Negative","Positive"], rotation=0)
plt.ylabel("Proportion")
plt.title("Figure 1: Positive vs Negative Reviews")
plt.tight_layout()
plt.show()

# Figure 2: User Review Count Distribution
plt.figure(figsize=(4,3))
plt.hist(df["user_review_count"], bins=50)
plt.yscale("log")
plt.xlabel("User Review Count")
plt.ylabel("Frequency (log)")
plt.title("Figure 2: User Review Count Distribution")
plt.tight_layout()
plt.show()

# Figure 3: Business Review Count Distribution
plt.figure(figsize=(4,3))
plt.hist(df["biz_review_count"], bins=50)
plt.yscale("log")
plt.xlabel("Business Review Count")
plt.ylabel("Frequency (log)")
plt.title("Figure 3: Business Review Count Distribution")
plt.tight_layout()
plt.show()

# Figure 4: Textual Feature Distributions
fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes[0,0].hist(df["review_length"], bins=50)
axes[0,0].set_title("Review Length")
axes[0,1].hist(df["word_count"], bins=50)
axes[0,1].set_ti_
