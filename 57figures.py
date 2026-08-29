import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import re

# PARAMETERS
CSV_REVIEWS = "yelp_reviews.csv"
CSV_USERS   = "yelp_users.csv"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# 1) Load and sample reviews
df = pd.read_csv(CSV_REVIEWS, usecols=["review_stars", "user_id", "text"], encoding="utf-8")
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# 2) Compute textual features
df["review_length"]      = df["text"].str.len()
df["word_count"]         = df["text"].str.split().str.len()
df["exclamation_count"]  = df["text"].str.count("!")
df["question_count"]     = df["text"].str.count(r"\?")
df["upper_word_count"]   = df["text"].str.count(r"\b[A-Z]{2,}\b")

# 3) Load user engagement metrics and merge
users = pd.read_csv(CSV_USERS, usecols=["user_id", "review_count", "average_stars", "useful", "funny", "cool", "fans"])
users = users.rename(columns={
    "review_count": "user_review_count",
    "average_stars": "user_avg_stars"
})
df = df.merge(users, on="user_id", how="left").dropna(subset=["user_review_count"])


plt.figure(figsize=(6,4))
plt.scatter(
    np.log1p(df["useful"]),
    np.log1p(df["funny"]),
    alpha=0.3
)
plt.xlabel("log1p(useful votes)")
plt.ylabel("log1p(funny votes)")
r = df[["useful","funny"]].apply(lambda x: np.log1p(x)).corr().iloc[0,1]
plt.title(f"log1p(useful) vs log1p(funny) (r = {r:.2f})")
plt.tight_layout()
plt.show()

# --- Figure 1: Star Rating Histogram ---
plt.figure(figsize=(5,4))
plt.hist(df["review_stars"], bins=5, edgecolor="black", align="left")
plt.xticks([1,2,3,4,5])
plt.xlabel("Star Rating")
plt.ylabel("Count")
plt.title("Figure 1: Distribution of Review Star Ratings")
plt.tight_layout()
plt.show()

# --- Figure 3e: Uppercase Word Count Distribution ---
plt.figure(figsize=(5,4))
plt.hist(df["upper_word_count"], bins=30, edgecolor="black")
plt.xlabel("Uppercase Word Count")
plt.ylabel("Count")
plt.title("Figure 3e: Distribution of Uppercase Word Count")
plt.tight_layout()
plt.show()

# --- Figure 4: Engagement Metrics Distributions ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 4a: User Review Count (log scale)
axes[0,0].hist(df["user_review_count"], bins=50)
axes[0,0].set_yscale("log")
axes[0,0].set_xlabel("User Review Count")
axes[0,0].set_ylabel("Frequency (log)")
axes[0,0].set_title("4a: User Review Count (log)")

# 4b: User Average Stars
axes[0,1].hist(df["user_avg_stars"], bins=10, range=(1,5), edgecolor="black")
axes[0,1].set_xlabel("User Average Stars")
axes[0,1].set_ylabel("Frequency")
axes[0,1].set_title("4b: User Average Stars")

# 4c: Useful Votes
axes[0,2].hist(df["useful"], bins=50)
axes[0,2].set_yscale("log")
axes[0,2].set_xlabel("Useful Votes")
axes[0,2].set_ylabel("Frequency (log)")
axes[0,2].set_title("4c: Useful Votes Distribution")

# 4d: Funny Votes
axes[1,0].hist(df["funny"], bins=50)
axes[1,0].set_yscale("log")
axes[1,0].set_xlabel("Funny Votes")
axes[1,0].set_ylabel("Frequency (log)")
axes[1,0].set_title("4d: Funny Votes Distribution")

# 4e: Cool Votes
axes[1,1].hist(df["cool"], bins=50)
axes[1,1].set_yscale("log")
axes[1,1].set_xlabel("Cool Votes")
axes[1,1].set_ylabel("Frequency (log)")
axes[1,1].set_title("4e: Cool Votes Distribution")

# 4f: Fan Count
axes[1,2].hist(df["fans"], bins=50)
axes[1,2].set_yscale("log")
axes[1,2].set_xlabel("Fan Count")
axes[1,2].set_ylabel("Frequency (log)")
axes[1,2].set_title("4f: Fan Count Distribution")

plt.tight_layout()
plt.show()
