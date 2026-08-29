import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the enriched feature CSV (from previous step)
df = pd.read_csv("yelp_full_features.csv", 
                 skiprows=lambda i: i>0 and random.random() > 0.01)

# 1. Summary statistics
summary_stats = df[
    ["review_length","word_count","exclamation_count","question_count","upper_word_count",
     "user_review_count","useful","funny","cool","fans",
     "biz_review_count","biz_avg_stars","sentiment"]
].describe().T
print("Summary Statistics:\n", summary_stats)

# 2. Plots

# 2.1 Dataset Overview
plt.figure(figsize=(6,4))
sent_counts = df["sentiment"].value_counts(normalize=True)
sns.barplot(x=sent_counts.index, y=sent_counts.values)
plt.xticks([0,1], ["Negative","Positive"])
plt.ylabel("Proportion")
plt.title("Proportion of Positive vs Negative Reviews")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["user_review_count"], bins=50)
plt.yscale("log")
plt.xlabel("User Review Count")
plt.ylabel("Number of Users (log scale)")
plt.title("Distribution of User Review Counts")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["biz_review_count"], bins=50)
plt.yscale("log")
plt.xlabel("Business Review Count")
plt.ylabel("Number of Businesses (log scale)")
plt.title("Distribution of Business Review Counts")
plt.show()

# 2.2 Textual Feature Distributions
fig, axes = plt.subplots(2,2, figsize=(12,8))
sns.histplot(df["review_length"], bins=50, ax=axes[0,0])
axes[0,0].set_title("Review Length Distribution")

sns.histplot(df["word_count"], bins=50, ax=axes[0,1])
axes[0,1].set_title("Word Count Distribution")

sns.histplot(df["exclamation_count"], bins=20, ax=axes[1,0])
axes[1,0].set_title("Exclamation Count Distribution")

sns.histplot(df["question_count"], bins=20, ax=axes[1,1])
axes[1,1].set_title("Question Count Distribution")

plt.tight_layout()
plt.show()

# 2.3 User Engagement vs. Text Features
plt.figure(figsize=(6,4))
plt.scatter(df["user_review_count"], df["review_length"], alpha=0.2)
plt.xscale("log")
plt.xlabel("User Review Count (log)")
plt.ylabel("Review Length")
plt.title("User Review Count vs. Review Length")
plt.show()

# Rating variation per user
user_stats = df.groupby("user_id")["review_stars"].agg(["mean","std","count"]).reset_index()
plt.figure(figsize=(6,4))
plt.scatter(user_stats["count"], user_stats["std"], alpha=0.2)
plt.xscale("log")
plt.xlabel("User Review Count (log)")
plt.ylabel("Rating Standard Deviation")
plt.title("User Review Count vs. Rating Variation")
plt.show()

# 2.4 Business Metrics
plt.figure(figsize=(6,4))
sns.histplot(df["biz_avg_stars"], bins=20)
plt.xlabel("Business Average Stars")
plt.ylabel("Number of Businesses")
plt.title("Distribution of Business Average Stars")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["biz_review_count"], df["biz_avg_stars"], alpha=0.2)
plt.xscale("log")
plt.xlabel("Business Review Count (log)")
plt.ylabel("Business Average Stars")
plt.title("Biz Review Count vs. Average Stars")
plt.show()

# 2.5 Pairwise Correlation Matrix
corr = df[
    ["review_length","word_count","exclamation_count","question_count","upper_word_count",
     "user_review_count","useful","funny","cool","fans",
     "biz_review_count","biz_avg_stars","sentiment"]
].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation Matrix of Text & Engagement Features")
plt.show()

