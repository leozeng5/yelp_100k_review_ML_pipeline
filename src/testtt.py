import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load and Clean Data
# ----------------------------
df = pd.read_csv("yelp_reviews.csv", encoding="utf-8")
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
df.drop_duplicates(subset=['review_text'], inplace=True)
df.dropna(subset=['review_text', 'sentiment'], inplace=True)

# ----------------------------
# 2. Aggregate User-Level Statistics
# ----------------------------
# Calculate the total number of reviews and average rating per user
user_stats = df.groupby('user_id').agg(
    review_count=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    std_rating=('rating', 'std')
).reset_index()

# Calculate review length (number of words) for each review
df['review_length'] = df['review_text'].apply(lambda x: len(x.split()))
# Compute average review length per user
user_length = df.groupby('user_id')['review_length'].mean().reset_index(name='avg_review_length')
# Merge with user_stats
user_stats = pd.merge(user_stats, user_length, on='user_id')

# ----------------------------
# 3. Diagram 1: Distribution of User Review Count
# ----------------------------
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['review_count'], bins=50, kde=True)
plt.xlabel("Review Count")
plt.ylabel("Number of Users")
plt.title("Distribution of User Review Count")
plt.grid(True)
plt.show()

# ----------------------------
# 4. Diagram 2: User Review Count vs. Average Rating
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='avg_rating', alpha=0.6)
plt.xlabel("Review Count")
plt.ylabel("Average Rating")
plt.title("User Review Count vs. Average Rating")
plt.grid(True)
plt.show()

# ----------------------------
# 5. Diagram 3: User Review Count vs. Average Review Length
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='avg_review_length', alpha=0.6)
plt.xlabel("Review Count")
plt.ylabel("Average Review Length (words)")
plt.title("User Review Count vs. Average Review Length")
plt.grid(True)
plt.show()

# ----------------------------
# 6. Diagram 4: User Review Count vs. Rating Variation
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='std_rating', alpha=0.6)
plt.xlabel("Review Count")
plt.ylabel("Rating Standard Deviation")
plt.title("User Review Count vs. Rating Variation")
plt.grid(True)
plt.show()
