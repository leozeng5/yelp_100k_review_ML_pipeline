import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load Data
# ----------------------------
# Read Yelp reviews data (ensure the file path is correct)
df = pd.read_csv("yelp_reviews.csv", encoding="utf-8")

# Check the first few rows of data
print("Data Preview:")
print(df.head())

# ----------------------------
# 2. Data Preprocessing: Compute Additional Features
# ----------------------------
# Calculate the number of words in review text (as an indicator of review length)
df['review_length'] = df['review_text'].apply(lambda x: len(x.split()))

# ----------------------------
# 3. Aggregate Statistics by User
# ----------------------------
# Group by user_id and calculate:
# - review_count: number of reviews per user
# - avg_rating: average rating per user
# - std_rating: rating standard deviation (reflecting rating variation)
# - avg_review_length: average review length (in number of words)
user_stats = df.groupby('user_id').agg(
    review_count=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    std_rating=('rating', 'std'),
    avg_review_length=('review_length', 'mean')
).reset_index()

# View the top 10 users by review count
top_users = user_stats.sort_values(by='review_count', ascending=False).head(10)
print("Top 10 users by review count:")
print(top_users)

# ----------------------------
# 4. Plot Diagrams to Display User Review Behavior Patterns
# ----------------------------

# Diagram 1: Distribution of Review Count per User
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['review_count'], bins=50, kde=True)
plt.xlabel("Review Count")
plt.ylabel("Number of Users")
plt.title("Distribution of User Review Count")
plt.grid(True)
plt.show()

# Diagram 2: Scatter Plot - User Review Count vs. Average Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='avg_rating', alpha=0.5)
plt.xlabel("Review Count")
plt.ylabel("Average Rating")
plt.title("User Review Count vs. Average Rating")
plt.grid(True)
plt.show()

# Diagram 3: Scatter Plot - User Review Count vs. Average Review Length (words)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='avg_review_length', alpha=0.5)
plt.xlabel("Review Count")
plt.ylabel("Average Review Length (words)")
plt.title("User Review Count vs. Average Review Length")
plt.grid(True)
plt.show()

# Diagram 4 (Optional): Scatter Plot - User Review Count vs. Rating Standard Deviation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_stats, x='review_count', y='std_rating', alpha=0.5)
plt.xlabel("Review Count")
plt.ylabel("Rating Standard Deviation")
plt.title("User Review Count vs. Rating Variation")
plt.grid(True)
plt.show()

# You can further compute and visualize other aspects like the time distribution of reviews if needed.
