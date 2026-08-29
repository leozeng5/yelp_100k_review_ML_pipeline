import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the converted reviews CSV
df = pd.read_csv('/mnt/data/yelp_reviews.csv', encoding='utf-8')

# 2. Load user and business JSON files
users = pd.read_json('/mnt/data/yelp_academic_dataset_user.json', lines=True)
biz   = pd.read_json('/mnt/data/yelp_academic_dataset_business.json', lines=True)

# 3. Merge user-level metrics
df = df.merge(
    users[['user_id','review_count','average_stars','useful','funny','cool','fans']],
    on='user_id', how='left'
).rename(columns={
    'review_count':'user_review_count',
    'average_stars':'user_avg_stars'
})

# 4. Merge business-level metrics
df = df.merge(
    biz[['business_id','review_count','stars','latitude','longitude']],
    on='business_id', how='left'
).rename(columns={
    'review_count':'biz_review_count',
    'stars':'biz_avg_stars'
})

# 5. Engineer additional numeric features from review text
df['review_length']     = df['review_text'].str.len()
df['word_count']        = df['review_text'].str.split().str.len()
df['exclamation_count'] = df['review_text'].str.count('!')
df['question_count']    = df['review_text'].str.count(r'\?')
df['upper_word_count']  = df['review_text'].str.findall(r'\b[A-Z]{2,}\b').str.len()

# 6. Save the enriched DataFrame to CSV
output_path = '/mnt/data/yelp_full_features.csv'
df.to_csv(output_path, index=False)
print(f"Saved full feature set to: {output_path}")

# 7. Plot a correlation matrix of these numeric features
num_cols = [
    'rating', 'sentiment',
    'user_review_count','user_avg_stars','useful','funny','cool','fans',
    'biz_review_count','biz_avg_stars','latitude','longitude',
    'review_length','word_count','exclamation_count','question_count','upper_word_count'
]

corr = df[num_cols].corr().values
labels = num_cols

plt.figure(figsize=(10,10))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar(label='Pearson r')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.title('Correlation Matrix of Engineered Numeric Features')
plt.tight_layout()
plt.show()