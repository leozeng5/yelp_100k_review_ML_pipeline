import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, f1_score
)
from scipy.stats import gaussian_kde


# ─── 1. LOAD & PREPARE DATA ────────────────────────────────────────

# 1.1 Read your CSV into `df`
df = pd.read_csv("yelp_reviews.csv", encoding="utf-8")

# 1.2 Create the binary sentiment column
df['sentiment'] = (df['rating'] >= 4).astype(int)

# 1.3 Drop duplicates or any rows missing text
df.drop_duplicates(subset=['review_text'], inplace=True)
df.dropna(subset=['review_text'], inplace=True)

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────

# 2.1 A numeric feature: review length
df['review_length'] = df['review_text'].str.len()

# 2.2 (Optional) If you have user_metrics, compute and merge here
# user_stats = df.groupby('user_id').agg(
#     review_count=('rating','count'),
#     avg_rating=('rating','mean')
# ).reset_index()
# df = df.merge(user_stats, on='user_id', how='left')

# ─── 3. TEXT VECTORIZATION & MODEL ─────────────────────────────────

# 3.1 Split into train/test
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['review_text'], df['sentiment'], test_size=0.2, random_state=42
)

# 3.2 TF–IDF on text
tfidf = TfidfVectorizer(stop_words="english", max_features=1500)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

# 3.3 Train a simple LogisticRegression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)

# 3.4 Compute test predictions & probabilities
y_test_pred = lr.predict(X_test_tfidf)
y_test_prob = lr.predict_proba(X_test_tfidf)[:,1]

# Optionally print metrics
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("F1 score:", f1_score(y_test, y_test_pred))

# ─── 4. PLOTTING BLOCKS ────────────────────────────────────────────

# Now **after** df is defined and your model is trained, you can run:

# ---- 4.1 Correlation matrix ----
num_cols = ['review_length', 'sentiment']  # + any user metrics
corr = df[num_cols].corr().values
labels = num_cols

plt.figure()
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar(label='Pearson r')
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# ---- 4.2 Coverage histogram ----
coverage = df[num_cols].notnull().mean()*100
plt.figure()
plt.hist(coverage, bins=10)
plt.xlabel('% Non-null')
plt.ylabel('Feature Count')
plt.title('Feature Coverage Distribution')
plt.tight_layout()
plt.show()

# ---- 4.3 Histograms & Boxplots ----
for feat in num_cols:
    vals = df[feat].dropna()
    # Histogram + density
    plt.figure()
    plt.hist(vals, bins=30, alpha=0.6)
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 200)
    plt.plot(xs, kde(xs)*len(vals)*(xs[1]-xs[0]), lw=1)
    plt.title(f'Distribution of {feat}')
    plt.xlabel(feat); plt.ylabel('Count')
    plt.tight_layout(); plt.show()

    # Boxplot by true sentiment
    neg = df[df['sentiment']==0][feat].dropna()
    pos = df[df['sentiment']==1][feat].dropna()
    plt.figure()
    plt.boxplot([neg, pos], tick_labels=['neg','pos'])
    plt.title(f'{feat} by True Sentiment')
    plt.ylabel(feat)
    plt.tight_layout(); plt.show()

# ---- 4.4 Scatter + marginals ----
# Align indices for test set:
test_idx = X_test_text.index
revs = df.loc[test_idx, 'review_length']
probs = y_test_prob

fig = plt.figure(figsize=(6,6))
gs  = fig.add_gridspec(4,4, hspace=0.4, wspace=0.4)
axm = fig.add_subplot(gs[1:4,0:3])
axx = fig.add_subplot(gs[0,0:3], sharex=axm)
axy = fig.add_subplot(gs[1:4,3], sharey=axm)

axm.scatter(revs, probs, s=5)
axm.set_xlabel('Review Length')
axm.set_ylabel('P(sentiment=1)')
axx.hist(revs, bins=30); axx.axis('off')
axy.hist(probs, bins=30, orientation='horizontal'); axy.axis('off')
plt.show()

# ---- 4.5 Boxplot of predicted prob by true class ----
neg_probs = y_test_prob[y_test == 0]
pos_probs = y_test_prob[y_test == 1]
plt.figure()
plt.boxplot([neg_probs, pos_probs], tick_labels=['true=0','true=1'])
plt.ylabel('P(sentiment=1)')
plt.title('Predicted Probabilities by True Class')
plt.tight_layout()
plt.show()

# ---- 4.6 TF–IDF clustering boxplot ----
# Sample for MiniBatchKMeans to avoid OOM
sample_idx = np.random.choice(X_train_tfidf.shape[0], size=50000, replace=False)
mbkm = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=5000)
mbkm.fit(X_train_tfidf[sample_idx])
cluster_test = mbkm.predict(X_test_tfidf)

data0 = y_test_prob[cluster_test==0]
data1 = y_test_prob[cluster_test==1]
plt.figure()
plt.boxplot([data0, data1], tick_labels=['cluster 0','cluster 1'])
plt.ylabel('P(sentiment=1)')
plt.title('Predicted Prob by TF–IDF Cluster')
plt.tight_layout()
plt.show()

# ---- 4.7 ROC & Confusion ----
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.title(f'ROC Curve (AUC={auc(fpr,tpr):.2f})')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test, y_test_pred)
plt.figure()
plt.imshow(cm, cmap='viridis')
plt.colorbar()
plt.xticks([0,1], ['neg','pos']); plt.yticks([0,1], ['neg','pos'])
plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()
