

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
# 2. User Behavior Analysis: Aggregate User Statistics
# ----------------------------
user_stats = df.groupby('user_id').agg(
review_count=('rating', 'count'),
avg_rating=('rating', 'mean')
).reset_index()
top_users = user_stats.sort_values(by='review_count', ascending=False).head(10)
print("Top 10 users by review count:")
print(top_users)

# ----------------------------
# 3. Text Preprocessing: TF‑IDF Feature Extraction
# ----------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = df['review_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# ----------------------------
# 4. Build and Evaluate Logistic Regression Model
# ----------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print("\nLogistic Regression Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ----------------------------
# 5. Generate Diagrams from the Logistic Regression Model
# ----------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 5.1 ROC Curve & AUC
y_test_prob = lr_model.predict_proba(X_test_tfidf)[:, 1]  # Probability estimates for the positive class
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 5.2 Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 5.3 Feature Coefficient Plot (Top 20 Features)
coefficients = lr_model.coef_[0]  # Since it's a binary classification model
feature_names = tfidf_vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
top_features = coef_df.sort_values(by='abs_coefficient', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='coefficient', y='feature', data=top_features, palette='viridis')
plt.title('Top 20 Feature Coefficients from Logistic Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# 1) Load only review_text & sentiment
df = pd.read_csv("yelp_reviews.csv", usecols=["review_text","sentiment"])

# 2) Split
X_train, X_val, y_train, y_val = train_test_split(
    df["review_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

# 3) Vectorize
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(  X_val)

# 4) Fit LR
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_val_tfidf)
y_proba = lr.predict_proba(X_val_tfidf)[:,1]

# 5) Metrics
metrics_lr = {
    "Accuracy":  accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall":    recall_score(y_val, y_pred),
    "F1":        f1_score(y_val, y_pred),
    "AUC":       auc(*roc_curve(y_val, y_proba)[:2])
}

# 6) Confusion matrix (Fig.4)
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
plt.title("Fig.4: LR Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# 7) ROC curve (Fig.5)
fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC={metrics_lr['AUC']:.2f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.title("Fig.5: LR ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.tight_layout(); plt.show()

# 8) Top-20 coefficients (Fig.6)
coef = lr.coef_[0]
feat = tfidf.get_feature_names_out()
df_coef = pd.DataFrame({"feature": feat, "coef": coef})
top20 = df_coef.reindex(df_coef.coef.abs().sort_values(ascending=False).index)[:20]
plt.figure(figsize=(6,5))
sns.barplot(x="coef", y="feature", data=top20, palette="viridis")
plt.title("Fig.6: LR Top-20 Features")
plt.xlabel("Coefficient")
plt.tight_layout(); plt.show()

# 9) Print metrics table (Table 1)
print("\nTable 1: Logistic Regression Metrics")
print(pd.Series(metrics_lr).rename("LogisticRegression").to_frame())
