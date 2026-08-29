import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb

# ─── Configuration ─────────────────────────────────────────────────────────
CSV_PATH       = "yelp_reviews.csv"
SAMPLE_SIZE    = 100_000   # only work on 100k reviews
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
TFIDF_MAX_FEAT = 5000

# ─── 1) Load & Sample ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, usecols=["review_stars","text"], encoding="utf-8")
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# ─── 2) Label ───────────────────────────────────────────────────────────────
df["sentiment"] = (df["review_stars"] >= 4).astype(int)

# ── NEW ── 2.1) Class balance plot
plt.figure(figsize=(4,3))
sns.countplot(x="sentiment", data=df, palette=["#a6611a","#018571"])
plt.xticks([0,1], ["Negative","Positive"])
plt.title("Class Balance")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ─── 3) Split ───────────────────────────────────────────────────────────────
X = df["text"]
y = df["sentiment"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ─── 4) Vectorize ───────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1,2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)

# ─── 5) Logistic Regression ────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_tfidf, y_train)
y_pred_lr  = lr.predict(X_val_tfidf)
y_proba_lr = lr.predict_proba(X_val_tfidf)[:,1]

metrics_lr = {
    "Accuracy":  accuracy_score(y_val, y_pred_lr),
    "Precision": precision_score(y_val, y_pred_lr),
    "Recall":    recall_score(y_val, y_pred_lr),
    "F1":        f1_score(y_val, y_pred_lr),
    "AUC":       auc(*roc_curve(y_val, y_proba_lr)[:2])
}

# Confusion Matrix
cm_lr = confusion_matrix(y_val, y_pred_lr)
plt.figure(figsize=(4,3))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
plt.title("LR Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_proba_lr)
plt.figure(figsize=(4,3))
plt.plot(fpr_lr, tpr_lr, label=f"AUC={metrics_lr['AUC']:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("LR ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ── NEW ── Precision–Recall for LR
prec_lr, rec_lr, _ = precision_recall_curve(y_val, y_proba_lr)
plt.figure(figsize=(4,3))
plt.plot(rec_lr, prec_lr, label=f"F1={metrics_lr['F1']:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("LR Precision–Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# Top-20 Coefs
coef = lr.coef_[0]
feat = tfidf.get_feature_names_out()
df_coef = pd.DataFrame({"feature":feat,"coef":coef})
top20 = df_coef.reindex(df_coef.coef.abs().sort_values(ascending=False).index)[:20]
plt.figure(figsize=(4,4))
sns.barplot(x="coef", y="feature", data=top20, palette="viridis")
plt.title("LR Top-20 Coefficients")
plt.tight_layout()
plt.show()

# ─── 6) XGBoost ────────────────────────────────────────────────────────────
dtrain = xgb.DMatrix(X_train_tfidf, label=y_train, feature_names=tfidf.get_feature_names_out())
dval   = xgb.DMatrix(X_val_tfidf,   label=y_val,   feature_names=tfidf.get_feature_names_out())

bst    = xgb.train(
    {"objective":"binary:logistic","eval_metric":"auc","seed":RANDOM_STATE},
    dtrain,
    num_boost_round=100,
    evals=[(dval,"val")],
    early_stopping_rounds=10,
    verbose_eval=False
)

y_proba_xgb = bst.predict(dval)
y_pred_xgb  = (y_proba_xgb > 0.5).astype(int)

metrics_xgb = {
    "Accuracy":  accuracy_score(y_val, y_pred_xgb),
    "Precision": precision_score(y_val, y_pred_xgb),
    "Recall":    recall_score(y_val, y_pred_xgb),
    "F1":        f1_score(y_val, y_pred_xgb),
    "AUC":       auc(*roc_curve(y_val, y_proba_xgb)[:2])
}

# Confusion Matrix
cm_xgb = confusion_matrix(y_val, y_pred_xgb)
plt.figure(figsize=(4,3))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
plt.title("XGB Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_proba_xgb)
plt.figure(figsize=(4,3))
plt.plot(fpr_xgb, tpr_xgb, label=f"AUC={metrics_xgb['AUC']:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("XGB ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ── NEW ── Precision–Recall for XGB
prec_xgb, rec_xgb, _ = precision_recall_curve(y_val, y_proba_xgb)
plt.figure(figsize=(4,3))
plt.plot(rec_xgb, prec_xgb, label=f"F1={metrics_xgb['F1']:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("XGB Precision–Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# ─── Feature Importance with real TF-IDF names ──────────────────────────────
# bst is your trained booster, tfidf is your fitted vectorizer

# 1) Get the raw gain dictionary: {'f123': 42.1, ...}
raw_imp = bst.get_score(importance_type="gain")

# 2) Map 'f123' → actual feature name
feature_names = tfidf.get_feature_names_out()
mapped = []
for fid, gain in raw_imp.items():
    idx = int(fid[1:])             # remove the leading 'f' and cast
    mapped.append((feature_names[idx], gain))

# 3) Build a DataFrame, sort, take top 20
imp_df = (
    pd.DataFrame(mapped, columns=["feature","gain"])
      .sort_values("gain", ascending=False)
      .head(20)
)

# 4) Plot with seaborn (horizontal bar)
plt.figure(figsize=(6,4))
sns.barplot(x="gain", y="feature", data=imp_df, palette="magma")
plt.title("XGB Top-20 Feature Gains")
plt.xlabel("Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ─── 7) Comparison ─────────────────────────────────────────────────────────
comp = pd.DataFrame([metrics_lr, metrics_xgb], index=["LogReg","XGB"]).T
print("\nModel comparison (on 100k sample):\n", comp.round(3))

# ── NEW ── Side-by-side bar chart of all metrics
fig, ax = plt.subplots(figsize=(6,4))
comp.plot(kind="bar", ax=ax)
ax.set_ylabel("Score")
ax.set_ylim(0,1)
ax.set_title("LogReg vs XGB: Metric Comparison")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
