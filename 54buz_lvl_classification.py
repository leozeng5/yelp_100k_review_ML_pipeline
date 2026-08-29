import csv, re
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ─── 1) Load user and business tables ───────────────────────────────────────
user_info = {}
with open("yelp_users.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        user_info[row["user_id"]] = {
            "user_review_count": int(row["review_count"]),
            "user_avg_stars":    float(row["average_stars"]),
            "useful":            int(row["useful"]),
            "funny":             int(row["funny"]),
            "cool":              int(row["cool"]),
            "fans":              int(row["fans"])
        }

business_info = {}
with open("yelp_business.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        business_info[row["business_id"]] = {
            "biz_review_count": int(row["review_count"]),
            "biz_avg_stars":    float(row["stars"])
        }

# ─── 2) Stream reviews and aggregate review-/user-level features ────────────
upper_re    = re.compile(r"\b[A-Z]{2,}\b")
question_re = re.compile(r"\?")
exclaim_re  = re.compile(r"!")

sums   = defaultdict(lambda: defaultdict(float))
counts = defaultdict(int)

with open("yelp_reviews.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bid = row["business_id"]
        uid = row["user_id"]
        text = row["text"]

        u = user_info.get(uid)
        b = business_info.get(bid)
        if u is None or b is None:
            continue

        # compute features
        rl = len(text)
        wc = len(text.split())
        eq = len(exclaim_re.findall(text))
        qq = len(question_re.findall(text))
        up = len(upper_re.findall(text))

        # accumulate
        sums[bid]["sum_review_length"]     += rl
        sums[bid]["sum_word_count"]        += wc
        sums[bid]["sum_exclamation_count"] += eq
        sums[bid]["sum_question_count"]    += qq
        sums[bid]["sum_upper_word_count"]  += up
        sums[bid]["sum_user_review_count"] += u["user_review_count"]
        sums[bid]["sum_user_avg_stars"]    += u["user_avg_stars"]
        sums[bid]["sum_useful"]            += u["useful"]
        sums[bid]["sum_funny"]             += u["funny"]
        sums[bid]["sum_cool"]              += u["cool"]
        sums[bid]["sum_fans"]              += u["fans"]

        counts[bid] += 1

# ─── 3) Build aggregated DataFrame ──────────────────────────────────────────
rows = []
for bid, c in counts.items():
    binfo = business_info.get(bid)
    if binfo is None or c == 0:
        continue
    rows.append({
        "business_id":            bid,
        "avg_review_length":      sums[bid]["sum_review_length"]    / c,
        "avg_word_count":         sums[bid]["sum_word_count"]       / c,
        "avg_exclamation_count":  sums[bid]["sum_exclamation_count"]/ c,
        "avg_question_count":     sums[bid]["sum_question_count"]   / c,
        "avg_upper_word_count":   sums[bid]["sum_upper_word_count"] / c,
        "avg_user_review_count":  sums[bid]["sum_user_review_count"]/ c,
        "avg_user_avg_stars":     sums[bid]["sum_user_avg_stars"]   / c,
        "total_useful":           sums[bid]["sum_useful"],
        "total_funny":            sums[bid]["sum_funny"],
        "total_cool":             sums[bid]["sum_cool"],
        "avg_fans":               sums[bid]["sum_fans"]            / c,
        "biz_review_count":       binfo["biz_review_count"],
        "biz_avg_stars":          binfo["biz_avg_stars"]
    })

agg = pd.DataFrame(rows)
print(f"Aggregated {len(agg)} businesses.")

# ─── 4) Binarize target ──────────────────────────────────────────────────────
agg["biz_label"] = (agg["biz_avg_stars"] >= 4.0).astype(int)

# ─── 5) Train/test split ────────────────────────────────────────────────────
X = agg.drop(columns=["business_id", "biz_avg_stars", "biz_label"])
y = agg["biz_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ─── Scale for Logistic Regression ──────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 6) Logistic Regression ────────────────────────────────────────────────
lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=1.0,
    max_iter=5000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lr.fit(X_train_scaled, y_train)
y_pred_lr  = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:,1]

metrics_lr = {
    "Accuracy":  accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr),
    "Recall":    recall_score(y_test, y_pred_lr),
    "F1":        f1_score(y_test, y_pred_lr),
    "AUC":       auc(*roc_curve(y_test, y_proba_lr)[:2])
}

# plot LR confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low","High"], yticklabels=["Low","High"])
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# plot LR ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr, label=f"AUC={metrics_lr['AUC']:.2f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.title("Logistic Regression ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# plot LR top-20 coefficients
df_coef = pd.DataFrame({
    "feature": X.columns,
    "coef": lr.coef_[0]
})
top20_lr = df_coef.reindex(df_coef.coef.abs().sort_values(ascending=False).index)[:20]
plt.figure(figsize=(5,5))
sns.barplot(x="coef", y="feature", data=top20_lr, color="teal")
plt.title("LR Top-20 Coefficients")
plt.tight_layout()
plt.show()

# ─── 7) XGBoost Classifier ─────────────────────────────────────────────────
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    seed=RANDOM_STATE,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
xgb_clf.fit(X_train, y_train)

y_pred_xgb  = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)[:,1]

metrics_xgb = {
    "Accuracy":  accuracy_score(y_test, y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb),
    "Recall":    recall_score(y_test, y_pred_xgb),
    "F1":        f1_score(y_test, y_pred_xgb),
    "AUC":       auc(*roc_curve(y_test, y_proba_xgb)[:2])
}

# plot XGB confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low","High"], yticklabels=["Low","High"])
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# ─── XGB ROC Curve ─────────────────────────────────────────────────────────
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
plt.figure(figsize=(4,3))
plt.plot(fpr_xgb, tpr_xgb, label=f"AUC={metrics_xgb['AUC']:.2f}")
plt.plot([0,1],[0,1], "--", color="gray")
plt.title("XGBoost ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ─── XGB Top-20 Feature Gains ────────────────────────────────────────────────
raw_imp = xgb_clf.get_booster().get_score(importance_type="gain")

# Build DataFrame directly from keys & values
imp_df = (
    pd.DataFrame({
        "feature": list(raw_imp.keys()),
        "gain":    list(raw_imp.values())
    })
    .sort_values("gain", ascending=False)
    .head(20)
)

plt.figure(figsize=(5,5))
sns.barplot(x="gain", y="feature", data=imp_df, palette="magma")
plt.title("XGBoost Top-20 Feature Gains")
plt.xlabel("Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ─── Summary Table ─────────────────────────────────────────────────────────
comp = pd.DataFrame([metrics_lr, metrics_xgb], index=["LogisticRegression","XGBoost"]).T
print("\nValidation Metrics for Business-Level Classification:\n")
print(comp.round(3))
