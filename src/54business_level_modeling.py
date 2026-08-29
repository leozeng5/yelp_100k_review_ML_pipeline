import csv, re
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error

# Streaming aggregation to avoid loading the entire reviews CSV

# Pre-load user and business info into dicts
user_info = {}
with open("yelp_users.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        user_info[row["user_id"]] = {
            "review_count": int(row["review_count"]),
            "avg_stars": float(row["average_stars"]),
            "useful": int(row["useful"]),
            "funny": int(row["funny"]),
            "cool": int(row["cool"]),
            "fans": int(row["fans"])
        }

business_info = {}
with open("yelp_business.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        business_info[row["business_id"]] = {
            "biz_review_count": int(row["review_count"]),
            "biz_avg_stars": float(row["stars"])
        }

# Initialize aggregator
agg = defaultdict(lambda: defaultdict(float))
counts = defaultdict(int)

upper_re = re.compile(r"\b[A-Z]{2,}\b")
question_re = re.compile(r"\?")
exclaim_re = re.compile(r"!")

# Stream through reviews
with open("yelp_reviews.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bid = row["business_id"]
        uid = row["user_id"]
        txt = row["text"]
        # Compute features
        rl = len(txt)
        wc = len(txt.split())
        eq = len(exclaim_re.findall(txt))
        qq = len(question_re.findall(txt))
        up = len(upper_re.findall(txt))
        # user metrics
        u = user_info.get(uid)
        b = business_info.get(bid)
        if u is None or b is None:
            continue
        # accumulate sums
        agg[bid]["sum_review_length"]    += rl
        agg[bid]["sum_word_count"]       += wc
        agg[bid]["sum_exclamation_count"]+= eq
        agg[bid]["sum_question_count"]   += qq
        agg[bid]["sum_upper_word_count"] += up
        agg[bid]["sum_user_review_count"]+= u["review_count"]
        agg[bid]["sum_user_avg_stars"]   += u["avg_stars"]
        agg[bid]["sum_useful"]           += u["useful"]
        agg[bid]["sum_funny"]            += u["funny"]
        agg[bid]["sum_cool"]             += u["cool"]
        agg[bid]["sum_fans"]             += u["fans"]
        counts[bid] += 1

# Build aggregated DataFrame
rows = []
for bid, sums in agg.items():
    n = counts[bid]
    binfo = business_info.get(bid)
    if binfo is None:
        continue
    row = {
        "business_id": bid,
        "avg_review_length":    sums["sum_review_length"] / n,
        "avg_word_count":       sums["sum_word_count"] / n,
        "avg_exclamation_count":sums["sum_exclamation_count"] / n,
        "avg_question_count":   sums["sum_question_count"] / n,
        "avg_upper_word_count": sums["sum_upper_word_count"] / n,
        "avg_user_review_count":sums["sum_user_review_count"] / n,
        "avg_user_avg_stars":   sums["sum_user_avg_stars"] / n,
        "total_useful":         sums["sum_useful"],
        "total_funny":          sums["sum_funny"],
        "total_cool":           sums["sum_cool"],
        "avg_fans":             sums["sum_fans"] / n,
        "biz_review_count":     binfo["biz_review_count"],
        "biz_avg_stars":        binfo["biz_avg_stars"]
    }
    rows.append(row)

df_agg = pd.DataFrame(rows)

# Split and model
X = df_agg.drop(columns=["business_id","biz_avg_stars"])
y = df_agg["biz_avg_stars"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regression
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)
bst = xgb.train({"objective":"reg:squarederror","eval_metric":"rmse"}, dtrain, num_boost_round=100, evals=[(dtest,"test")], early_stopping_rounds=10, verbose_eval=False)
y_pred_xgb = bst.predict(dtest)
print("XGB R2:", r2_score(y_test, y_pred_xgb), "RMSE:", np.sqrt(mean_squared_error(y_test,y_pred_xgb)))

# ElasticNetCV Regression
en = ElasticNetCV(l1_ratio=[.1,.5,.9], cv=5, random_state=42).fit(X_train, y_train)
y_pred_en = en.predict(X_test)
print("EN R2:", r2_score(y_test,y_pred_en), "RMSE:", np.sqrt(mean_squared_error(y_test,y_pred_en)))

# Plot feature importances as before...

