import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost_model as xgb

# ----------------------------
# 1. 数据加载与清洗
# ----------------------------
df = pd.read_csv("yelp_reviews.csv", encoding="utf-8")
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
df.drop_duplicates(subset=['review_text'], inplace=True)
df.dropna(subset=['review_text', 'sentiment'], inplace=True)

# ----------------------------
# 2. 划分数据集
# ----------------------------
X = df['review_text']
y = df['sentiment']
# stratified split 保证类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 3. TF-IDF 特征提取（加入 n-gram 特征）
# ----------------------------
# 这里设置 ngram_range=(1,2) 表示同时提取 unigram 和 bigram，
# min_df=2 用于过滤只在少数文档中出现的词，max_features 可以控制特征数量
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_features=1500)
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)
print(f"Feature Dimension: {X_train_vec.shape[1]}")  # 输出特征维度检查

# ----------------------------
# 4. 处理类别不平衡（如果需要）
# ----------------------------
if len(np.unique(y_train)) == 2:
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
else:
    scale_pos_weight = 1.0
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# ----------------------------
# 5. 定义 XGBoost 分类器并训练（使用 GPU 加速）
# ----------------------------
model = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='gpu_hist',  # 使用 GPU 加速（若收到警告可调整为 'hist', device='cuda'）
    max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    max_bin=128,
    subsample=1.0,
    colsample_bytree=1.0,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(
    X_train_vec, y_train,
    eval_set=[(X_test_vec, y_test)],
    verbose=True
)

# ----------------------------
# 6. 在测试集上评估模型
# ----------------------------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
