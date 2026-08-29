import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost_model as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. 数据加载与清洗
# ----------------------------
df = pd.read_csv("yelp_reviews.csv", encoding="utf-8")
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
df.drop_duplicates(subset=['review_text'], inplace=True)
df.dropna(subset=['review_text', 'sentiment'], inplace=True)

# ----------------------------
# 2. 文本预处理：TF-IDF 特征提取
# ----------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
X = df['review_text']
y = df['sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
# 转换为 float32 以降低内存占用
X_train_tfidf = X_train_tfidf.astype(np.float32)
X_val_tfidf = X_val_tfidf.astype(np.float32)

# 构造 XGBoost 所需的 DMatrix 格式
dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
dval = xgb.DMatrix(X_val_tfidf, label=y_val)

# ----------------------------
# 3. 设置 XGBoost 参数并训练（使用 GPU 加速）
# ----------------------------
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'tree_method': 'hist',    # 使用新版 histogram 算法
    'device': 'cuda',         # 使用 GPU 训练
    'eval_metric': 'error',
    'max_depth': 8,           # 调整为8（比原先9略低）
    'learning_rate': 0.05,    # 降低学习率
    'scale_pos_weight': 0.6,  # 根据负正样本比例，初步设置为0.5
    'max_bin': 128,
    'seed': 42
}

num_round = 100  # 增加迭代轮数
evals = [(dval, 'Test')]

print("开始训练 XGBoost 模型（GPU 加速）...")
bst = xgb.train(params, dtrain, num_round, evals=evals, verbose_eval=10, early_stopping_rounds=10)

# ----------------------------
# 4. 预测及阈值优化
# ----------------------------
# 在验证集上预测得到正类概率
y_val_prob = bst.predict(dval)

# 扫描一系列阈值（从 0.4 到 0.6），计算每个阈值下的 F1 分数，选择最佳阈值
thresholds = np.linspace(0.4, 0.6, 21)
best_threshold = 0.5
best_f1 = 0

from sklearn.metrics import f1_score
for thresh in thresholds:
    y_pred_temp = (y_val_prob > thresh).astype(int)
    current_f1 = f1_score(y_val, y_pred_temp)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh

print(f"最佳阈值: {best_threshold:.3f} 对应的 F1 分数: {best_f1:.4f}")

# 根据最佳阈值得到最终预测结果
y_val_pred = (y_val_prob > best_threshold).astype(int)

print("XGBoost Model Result：")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("F1 Score:", f1_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))


# ----------------------------
# 5. 绘制 ROC 曲线
# ----------------------------
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ----------------------------
# 6. 绘制混淆矩阵
# ----------------------------
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
