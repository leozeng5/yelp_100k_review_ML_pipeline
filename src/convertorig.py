import json
import pandas as pd

# 定义输入和输出文件名
input_file = "yelp_academic_dataset_review.json"  # Yelp 原始 JSON 数据文件
output_file = "yelp_reviews.csv"  # 生成的 CSV 文件

reviews = []

# 遍历 JSON 文件的每一行，每行对应一条评论
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)
        # 提取评论文本和评分，注意：这里使用 "text" 和 "stars" 字段
        reviews.append({"review_text": review["text"], "rating": review["stars"]})

# 将所有评论转换为 DataFrame，然后保存为 CSV 文件
df = pd.DataFrame(reviews)
df.to_csv(output_file, index=False)
print(f"转换完成，CSV 文件已保存为 {output_file}")
