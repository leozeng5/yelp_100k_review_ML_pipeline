# yelp_sampled_full_corr_reservoir.py

import json, csv, random, re
import matplotlib.pyplot as plt

# 1) Convert the three JSON files into lean CSVs (same as before)
def convert_json_to_csv():
    # Reviews
    with open("yelp_academic_dataset_review.json","r",encoding="utf-8") as fin, \
         open("yelp_reviews.csv","w",newline="",encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["review_id","user_id","business_id","review_stars","text"])
        for line in fin:
            o = json.loads(line)
            txt = o["text"].replace("\n"," ").replace("\r"," ")
            w.writerow([o["review_id"], o["user_id"], o["business_id"], o["stars"], txt])
    print("→ yelp_reviews.csv done")

    # Users
    with open("yelp_academic_dataset_user.json","r",encoding="utf-8") as fin, \
         open("yelp_users.csv","w",newline="",encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["user_id","user_review_count","user_avg_stars","useful","funny","cool","fans"])
        for line in fin:
            o = json.loads(line)
            w.writerow([o["user_id"], o["review_count"], o["average_stars"],
                        o["useful"], o["funny"], o["cool"], o["fans"]])
    print("→ yelp_users.csv done")

    # Businesses
    with open("yelp_academic_dataset_business.json","r",encoding="utf-8") as fin, \
         open("yelp_business.csv","w",newline="",encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["business_id","biz_review_count","biz_avg_stars"])
        for line in fin:
            o = json.loads(line)
            w.writerow([o["business_id"], o["review_count"], o["stars"]])
    print("→ yelp_business.csv done")


# 2) Reservoir‐sample reviews.csv, enrich and build our sample DataFrame
def build_sample_reservoir(k=200_000):
    # Load users & business into dicts for fast lookup
    user_dict = {}
    with open("yelp_users.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_dict[row["user_id"]] = {
                "user_review_count": int(row["user_review_count"]),
                "user_avg_stars": float(row["user_avg_stars"]),
                "useful": int(row["useful"]),
                "funny": int(row["funny"]),
                "cool": int(row["cool"]),
                "fans": int(row["fans"])
            }

    biz_dict = {}
    with open("yelp_business.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            biz_dict[row["business_id"]] = {
                "biz_review_count": int(row["biz_review_count"]),
                "biz_avg_stars": float(row["biz_avg_stars"])
            }

    # Prepare regex once
    upper_re    = re.compile(r"\b[A-Z]{2,}\b")
    question_re = re.compile(r"\?")
    exclaim_re  = re.compile(r"!")

    reservoir = []
    for i, row in enumerate(csv.DictReader(open("yelp_reviews.csv",encoding="utf-8")), start=1):
        # For each new record, decide if it enters the reservoir
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j-1] = row

    # Now reservoir has k sampled rows—process them
    data = {col: [] for col in [
        "user_review_count","user_avg_stars","useful","funny","cool","fans",
        "biz_review_count","biz_avg_stars",
        "review_length","word_count","exclamation_count","question_count","upper_word_count",
        "sentiment"
    ]}

    for row in reservoir:
        uid, bid, txt = row["user_id"], row["business_id"], row["text"]
        u = user_dict.get(uid)
        b = biz_dict.get(bid)
        if not u or not b:
            continue  # skip if lookup fails

        rl = len(txt)
        wc = len(txt.split())
        eq = len(exclaim_re.findall(txt))
        qq = len(question_re.findall(txt))
        up = len(upper_re.findall(txt))
        stars = float(row["review_stars"])
        sent = 1 if stars >= 4.0 else 0

        # append features
        for k in ["user_review_count","user_avg_stars","useful","funny","cool","fans"]:
            data[k].append(u[k])
        data["biz_review_count"].append(b["biz_review_count"])
        data["biz_avg_stars"].append(b["biz_avg_stars"])
        data["review_length"].append(rl)
        data["word_count"].append(wc)
        data["exclamation_count"].append(eq)
        data["question_count"].append(qq)
        data["upper_word_count"].append(up)
        data["sentiment"].append(sent)

    # Build and return DataFrame-like dict
    return data


# 3) Plot correlation heatmap
def plot_corr(data_dict):
    import numpy as np
    cols = list(data_dict.keys())
    mat = np.column_stack([data_dict[c] for c in cols])
    corr = np.corrcoef(mat, rowvar=False)

    plt.figure(figsize=(10,9))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(f"Reservoir Sampled Correlations (n={len(data_dict[cols[0]])})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convert_json_to_csv()
    data = build_sample_reservoir(200_000)
    plot_corr(data)
