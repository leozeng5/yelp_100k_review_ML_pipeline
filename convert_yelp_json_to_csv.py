# convert_yelp_json_to_csv.py

import json
import csv

def convert_review_json():
    with open("yelp_academic_dataset_review.json", "r", encoding="utf-8") as fin, \
         open("yelp_reviews.csv", "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        # select only the columns we’ll use
        writer.writerow(["review_id", "user_id", "business_id", "stars", "text"])
        for line in fin:
            obj = json.loads(line)
            # remove newlines from text so CSV stays on one line per record
            text = obj["text"].replace("\n", " ").replace("\r", " ")
            writer.writerow([obj["review_id"], obj["user_id"], obj["business_id"], obj["stars"], text])
    print("Wrote yelp_reviews.csv")

def convert_user_json():
    with open("yelp_academic_dataset_user.json", "r", encoding="utf-8") as fin, \
         open("yelp_users.csv", "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "review_count", "average_stars", "useful", "funny", "cool", "fans"])
        for line in fin:
            obj = json.loads(line)
            writer.writerow([
                obj["user_id"],
                obj["review_count"],
                obj["average_stars"],
                obj["useful"],
                obj["funny"],
                obj["cool"],
                obj["fans"],
            ])
    print("Wrote yelp_users.csv")

def convert_business_json():
    with open("yelp_academic_dataset_business.json", "r", encoding="utf-8") as fin, \
         open("yelp_business.csv", "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["business_id", "review_count", "stars", "latitude", "longitude"])
        for line in fin:
            obj = json.loads(line)
            writer.writerow([
                obj["business_id"],
                obj["review_count"],
                obj["stars"],
                obj["latitude"],
                obj["longitude"],
            ])
    print("Wrote yelp_business.csv")

if __name__ == "__main__":
    convert_review_json()
    convert_user_json()
    convert_business_json()
