import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 你本地有 train.csv（不上傳）
df = pd.read_csv("data/train.csv")

X = df["text"]
y = df["label"]  # 0=human, 1=AI

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(max_iter=2000))
])

pipeline.fit(X, y)

# ✔ 只存這個
joblib.dump(pipeline, "model/model.pkl")

print("Saved model/model.pkl")
