import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/train.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Classifier
clf = LogisticRegression(
    max_iter=2000,
    n_jobs=-1
)

clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(clf, "model/clf.pkl")

print("Model saved to /model")
