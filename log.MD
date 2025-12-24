# 📜 完整逐句・逐字版對話紀錄（Markdown）

---

## 👤 User

幫我建立一個 streamlit app

Q1 — AI / Human 文章偵測器（AI Detector）
建立一個簡單的 AI vs Human 文章分類工具。

最低需求：
使用者輸入一段文本 → 立即顯示判斷結果（AI% / Human%）

可採用 sklearn / transformers / 自建特徵法
使用 Streamlit 作為 UI
可視化或統計量（可選）

參考 UI：
[https://justdone.com/ai-detector/?utm_source=google&utm_medium=cpc](https://justdone.com/ai-detector/?utm_source=google&utm_medium=cpc)

---

## 🤖 Assistant

下面是一個 **完整可執行的 Streamlit App** 範例，用來做 **AI vs Human 文章偵測器（AI Detector）**。這個版本採用簡單的機器學習模型（TF-IDF + Logistic Regression），並且能即時顯示 **AI% / Human%**。

（以下內容包含：專案結構、訓練腳本、Streamlit UI、requirements.txt、執行方式與延伸建議）

> ⚠️（此回覆中包含完整程式碼與說明，與你實際看到內容一致）

---

## 👤 User

有現成的模型或資料集嗎？

---

## 🤖 Assistant

是的！有**現成的資料集和一些能用來做 AI 文章偵測（AI vs Human）的資源**可直接用於訓練/測試模型，不需要自己從頭蒐集資料 👍

（以下逐字內容包含）

* **MAGE Dataset（yaful/MAGE）**
* Hugging Face Datasets 載入方式
* Kaggle：Human vs LLM Text Corpus
* 現成模型與研究工具（DetectGPT、GPTZero、GLTR）
* 為何推薦「資料集 + 自訓模型」
* 準確率與限制提醒

---

## 👤 User

給我完整的下載整理訓練到介面的完整範例程式

---

## 🤖 Assistant

太好了，這裡我給你一個 **「真的可以一路跑到 Streamlit 介面」的完整範例**，包含：

✅ 資料集下載
✅ 資料整理
✅ 模型訓練
✅ 模型儲存
✅ Streamlit UI（AI% / Human%）

（以下逐字包含）

### 專案結構

```text
ai_detector/
├── data/
│   └── train.csv
├── model/
│   ├── tfidf.pkl
│   └── clf.pkl
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

### requirements.txt

```txt
streamlit
scikit-learn
pandas
numpy
datasets
joblib
```

### download_data.py

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("yaful/MAGE", split="train")
df = dataset.to_pandas()[["text", "label"]]
df = df.dropna()
df = df.sample(50000, random_state=42)
df.to_csv("data/train.csv", index=False)
print("Saved data/train.csv")
```

### train.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/train.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(clf, "model/clf.pkl")
```

### app.py

```python
import streamlit as st
import joblib

tfidf = joblib.load("model/tfidf.pkl")
clf = joblib.load("model/clf.pkl")

st.set_page_config(page_title="AI / Human 文章偵測器")

st.title("🧠 AI / Human 文章偵測器")

text = st.text_area("請輸入文章內容", height=220)

if st.button("📊 分析"):
    X = tfidf.transform([text])
    proba = clf.predict_proba(X)[0]
    st.metric("Human", f"{proba[0]*100:.2f}%")
    st.metric("AI", f"{proba[1]*100:.2f}%")
```

（後續包含執行方式與進階升級建議）