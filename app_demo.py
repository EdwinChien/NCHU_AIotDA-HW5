import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datasets import load_dataset

st.set_page_config(
    page_title="AI / Human 文章偵測器",
    layout="centered"
)

st.title("🧠 AI / Human 文章偵測器")
st.caption("TF-IDF + Logistic Regression (Streamlit Cloud)")

# =========================
# Cache dataset + model
# =========================
@st.cache_resource
def build_model():
    # 下載 MAGE dataset（Hugging Face）
    dataset = load_dataset("yaful/MAGE", split="train")
    df = dataset.to_pandas()[["text", "label"]].dropna()

    # 減少資料量，避免 Streamlit Cloud 卡
    df = df.sample(5000, random_state=42)  # demo 用
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Pipeline：TF-IDF + LogisticRegression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

model = build_model()

# =========================
# UI
# =========================
text = st.text_area(
    "請輸入文章內容",
    height=220,
    placeholder="Paste your text here..."
)

if st.button("📊 分析"):
    if not text.strip():
        st.warning("請輸入文字")
    else:
        with st.spinner("分析中..."):
            proba = model.predict_proba([text])[0]

        human = proba[0]*100
        ai = proba[1]*100

        col1, col2 = st.columns(2)
        col1.metric("👤 Human", f"{human:.2f}%")
        col2.metric("🤖 AI", f"{ai:.2f}%")

        if ai > human:
            st.success("➡️ 判定：AI 生成文本")
        else:
            st.info("➡️ 判定：人類撰寫文本")

        st.bar_chart({"Human (%)": human, "AI (%)": ai})
