import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="AI / Human 文章偵測器",
    layout="centered"
)

st.title("🧠 AI / Human 文章偵測器")
st.caption("TF-IDF + Logistic Regression (Streamlit Cloud Safe)")

# =========================
# 建立示範資料
# =========================
@st.cache_resource
def build_model():
    # Sample texts
    texts = [
        "I went to the store today and bought some apples.",
        "The stock market fluctuates daily based on investor sentiment.",
        "Artificial intelligence can generate human-like text easily.",
        "GPT models are trained on massive datasets to predict text.",
        "The cat sat on the mat and purred softly."
    ]
    labels = [0, 0, 1, 1, 0]  # 0=Human, 1=AI

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(texts, labels)
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
