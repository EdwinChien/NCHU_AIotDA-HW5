import streamlit as st
import joblib
import numpy as np

# Load model
tfidf = joblib.load("model/tfidf.pkl")
clf = joblib.load("model/clf.pkl")

st.set_page_config(
    page_title="AI / Human 文章偵測器",
    layout="centered"
)

st.title("🧠 AI / Human 文章偵測器")
st.caption("使用 TF-IDF + Logistic Regression")

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
            X = tfidf.transform([text])
            proba = clf.predict_proba(X)[0]

            human = proba[0] * 100
            ai = proba[1] * 100

        st.subheader("判斷結果")
        col1, col2 = st.columns(2)

        col1.metric("👤 Human", f"{human:.2f}%")
        col2.metric("🤖 AI", f"{ai:.2f}%")

        if ai > human:
            st.success("➡️ 判定：AI 生成文本")
        else:
            st.info("➡️ 判定：人類撰寫文本")

        st.bar_chart(
            {
                "Human (%)": human,
                "AI (%)": ai
            }
        )

        st.caption("⚠️ AI 偵測僅為機率判斷，非 100% 準確")
