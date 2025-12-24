import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI / Human 文章偵測器",
    layout="centered"
)

st.title("🧠 AI / Human 文章偵測器")
st.caption("Powered by Hugging Face Transformers")

@st.cache_resource
def load_detector():
    return pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        tokenizer="roberta-base-openai-detector",
        return_all_scores=True
    )

detector = load_detector()

text = st.text_area(
    "請輸入文章內容",
    height=220,
    placeholder="Paste text here..."
)

if st.button("📊 分析"):
    if not text.strip():
        st.warning("請輸入文字")
    else:
        with st.spinner("分析中..."):
            result = detector(text)[0]

        # label 轉換
        scores = {r["label"]: r["score"] for r in result}

        ai = scores.get("AI", scores.get("LABEL_1", 0)) * 100
        human = scores.get("HUMAN", scores.get("LABEL_0", 0)) * 100

        col1, col2 = st.columns(2)
        col1.metric("👤 Human", f"{human:.2f}%")
        col2.metric("🤖 AI", f"{ai:.2f}%")

        if ai > human:
            st.success("➡️ 判定：AI 生成文本")
        else:
            st.info("➡️ 判定：人類撰寫文本")
