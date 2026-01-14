import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import streamlit as st
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

VECTORIZER_PATH = PROJECT_ROOT / "artifacts" / "tf_idf_vectorizer_v1.pkl"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "linear_SVC_v1.pkl"
print("VECTOR PATH:", VECTORIZER_PATH)
from ML_Model_Copilot.pipelines.inference_pipeline import Sentiment_Inference
from ML_Model_Copilot.genai.llm_explainer import GENAIExplainer
from ML_Model_Copilot.genai.llm_explainer import DummyLLMClient
st.set_page_config(
    page_title="Medical Drug Review Sentiment Analyzer",
    layout="centered"
)
st.title("💊 Medical Drug Review Sentiment Analyzer")
st.markdown(
    "Analyze medical drug reviews using **Machine Learning + GenAI explanations**."
)
#@st.cache_resource
def load_pipelines():
    vectorizer_path = str(VECTORIZER_PATH)

    model_path = str(MODEL_PATH)

    inferencer = Sentiment_Inference(vectorizer_path=vectorizer_path,model_path=model_path)

    explainer = GENAIExplainer(DummyLLMClient())
    return inferencer,explainer
inferencer,explainer = load_pipelines()
review_text = st.text_area(
    "📝 Enter a medical drug review:",
    height=180,
    placeholder="Example: The medicine reduced my pain but caused mild nausea."
)

analyze_button = st.button("🔍 Analyze Sentiment")



if analyze_button:
    if not review_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing review..."):
            prediction = inferencer.predict(review_text)

            explanation = explainer.explain(
                text=review_text,
                sentiment=prediction["sentiment"],
                score=prediction["score"]
            )


        st.subheader("📊 Prediction Result")

        if prediction["sentiment"] == "Positive":
            st.success(f"**Sentiment:** {prediction['sentiment']}")
        else:
            st.error(f"**Sentiment:** {prediction['sentiment']}")

        st.markdown(
            f"**Confidence Score:** `{prediction['score']:.3f}`"
        )

        st.subheader("🧠 Model Explanation")
        st.info(explanation)
