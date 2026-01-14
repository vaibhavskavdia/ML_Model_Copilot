# ML_Model_Copilot
# 💊 Medical Drug Review Sentiment Analyzer

An end-to-end **Machine Learning + GenAI** application that analyzes medical drug reviews and predicts **sentiment (Positive / Negative)** along with **human-readable explanations**.

---

## 🚀 Project Highlights

- 🔍 TF-IDF + Linear SVM sentiment classifier
- 🧠 Error analysis & misclassification inspection
- 🤖 GenAI-powered explanation layer
- 🌐 Interactive Streamlit frontend
- 🐳 Fully Dockerized for reproducibility
- 📦 Modular, production-style project structure

---

## 🏗 Architecture Overview
User Review (Streamlit UI)
↓
Text Preprocessing
↓
TF-IDF Vectorizer
↓
Linear SVM Model
↓
Sentiment Prediction + Confidence
↓
GenAI Explanation


---

## 🧪 Models Used

| Component | Technique |
|---------|-----------|
| Feature Extraction | TF-IDF |
| Classifier | Linear SVM (class_weight=balanced) |
| Explainability | GenAI (LLM-based reasoning) |

---

## 📊 Results

- **Accuracy:** ~85%
- **Class imbalance handled**
- Strong performance on negative reviews
- Detailed false-positive / false-negative analysis

---

## 🧠 Example Output

**Input:**
> “The medicine made my condition much worse. I had severe nausea and pain.”

**Prediction:**
- Sentiment: ❌ Negative
- Confidence Score: -0.74

**Explanation:**
> The review emphasizes worsening symptoms and adverse side effects, which strongly influenced the model to classify the sentiment as negative.

---

## 🖥️ Frontend (Streamlit)

Screenshots:

<p align="center">
  <img src="/Users/vaibhavkavdia/Desktop/medical_bot_ss.png" width="600">
</p>

---

## 🐳 Run with Docker

```bash
docker build -t medical-sentiment-app .
docker run -p 8501:8501 medical-sentiment-app

#then open:

http://localhost:8501

locally run:
pip install -r requirements.txt
python app.py```


Project Structure

components/ → data ingestion, preprocessing

pipelines/ → training & inference pipelines

genai/ → explanation & error analysis

frontend/ → Streamlit UI

artifacts/ → trained models

📌 Future Improvements

FastAPI backend

Cloud deployment (AWS / Render)

SHAP-based explanations

Continuous model monitoring

👤 Author

Vaibhav Kavdia
B.Tech, IIT Roorkee
Aspiring ML / AI Engineer



---

