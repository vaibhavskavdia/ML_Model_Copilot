import joblib
import numpy as np
from ML_Model_Copilot.logger import logger
from ML_Model_Copilot.utils import TextCleaner
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

VECTORIZER_PATH = PROJECT_ROOT / "artifacts" / "tf_idf_vectorizer_v1.pkl"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "linear_SVC_v1.pkl"
class Sentiment_Inference:
    def __init__(self,model_path:str,vectorizer_path:str):      
        logger.info("starting inference pipeline")
        self.vectorizer=joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.preprocessor=TextCleaner()
        logger.info("artifacts loaded successfully")
        assert hasattr(self.vectorizer, "idf_"), "Vectorizer is NOT fitted"
        print("VECTOR TYPE:", type(self.vectorizer))
        print("HAS IDF:", hasattr(self.vectorizer, "idf_"))

    def predict(self,text:str):
        logger.info("inferencing on new text")
        logger.info(f"using model:{self.model}")
        clean_text=self.preprocessor.clean(text)
        print("CLEANED TEXT:", clean_text)
        print("VECTOR TYPE:", type(self.vectorizer))
        print("HAS IDF:", hasattr(self.vectorizer, "idf_"))

        X_tfidf = self.vectorizer.transform([clean_text])
        
        prediction = self.model.predict(X_tfidf)[0]
        
        decision_score = self.model.decision_function(X_tfidf)[0]
        
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        
        logger.info(f"Inference result → Sentiment: {sentiment_label}, Score: {decision_score:.4f}")
        
        return {"sentiment": {sentiment_label},"score": float(decision_score)}
        
        
        
        
        
        
        
        
        