import joblib
import numpy as np
from logger import logger
from src.ML_Model_Copilot.utils import TextCleaner

class Sentiment_Inference:
    def __init__(self,model_path:str,vectorizer_path:str):
        logger.info("starting inference pipeline")
        self.vectorizer=joblib.load(vectorizer_path)
        self.model=joblib.load(model_path)
        self.preprocessor=TextCleaner()
        logger.info("artifacts loaded successfully")
        
    def predict(self,text:str):
        logger.info("inferencing on new text")
        logger.info(f"using model:{self.model}")
        clean_text=self.preprocessor.clean("text")
        
        X_tfidf = self.vectorizer.transform([clean_text])
        
        prediction = self.model.predict(X_tfidf)[0]
        
        decision_score = self.model.decision_function(X_tfidf)[0]
        
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        
        logger.info(f"Inference result → Sentiment: {sentiment_label}, Score: {decision_score:.4f}")
        
        return {"sentiment": {sentiment_label},"score": float(decision_score)}
        
        
        
        
        
        
        
        
        