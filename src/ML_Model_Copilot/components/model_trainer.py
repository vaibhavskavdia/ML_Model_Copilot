import os
import pandas as pd 
import joblib 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from src.ML_Model_Copilot.logger import logger

class Model_Selection:
    def __init__(self,artifacts_dir:str):
        self.artifacts_dir=artifacts_dir
        self.X_path=os.path.join(artifacts_dir,"X_tfidf_vectorizer_v1.pkl")
        self.y_path=os.path.join(artifacts_dir,"y_labels_v1.pkl")
        self.model_path=os.path.join(artifacts_dir,"logistic_regression_v1.pkl")
        self.model=LogisticRegression(max_iter=1000,n_jobs=-1,class_weight="balanced")
        
    def load_features(self):
        logger.info("loading tf-idf vectors and labels")
        X=joblib.load(self.X_path)
        y=joblib.load(self.y_path)
        return X,y
    
    def train(self,X_train,y_train):
       logger.info("training regression model")
       self.model.fit(X_train,y_train) 
       
    def evaluation(self,X_val,y_val):
        logger.info("evaluating the metrics results")
        y_pred=self.model.predict(X_val)
        
        acc=accuracy_score(y_val,y_pred)
        logger.info(f"accuracy of the model is {acc:.4f}")
        
        class_metrics=classification_report(y_val,y_pred)
        logger.info(f"Classification Report:\n" + class_metrics)
        
    def save_model(self):
        joblib.dump(self.model,self.model_path)
        logger.info(f"model savex at {self.model_path}")
        
    def run(self):
        X,y=self.load_features()
        X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        logger.info(f"loaaded x_train shape:{X_train.shape[0]}")
        logger.info(f"loaded validation of shape:{X_val.shape[0]}")
        
        self.train(X_train,y_train)
        self.evaluation(X_val,y_val)
        self.save_model()
        
"""The baseline Logistic Regression achieved ~83% accuracy with strong performance on the majority class and reasonable recall on the minority class. 
The error pattern indicates class imbalance as the primary limitation rather than feature quality."""