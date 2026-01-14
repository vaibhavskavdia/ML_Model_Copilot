import os
import pandas as pd 
import joblib 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from ML_Model_Copilot.logger import logger
from sklearn.svm import LinearSVC

class Model_Selection:
    def __init__(self,artifacts_dir:str):
        self.artifacts_dir=artifacts_dir
        self.X_path=os.path.join(artifacts_dir,"X_tfidf_vectorizer_v1.pkl")
        self.y_path=os.path.join(artifacts_dir,"y_labels_v1.pkl")
        self.model_path1=os.path.join(artifacts_dir,"logistic_regression_v1.pkl")
        self.model_path2=os.path.join(artifacts_dir,"linear_SVC_v1.pkl")
        self.model1=LogisticRegression(max_iter=1000,n_jobs=-1,class_weight="balanced")
        self.model2=LinearSVC(max_iter=1000,class_weight="balanced",random_state=42)
    def load_features(self):
        logger.info("loading tf-idf vectors and labels")
        X=joblib.load(self.X_path)
        y=joblib.load(self.y_path)
        return X,y
    
    def train(self,X_train,y_train):
       logger.info("training regression model")
       self.model2.fit(X_train,y_train) 
       
    def evaluation(self,X_val,y_val):
        logger.info("evaluating the metrics results")
        y_pred=self.model2.predict(X_val)
        
        acc=accuracy_score(y_val,y_pred)
        logger.info(f"accuracy of the {self.model2}model is {acc:.4f}")
        
        class_metrics=classification_report(y_val,y_pred)
        logger.info(f"Classification Report of {self.model2} is:\n" + class_metrics)
        
    def save_model(self):
        joblib.dump(self.model2,self.model_path2)
        logger.info(f"{{self.model2}}model saved at {self.model_path2}")
        
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