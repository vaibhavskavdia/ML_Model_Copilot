import pandas as pd 
import joblib as jb
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from logger import logger

class FeatureExtraction:
    def __init__(self,input_path:str,artifacts_dir:str,max_features:int=8000, ngrams_range:tuple=(1,2),min_df:int=5,max_df:float=0.9):
        self.input_path=input_path
        self.artifacts_dir=artifacts_dir
        os.makedirs(self.artifacts_dir,exist_ok=True)
        
        self.vectorizer=TfidfVectorizer(ngram_range=ngrams_range,max_features=max_features,min_df=min_df,max_df=max_df)
        
    def load_data(self):
        logger.info("preprocessed data in loaded")
        df=pd.read_csv(self.input_path)
        X_test=df["clean_text"].astype(str)
        y=df["sentiment"]
        logger.info(f"Dataset shape:{df.shape}")
        return X_test,y
    
    def fit_transform(self,X_test):
        logger.info(f"applying tf-idf vectorizer")
        X_tfidf=self.vectorizer.fit_transform(X_test)
        logger.info(f"tf-idf matrix change:{X_tfidf.shape}")
        return X_tfidf
    
    def save_artifacts(self,X_tfidf,y):
        vectorizer_path=os.path.join(self.artifacts_dir,"tf_idf_vectorizer_v1.pkl")
        feature_path=os.path.join(self.artifacts_dir,"X_tfidf_vectorizer_v1.pkl")
        labels_path=os.path.join(self.artifacts_dir,"y_labels_v1.pkl")
        jb.dump(self.vectorizer,vectorizer_path)
        jb.dump(X_tfidf,feature_path)
        jb.dump(y,labels_path)
        logger.info(f"Saved TF-IDF vectorizer at {vectorizer_path}")
        logger.info(f"Saved feature matrix at {feature_path}")
        logger.info(f"Saved labels at {labels_path}")
    
    def run(self):
        X_test,y=self.load_data()
        X_tfidf=self.fit_transform(X_test)
        self.save_artifacts(X_tfidf,y)
        