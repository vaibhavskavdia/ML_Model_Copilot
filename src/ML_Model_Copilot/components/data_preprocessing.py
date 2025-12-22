import pandas as pd 
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.ML_Model_Copilot.logger import logger
#import logging
#nltk.download("stopwords")
#nltk.download("wordnet")
#nltk.download("omw-1.4")

class TextProcessing:
    def __init__(self,input_path:str,output_path:str):
        self.input_path=input_path
        self.output_path=output_path
        self.stopwords=set(stopwords.words("english"))
        self.stopwords -={'no','not','nor','never'}
        self.lemmatizer=WordNetLemmatizer()
        print("1")
    def load_data(self)-> pd.DataFrame:
        df=pd.read_csv("/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_processed_v1.csv")
        logger.info(f"Loaded dataset with {len(df)} rows")
        print("2")
        return df

    def clean_text(self,review:str):
        review=review.lower()
        review=re.sub(r"<.*?>", " ", review)#removing html tags 
        review=re.sub(r"[^a-z0-9\s]"," ",review) #removes punctuation
        review=re.sub(r"\s+"," ",review).strip()#normalizes whitespace
        print("3")
        return review
    
    def tokenize_and_lemmatize(self,review:str)->str:
        tokens= review.split() #simple white space tokenization
        cleaned_tokens=[]
        for token in tokens:
            if token not in self.stopwords or len(token)<3:
                lemma=self.lemmatizer.lemmatize(token)
                cleaned_tokens.append(lemma)
        return " ".join(cleaned_tokens)
        print("4")    
    def preprocess(self,df:pd.DataFrame)->pd.DataFrame:
        df=df.copy()
        initial_rows=len(df)
        avg_len_before=df["review"].astype(str).str.len().mean()
        logger.info(f"Starting preprocessing on {initial_rows} rows")
        logger.info(f"Average text length before cleaning: {avg_len_before:.2f}")
        df["clean_text"]=df["review"].astype(str)
        df["clean_text"] = df["clean_text"].apply(self.clean_text)
        df["clean_text"] = df["clean_text"].apply(self.tokenize_and_lemmatize)
        df["clean_text"]=df["clean_text"].str.strip()
        df=df[df["clean_text"]!=""]
        final_rows=len(df)
        dropped_rows=initial_rows-final_rows
        average_text_after=df["clean_text"].str.len().mean()
        logger.info(f"Completed preprocessing")
        logger.info(f"Rows after preprocessing: {final_rows}")
        logger.info(f"Rows dropped: {dropped_rows}")
        logger.info(f"Average text length after cleaning: {average_text_after:.2f}")
        print("5")
        return df[["clean_text", "sentiment"]]  #returns model ready features
    
    def save_data(self,df:pd.DataFrame):
        df.to_csv(self.output_path,index=False)
        logger.info(f"Preprocessed data saved at {self.output_path}")
        print("6")
    def run(self):
        df = self.load_data()
        df_processed = self.preprocess(df)
        self.save_data(df_processed)
        print("7")

        