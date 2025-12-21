import pandas as pd 
import os

class DataIngestion:
    
    def __init__(self,raw_data_path:str,processed_dir:str):
        self.raw_data_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/drug_data/drugTrain_raw.tsv"
        self.processed_dir = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/"
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_raw_data(self)-> pd.DataFrame:
        #Load raw TSV drug review dataset
        df = pd.read_csv(self.raw_data_path, sep="\t")
        return df
    
    def validate_schema(self,df:pd.DataFrame):
        # Validate required columns and basic schema
        df["benefitsReview"] = df["benefitsReview"].fillna("").astype(str)
        df["commentssReview"] = df["commentsReview"].fillna("").astype(str)
        df["sideEffectsReview"] = df["sideEffectsReview"].fillna("").astype(str)
        df["text"]=df["benefitsReview"]+" "+df["sideEffectsReview"]+" "+df["commentsReview"]
        df["text"] = df["text"].fillna("").astype(str)
        
        required_columns = {"text", "rating"}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def create_sentiment_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert rating into binary sentiment
        rating >= 7 -> positive (1)
        rating <= 4 -> negative (0)
        ratings 5,6 are dropped
        """
        df = df[["text", "rating"]].copy()

        df = df.dropna(subset=["text", "rating"])

        df["rating"] = df["rating"].astype(int)

        df = df[(df["rating"] <= 4) | (df["rating"] >= 7)]

        df["sentiment"] = df["rating"].apply(
            lambda x: 1 if x >= 7 else 0
        )

        df = df.drop(columns=["rating"])
        df = df.rename(columns={"text": "review"})

        return df
    
    def basic_quality_checks(self, df: pd.DataFrame):
        """
        Perform basic data quality checks
        """
        if df.empty:
            raise ValueError("Dataset is empty after preprocessing")

        if df["review"].str.len().min() < 5:
            print("Warning: Very short reviews detected")

        class_dist = df["sentiment"].value_counts(normalize=True)
        print("Class distribution:\n", class_dist)

    
    def save_processed_data(self, df: pd.DataFrame) -> str:
        """
        Save processed dataset snapshot
        """
        output_path = os.path.join(self.processed_dir, "drug_reviews_processed_v1.csv")
        df.to_csv(output_path, index=False)
        return output_path

    
    def run(self) -> str:
        """
        Execute full ingestion pipeline
        """
        df = self.load_raw_data()
        self.validate_schema(df)
        df = self.create_sentiment_label(df)
        self.basic_quality_checks(df)
        output_path = self.save_processed_data(df)

        print(f"Processed data saved at: {output_path}")
        return output_path    