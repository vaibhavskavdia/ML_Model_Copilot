from src.ML_Model_Copilot.components.data_ingestion import DataIngestion
from src.ML_Model_Copilot.components.data_preprocessing import TextProcessing
raw_data_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/drug_data/drugTrain_raw.tsv"
processed_dir = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/"
#Ingestion=DataIngestion(raw_data_path,processed_dir)
#Ingestion.run()
def main():
    preprocessor=TextProcessing(
    input_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_processed_v1.csv",
    output_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_processed_v2.csv")
    
if __name__=="__main__":
    main()