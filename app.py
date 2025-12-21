from src.ML_Model_Copilot.components.data_ingestion import DataIngestion
raw_data_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/drug_data/drugTrain_raw.tsv"
processed_dir = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/"
Ingestion=DataIngestion(raw_data_path,processed_dir)
Ingestion.run()