from src.ML_Model_Copilot.components.data_ingestion import DataIngestion
from src.ML_Model_Copilot.components.data_preprocessing import TextProcessing
from src.ML_Model_Copilot.components.feature_engineering import FeatureExtraction
from src.ML_Model_Copilot.components.model_trainer import Model_Selection
from src.ML_Model_Copilot.genai.error_analyser import ErrorAnalyzer
from src.ML_Model_Copilot.pipelines.inference_pipeline import Sentiment_Inference
raw_data_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/drug_data/drugTrain_raw.tsv"
processed_dir = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/"
#Ingestion=DataIngestion(raw_data_path,processed_dir)
#Ingestion.run()
'''def main():
    preprocessor=TextProcessing(
    input_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_processed_v1.csv",
    output_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_preprocessed_v1.csv")
    print("main function started")
    preprocessor.run()'''
input_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_preprocessed_v1.csv"
artifacts_dir="artifacts"
data_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/AI/ML-Model_Copilot/src/ML_Model_Copilot/data/processed/drug_reviews_preprocessed_v1.csv"
model1_path="artifacts/logistic_regression_v1.pkl"
model2_path="artifacts/linear_SVC_v1.pkl"
def main():
    #feature_engineering=FeatureExtraction(input_path,artifacts_dir)
    #feature_engineering.run()
    #model_Selection=Model_Selection(artifacts_dir)
    #model_Selection.run()
    #error_analyse=ErrorAnalyzer(data_path=data_path,artifacts_dir=artifacts_dir,model_path=model2_path)
    #error_analyse.run()
    inferencer = Sentiment_Inference(
        vectorizer_path="artifacts/tf_idf_vectorizer_v1.pkl",
        model_path=model2_path)
    
    sample_text = """The medicine reduced my headache significantly but caused mild nausea."""
    result = inferencer.predict(sample_text)
    print(result)
    
    
if __name__=="__main__":
    main()