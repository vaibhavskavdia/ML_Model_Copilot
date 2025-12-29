import os
import joblib
import pandas as pd

from logger import logger


class ErrorAnalyzer:
    def __init__(self,data_path: str,artifacts_dir: str,model_path: str,output_dir: str = "artifacts/error_analysis/Linear_svc"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.model_path = model_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.X_path = os.path.join(artifacts_dir, "X_tfidf_vectorizer_v1.pkl")
        self.y_path = os.path.join(artifacts_dir, "y_labels_v1.pkl")

    def load_data(self):
        logger.info("Loading data, features, labels, and model")

        df = pd.read_csv(self.data_path)
        X = joblib.load(self.X_path)
        y = joblib.load(self.y_path)
        model = joblib.load(self.model_path)

        return df, X, y, model

    def analyze(self):
        df, X, y_true, model = self.load_data()

        logger.info(f"Running predictions for error analysis for {self.model_path}")
        y_pred = model.predict(X)

        df["true_label"] = y_true
        df["predicted_label"] = y_pred

        # False Positives: predicted 1 but actually 0
        false_positives = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)]

        # False Negatives: predicted 0 but actually 1
        false_negatives = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)]

        logger.info(f"False Positives count: {len(false_positives)}")
        logger.info(f"False Negatives count: {len(false_negatives)}")

        fp_path = os.path.join(self.output_dir, "false_positives.csv")
        fn_path = os.path.join(self.output_dir, "false_negatives.csv")

        false_positives[["clean_text", "true_label", "predicted_label"]].to_csv(fp_path, index=False)
        false_negatives[["clean_text", "true_label", "predicted_label"]].to_csv(fn_path, index=False)
        logger.info(f"Saved false positives at {fp_path}")
        logger.info(f"Saved false negatives at {fn_path}")

    def run(self):
        self.analyze()
