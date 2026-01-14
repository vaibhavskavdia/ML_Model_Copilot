import joblib

vec = joblib.load("artifacts/tf_idf_vectorizer_v1.pkl")
print(type(vec))
print(hasattr(vec, "idf_"))

