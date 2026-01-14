import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ML_Model_Copilot.logger import logger

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        logger.info("text preprocessed")
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]
        logger.info("text cleaned")
        return " ".join(tokens)
