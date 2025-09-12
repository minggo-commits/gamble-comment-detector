# src/preprocessing.py
import re
import string
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_texts(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]

def build_vectorizer(texts, max_features=15000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def save_vectorizer(vectorizer, path="model/vectorizer.joblib"):
    joblib.dump(vectorizer, path)
