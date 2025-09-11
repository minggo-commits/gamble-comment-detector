import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocessing import clean_text

# Load dataset
data = pd.read_csv("data/komentar.csv")
data["clean_text"] = data["komentar"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"], data["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluasi
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
