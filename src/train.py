# src/train.py
import os
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.preprocessing import preprocess_texts, build_vectorizer, save_vectorizer

DATA_PATH = "data/comments.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text","label"])
    return df

def train():
    mlflow.set_experiment("judi-comment-detector")
    df = load_data()
    texts = preprocess_texts(df["text"].tolist())
    y = df["label"].astype(str).tolist()

    # train-test split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=42
    )

    # build vectorizer
    vectorizer, X_train = build_vectorizer(X_train_texts)
    save_vectorizer(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

    # transform test
    X_test = vectorizer.transform(X_test_texts)

    # model (baseline)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    with mlflow.start_run():
        model.fit(X_train, y_train)

        # predictions
        y_pred = model.predict(X_test)

        # metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="judi")
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_metric("accuracy", float(acc))

        # save model
        model_path = os.path.join(MODEL_DIR, "saved_model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

    print("Training finished. Metrics: precision=%.4f recall=%.4f f1=%.4f acc=%.4f" % (precision, recall, f1, acc))
    print("Model saved to", model_path)

if __name__ == "__main__":
    train()
