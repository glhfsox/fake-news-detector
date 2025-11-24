import os
from typing import Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report , f1_score 
from joblib import dump

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
VAL_PATH = "data/processed/val.csv"

MODEL_DIR = "src/models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_tfidf_logreg.joblib")


def load_split(path : str) -> Tuple[pd.Series , pd.Series]:
    df = pd.read_csv(path)

    x = df["full_text"]
    y = df["label"]

    return x , y 

def build_pipeline() -> Pipeline:
    #setting up hyperparameters 
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("logreg" , clf)
        ]
    )

    return pipeline


def train_and_eval() -> None:
    x_train , y_train = load_split(TRAIN_PATH)
    x_test , y_test = load_split(TEST_PATH)
    x_val , y_val = load_split(VAL_PATH)

    pipeline = build_pipeline()

    print("Training the model TF-IDF + Logistic Regression ...")
    pipeline.fit(x_train , y_train)
    
    def printing_metrics(split_name: str, y_true, y_pred) -> None:
        acc = accuracy_score(y_true , y_pred)
        f1 = f1_score(y_true , y_pred , average="macro")
        print(f"\n === {split_name} Metrics ===")
        print(f"Accuracy : {acc * 100:.4f} %")
        print(f"F1 Macro : {f1 * 100:.4f} %")
        print(classification_report(y_true, y_pred))
 

    y_train_prediction = pipeline.predict(x_train)
    printing_metrics("Train" ,y_train ,  y_train_prediction )


    y_val_prediction = pipeline.predict(x_val)
    printing_metrics("Evaluation" , y_val , y_val_prediction)

    y_test_prediction = pipeline.predict(x_test)
    printing_metrics("Test" , y_test , y_test_prediction)


    os.makedirs(MODEL_DIR , exist_ok=True)
    joblib.dump(pipeline , MODEL_PATH)
    print(f"Model is saved to : {MODEL_PATH}")



def main() -> None:
    train_and_eval()

if __name__ == "__main__":
    main()