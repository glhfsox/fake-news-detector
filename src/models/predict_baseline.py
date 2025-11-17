import os 
import joblib
import numpy as np

#script that predicts whether the news you entered is real or not 
# returns probability

MODEL_PATH = "src/models/baseline_tfidf_logreg.joblib"

LABEL_TO_NAME = { 
    0:"REAL",
    1:"FAKE",
}


def load_model() : 
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'file not found : {MODEL_PATH}')
    
    print(f"Loading model from {MODEL_PATH}")
    model= joblib.load(MODEL_PATH)
    return model

def predict_text(model , text : str):


    if not text.strip():
        raise ValueError("Empty text . Please enter something")

    probs = model.predict_proba([text])[0]
    pred_label = int(model.predict([text])[0])

    classes = model.classes_

    index_real = int(np.where(classes == 0)[0][0])
    index_fake = int(np.where(classes == 1)[0][0])
    
    real_probability = probs[index_real]
    fake_probability = probs[index_fake]

    return pred_label , LABEL_TO_NAME[pred_label] , real_probability , fake_probability


def main() : 
    model = load_model()

    print("\nFake News Baseline (TF-IDF + Logistic Regression)")
    print("Enter the next of your news . 'q' or enter for escaping.\n")

    while True:
        text = input("News: ").strip()
        if text.lower() in {"q" , "quit" ,"exit"} or text=="":
            print("Escaping..")
            break
        try:
            label , label_name , real_p , fake_p = predict_text(model , text)
        except ValueError as e:
            print(f"Error {e}")
            continue

        print(f"\nPredicted class : {label_name} (label = {label})")
        print(f"Probability of REAL {real_p * 100:.3f} %")
        print(f"Probability of FAKE: {fake_p * 100:.3f} %")


if __name__ == "__main__":
    main()