import os
from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "transformer-distilbert"

TRANSFORMER_DIR = f"src/models/{MODEL_NAME}"


def load_split_of(path: str) -> pd.DataFrame :
    print(f"Loading files {path}")

    df= pd.read_csv(path)
    return df



def df_to_hf(df : pd.DataFrame) -> Dataset:
    df = df[["full_text" , "label"]].copy()
    ds = Dataset.from_pandas(df , preserve_index=False)
    return ds


def tokenize_batch(examples : Dict[str , Any] , hf_tokenizer) -> Dict[str , Any] : 

    return hf_tokenizer (
        examples["full_text"],
        truncation=True,
        max_length = 256, # i think would perfectly fit for the typical size news 
        padding = "max_length"
    )

def build_and_model() :  
    print(f"Loading model {PRE_TRAINED_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels = 2
    )

    return tokenizer , model


def calculate_metrics (calc_pred : EvalPrediction) -> Dict[str, float]:
    # raw metrics without softmax func
    logits = calc_pred.predictions
    labels = calc_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels , predictions)
    f1 = f1_score(labels , predictions , average="macro")

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
    }

    print("\n=== Evaluation Metrics ===")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nF1 score: {f1:.4f}")
    print(classification_report(labels, predictions, digits=4))

    return metrics

def printing_metrics(name : str , metrics : Dict[str , Any]) -> None:


    acc = metrics.get("eval_accuracy" , metrics.get("accuracy"))
    f1 = metrics.get("eval_f1_score" , metrics.get("f1_score"))
    loss = metrics.get("eval_loss" , metrics.get("loss"))
     
    print("===  Printing Metrics === ")
    
    if loss is not None : 
        print("\n===  Loss  ===  , {loss:.4f}")
    if acc is not None :
        print(f"\n===  Accuracy ===  , {acc * 100:.3f} %")
    if f1 is not None : 
        print(f"\n===  F1  === {f1 * 100:.3f} %")



def train_and_eval() -> None:
    

    df_train = load_split_of(TRAIN_PATH)
    df_val = load_split_of(VAL_PATH)
    df_test = load_split_of(TEST_PATH)

    #converting to Hugging face Dataset format
 
    ds_train = df_to_hf(df_train)
    ds_val = df_to_hf(df_val)
    ds_test = df_to_hf(df_test)

    hf_tokenizer , model = build_and_model()
    
    print("Tokenizing data sets ...")
    tokenize_fn = partial(tokenize_batch, hf_tokenizer=hf_tokenizer)

    ds_train_tok = ds_train.map ( 
        tokenize_fn,
        batched=True,
    )

    ds_val_tok = ds_val.map (
        tokenize_fn,
        batched=True,
    )

    ds_test_tok = ds_test.map (
        tokenize_fn,
        batched=True,
    )


    ds_train_tok = ds_train_tok.rename_column("label" , "labels")
    ds_val_tok = ds_val_tok.rename_column("label" , "labels")
    ds_test_tok = ds_test_tok.rename_column("label" , "labels")

    ds_train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=os.path.join(TRANSFORMER_DIR, "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
    )

    hf_train_dataset: Any = ds_train_tok
    hf_val_dataset: Any = ds_val_tok
    hf_test_dataset: Any = ds_test_tok

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        compute_metrics=calculate_metrics,
    )

    print("Started training ... ")
    trainer.train()

    train_metrics = trainer.evaluate(eval_dataset=hf_train_dataset)
    printing_metrics("TRAIN" , train_metrics)
    
    eval_metrics = trainer.evaluate(eval_dataset=hf_val_dataset)
    printing_metrics("VAL" , eval_metrics)
    
    test_metrics = trainer.evaluate(eval_dataset=hf_test_dataset)
    printing_metrics("TEST" , test_metrics)
    
     
    print(f"\n Saving model and tokenizer to: {TRANSFORMER_DIR}")
    os.makedirs(TRANSFORMER_DIR, exist_ok=True)
    model.save_pretrained(TRANSFORMER_DIR)
    hf_tokenizer.save_pretrained(TRANSFORMER_DIR)


def main() : 
    
    if torch.cuda.is_available():
        print("Using cuda ..")
    else : 
        print("CUDA not available . Using CPU ...")
    
    train_and_eval()


if __name__ == "__main__":
    main()
