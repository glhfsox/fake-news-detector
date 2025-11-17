import os 
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_TRUE_PATH = "data/raw/True.csv"
RAW_FAKE_PATH = "data/raw/Fake.csv"
PROCESSED_DIR = "data/processed"

COLUMN_TITLE = "title"
COLUMN_TEXT = "text"

def load_and_merge(raw_true_path : str , raw_fake_path : str)-> pd.DataFrame:
    df_true = pd.read_csv(raw_true_path)
    df_fake = pd.read_csv(raw_fake_path)

    df_true["full_text"] = (
        df_true[COLUMN_TITLE].fillna("") + " " + df_true[COLUMN_TEXT].fillna("")
     )
    
    df_fake["full_text"] = (
        df_fake[COLUMN_TITLE].fillna("") + " " + df_fake[COLUMN_TEXT].fillna("")
     )
    
    df_true["label"] = 0
    df_fake["label"] = 1
        
    #reuniting and shuffling our true and fake files 
    df = pd.concat([df_true , df_fake] , ignore_index=True )
    df = df[df["full_text"].str.strip()!=""]
    df = df.sample(frac=1.0 , random_state=42).reset_index(drop=True)


    return df


def split_and_save(df: pd.DataFrame, out_dir: str) -> None:

    os.makedirs(out_dir,exist_ok=True)
    #setting our proportions , e.g 70% of data for training , 15% for eval. and 15% for testing
    # and then saving those parts as csv files
    train , temp = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )
    
    test , val = train_test_split (
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["label"],
    )

    train.to_csv(os.path.join(out_dir , "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir , "test.csv") , index=False)
    val.to_csv(os.path.join(out_dir , "val.csv"), index=False)

def main () : 
    df = load_and_merge(RAW_TRUE_PATH , RAW_FAKE_PATH)
    split_and_save(df , PROCESSED_DIR)



if __name__ == "__main__":
    main()