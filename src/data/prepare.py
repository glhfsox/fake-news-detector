from cProfile import label
import os 
import csv
import pandas as pd
from pandas.core.interchange.dataframe_protocol import ColumnNullType
from sklearn.model_selection import train_test_split

RAW_TRUE_PATH =[ 
    "data/raw/True.csv",
    "data/raw/DataSet_Misinfo_TRUE.csv",
    ]
RAW_FAKE_PATH = [
    "data/raw/Fake.csv",
    "data/raw/DataSet_Misinfo_FAKE.csv",
    "data/raw/EXTRA_RussianPropagandaSubset.csv",
    ]
MIXED_SOURCES_PATH = [
    "data/raw/welfake_dataset.csv",
]


PROCESSED_DIR = "data/processed"
COLUMN_TITLE = "title"
COLUMN_TEXT = "text"

LABEL_MAP = { 
    "true": 0 , "True" : 0 , "TRUE" : 0 , 0 : 0,
    "fake": 1 , "Fake" : 1  , "FAKE": 1 , 1 : 1,
}


def normalize_text(series : pd.Series) -> pd.Series:
    return series.str.lower().str.replace(r"\s+", " " ,regex=True).str.strip()


#for those datasets whos completely fake/true
def load_labeled(paths , label ): 
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path , usecols=[COLUMN_TITLE , COLUMN_TEXT])
        # if there is no title , reading only text and adding empty title
        except ValueError:
            df = pd.read_csv(path , usecols=[COLUMN_TEXT])
            df[COLUMN_TITLE] = ""
        title = df[COLUMN_TITLE].fillna("").astype(str)
        text = df[COLUMN_TEXT].fillna("").astype(str)
        df["full_text"] = title.str.cat(text, sep=" ")
        df["label"] = label
        frames.append(df[["full_text" , "label"]])
    return pd.concat(frames , ignore_index=True) if frames else pd.DataFrame(columns=["full_text" , "label"]) 


def load_mixed(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path , usecols=[COLUMN_TITLE , COLUMN_TEXT , "label"])
        if COLUMN_TITLE not in df or COLUMN_TEXT not in df or "label" not in df:
            raise ValueError(f"Missing columns in {path} : need '{COLUMN_TITLE}' , '{COLUMN_TEXT}' , 'label'")
        df["label"] = df["label"].map(LABEL_MAP)
        df = df.dropna(subset=["label"])
        title = df[COLUMN_TITLE].fillna("").astype(str)
        text = df[COLUMN_TEXT].fillna("").astype(str)
        df["full_text"] = title.str.cat(text, sep=" ")
        frames.append(df[["full_text" , "label"]])
    return pd.concat(frames , ignore_index=True) if frames else pd.DataFrame(columns=["full_text", "label"])

def load_and_merge_multi() -> pd.DataFrame:
    df_true = load_labeled(RAW_TRUE_PATH , label = 0)
    df_fake = load_labeled(RAW_FAKE_PATH , label = 1)
    df_mixed = load_mixed(MIXED_SOURCES_PATH)

    #one can optimize this algorithm regarding memory usage by using hashes 
    # and reading csv files by batches but honestly its a bit overcomplicated for my purposes
    # here im just creating a temporary file with all the datasets given 
    # and then sorting  
    df = pd.concat([df_true , df_fake , df_mixed] , ignore_index=True)
    df = df[df["full_text"].str.strip() != ""]
    df["norm"] = normalize_text(df["full_text"])

    #removing duplicates if there arre ones 
    df = df.sort_values("label") 
    dup_mask = df.duplicated(subset="norm" , keep=False)
    conflict_mask = dup_mask & (df.groupby("norm")["label"].transform("nunique") > 1)

    df = df[~conflict_mask] #removing conflict texts
    df = df.drop_duplicates(subset="norm")
    df = df.sample(frac = 1.0 , random_state=42).reset_index(drop=True) # shuffling our texts

    return df.drop(columns="norm")




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
    df = load_and_merge_multi()
    split_and_save(df , PROCESSED_DIR)



if __name__ == "__main__":
    main()
