# utility functions for the other scripts
# imports 
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# load sst-2 dataset from huggingface
def load_sst():
    """Load the sst2 dataset
    See also https://huggingface.co/datasets/stanfordnlp/sst2
    
    Parameters
    ----------
    None
    
    Returns
    -------
    pd.DataFrame
        The sst2 dataset
    """
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])
    return df


def subsample(df: pd.DataFrame, label_col: str = "label", n: int = 100):
    """Subsample the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to subsample
    label_col :  str
        The column containing the labels
    n : int
        The number of samples to get
        
    Returns
    -------
    pd.DataFrame
        The subsampled dataframe
    """
    sample = df.groupby(label_col).sample(n).reset_index(drop=True) # get n of each label
    return sample


def evaluate(y_true, y_pred):
    """Evaluate the model
    
    Parameters
    ----------
    y_true : list
        The true labels
    y_pred : list
        The predicted labels
        
    Returns
    -------
    dict
        A dictionary containing the accuracy and f1 score
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1}


def save_results(metrics: dict, path: str, task: str):
    """Save the results to a file
    
    Parameters
    ----------
    metrics : dict
        The metrics to save
    path : str
        The path to save the metrics
    """
    with open(path +  task, "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {metrics['duration']} seconds")
        
        
# utilities for rag.py
import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SummaryIndex
from llama_index.core import Settings


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        embed_model = Settings.embed_model
        index = VectorStoreIndex.from_documents(documents=data, embed_model=embed_model, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))
    
    return index


def get_summary(data, index_name):
    index = None
    if not os.path.exists(index_name + "_summary"):
        index = SummaryIndex.from_documents(documents=data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name + "_summary")
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name + "_summary"))
    
    return index