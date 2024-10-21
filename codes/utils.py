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