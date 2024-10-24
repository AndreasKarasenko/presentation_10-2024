### reproduce the sentiment task with Hartmann et al. (2023) model

# imports
import pandas as pd
import numpy as np
from utils import load_sst, subsample, evaluate, save_results
from time import time
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset


# load the sst2 dataset
df = load_sst()

# subsample the dataset
df = subsample(df, n=100) # 100 positive and 100 negative samples
df = df.loc[:,["sentence", "label"]]
df.rename(columns={"sentence":"text"}, inplace=True) # rename the column to adhere to the transformer API

# build the model
checkpoint = "siebert/sentiment-roberta-large-english"
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint) # don't need to set number of labels
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# prepare the dataset
test = Dataset.from_pandas(df)
test = test.map(preprocess_function, batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
bs = 2

test = test.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False, # don't shuffle or we can't evaluate!
    batch_size=bs,
    collate_fn=collator
)

start = time()
preds = model.predict(test)
end = time()
predicted_test = np.argmax(preds.logits, axis=1)
actual_test = df["label"]

metrics = evaluate(actual_test, predicted_test)
metrics["duration"] = end-start

save_results(metrics, "../results/", "transformer_sentiment.txt")