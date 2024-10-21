# Zero-Shot Classification is the task of categorizing a text without any training data.
# In the case of LLMs like Llama3, we can feed the text and the candidate labels to the model and get the most likely label.

# example using ollama and scikit-ollama
from utils import load_sst, subsample, evaluate, save_results
from time import time
from skollama.classification import ZeroShotOllamaClassifier


# load the sst2 dataset
df = load_sst()

# subsample the dataset
df = subsample(df, n=100) # 100 positive and 100 negative samples
X = df["sentence"]
y = df["label"]

# create a zero-shot classifier
clf = ZeroShotOllamaClassifier(model = "gemma2:9b")
clf.fit(X, y) # no actual fitting is happending, this is solely for data validation

# predict the labels
start = time()
preds = clf.predict(X) # predict the labels
end = time()

# # evaluate the model
metrics = evaluate(y, preds)
metrics["duration"] = end-start
print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
save_results(metrics, "../results/", "zero_shot.txt")

### use multi-processing to speed up the predictions
# predict the labels
start = time()
preds = clf.predict(X, num_workers=2) # predict the labels using 2 processes
end = time()

# evaluate the model
metrics = evaluate(y, preds)
metrics["duration"] = end-start
print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
save_results(metrics, "../results/", "zero_shot_MULTI.txt")

### change the prompt text to better fit sentiment analysis
SENTIMENT_PROMPT = """
What is the most likely sentiment (using these labels: {labels}) for the following review
(Provide your response in a JSON format containing a single key `label`):

Review: ```{x}```
"""
clf = ZeroShotOllamaClassifier(model = "gemma2:9b", prompt_template=SENTIMENT_PROMPT)
clf.fit(X,y)

# predict the labels
start = time()
preds = clf.predict(X) # predict the labels
end = time()

# # evaluate the model
metrics = evaluate(y, preds)
metrics["duration"] = end-start
print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
save_results(metrics, "../results/", "zero_shot_SENT.txt")