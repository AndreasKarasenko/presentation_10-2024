# Few-Shot classification is the task of categorizing a text with very few training examples.
# In the case of LLMs like Llama3, we feed both the text and a few known examples to the model and get the most likely label.

# example using ollama and scikit-ollama
from utils import load_sst, subsample, evaluate, save_results
from time import time
from skollama.classification import FewShotOllamaClassifier


# load the sst2 dataset
df = load_sst()

# subsample the dataset
df = subsample(df, n=200) # 100 positive and 100 negative samples
df_train = df.groupby("label").head(1).reset_index(drop=True).sample(frac=1) # k-shot with k=3, shuffle to avoid recency bias
df_test = df.groupby("label").tail(100).reset_index(drop=True)

X_train = df_train["sentence"]
y_train = df_train["label"]

X_test = df_test["sentence"]
y_test = df_test["label"]

# # create a few-shot classifier
# clf = FewShotOllamaClassifier(model="gemma2:9b")
# clf.fit(X_train, y_train) # no actual fitting is happending, this is solely for data validation

# # predict the labels
# start = time()
# preds = clf.predict(X_test) # predict the labels
# end = time()

# # evaluate the model
# metrics = evaluate(y_test, preds)
# metrics["duration"] = end-start
# print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
# save_results(metrics, "../results/", "few_shot.txt")

# ### use multi-processing to speed up the predictions
# clf = FewShotOllamaClassifier(model="gemma2:9b")
# clf.fit(X_train, y_train) # no actual fitting is happending, this is solely for data validation

# # predict the labels
# start = time()
# preds = clf.predict(X_test, num_workers=2) # predict the labels using 2 processes
# end = time()

# # evaluate the model
# metrics = evaluate(y_test, preds)
# metrics["duration"] = end-start
# print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
# save_results(metrics, "../results/", "few_shot_multi.txt")

# ### use more examples
# df_train = df.groupby("label").head(3).reset_index(drop=True).sample(frac=1) # k-shot with k=3, shuffle to avoid recency bias
# X_train = df_train["sentence"]
# y_train = df_train["label"]

# # create a few-shot classifier
# clf = FewShotOllamaClassifier(model="gemma2:9b")
# clf.fit(X_train, y_train) # no actual fitting is happending, this is solely for data validation

# # predict the labels
# start = time()
# preds = clf.predict(X_test, num_workers=2) # predict the labels
# end = time()

# # evaluate the model
# metrics = evaluate(y_test, preds)
# metrics["duration"] = end-start
# print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
# save_results(metrics, "../results/", "few_shot_multi_3.txt")

### use a better prompt
FEW_SHOT = """
What is the most likely sentiment (using these labels: {labels}) for the review at the end?
Provide your response in a JSON format containing a single key `label`.
Use these training examples:
Training data:
{training_data}

What is the sentiment for the following review?
Review: ```{x}```
"""

# create a few-shot classifier
clf = FewShotOllamaClassifier(model="gemma2:9b", prompt_template=FEW_SHOT)
clf.fit(X_train, y_train)

# predict the labels
start = time()
preds = clf.predict(X_test, num_workers=2) # predict the labels
end = time()

# evaluate the model
metrics = evaluate(y_test, preds)
metrics["duration"] = end-start
print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}, Duration: {end-start} seconds")
save_results(metrics, "../results/", "few_shot_prompt.txt")