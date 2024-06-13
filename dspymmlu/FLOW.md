Signature -> representing one qa set in the database. the signature will also contain the subject that the question belongs to.

DSPy Program -> a CoT based program that takes in a qa set and predicts the answer

Metric -> exact match for the answer and the prediction

Optimizer -> BootstrapFewShot (takes the validation set as the training set). the idea is to use qa sets to train and the test set (unseen data) to test the model's accuracy

CONFIG:

Model - gpt-3.5-turbo
