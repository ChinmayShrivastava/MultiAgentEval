## Tests and Results

01: Base MultiAgent framework with 4 spawned agents
- Overall Accuracy - 0.6402934054977923
NOTE - Fine the test results under `01-EVAL.ipynb`, the corresponding code can be found under `01.py`

02: Base MultiAgent framework with 10 spawned agents
Experiment: only generate answers for the last 5 ranking subjects for `TEST 01`
Observations: Some subjects saw significant improvements, where others had a significant deterioration. Performing the test multiple times on teh same subject made the score vary by atleast 10%.
Conclusion: This suggests that increasing the agents without making any other changes to the model doesn't offer any significant improvements overall. There is also a randomness that need to be considered, as based on multiple runs, the answers are different. This can be improved by adjusting the model's output parameters and finetuning, however, these claims need to be tested.
NOTE - Fine the test results under `01-EVAL.ipynb`, the corresponding code can be found under `02.py`
