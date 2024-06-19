import asyncio
import uuid

import pandas as pd
import mlflow

from DUPagent import MMLU, arun_eval

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "DUP"
SUBJECT="high_school_physics"
SPLIT="val"

def getdf(path):
	return pd.read_csv(path, names=['question', 'A', 'B', 'C', 'D', 'answer'])

async def eval(path):
	df = getdf(path)
	# questions = [MMLU(question=question, answers=[A, B, C, D], correct=answer) for question, A, B, C, D, answer in df.values]
	questions = []
	for question, A, B, C, D, answer in df.values:
		try:
			questions.append(MMLU(question=str(question), answers=[str(A), str(B), str(C), str(D)], correct=str(answer)))
		except:
			print(f"Error in question: {question}")
			raise Exception
	reasons, answers, hints = await arun_eval(questions)
	return questions, reasons, answers, hints

async def main(path, filename):
    
    # set up mlflow
	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

	questions, reasons, answers, hints = await eval(path)
	with mlflow.start_run(run_name=f"{SUBJECT}_{uuid.uuid4()}") as _:
		accuracy = sum([1 for question, answer in zip(questions, answers) if question.correct.lower() == answer.lower()]) / len(questions)
		mlflow.log_param("accuracy", accuracy)
		dict_to_table = {
			"question": [question.question for question in questions],
			"hints": [hint for hint in hints],
			"reason": [reason for reason in reasons],
			"answer": [answer for answer in answers],
			"correct": [question.correct for question in questions],
        }
		mlflow.log_table(data=dict_to_table, artifact_file=f"results/{filename}_answers.json")

if __name__ == "__main__":
	# filename = "clinical_knowledge_test"
	# path = f"data/test/{filename}.csv"
	# asyncio.run(main(path, filename))

	import os

	for i, filename in enumerate(os.listdir(f'data/{SPLIT}')):
		if SUBJECT not in filename:
			continue
		if filename.endswith('.csv'):
			print(f"Processing {filename} - {100*(i+1)/len(os.listdir(f'data/{SPLIT}')):.2f}%")
			path = f"data/{SPLIT}/{filename}"
			asyncio.run(main(path, filename[:-4]))
			print(f"Finished {filename}")
			break

	# async def process_file(filename):
	# 	if filename.endswith('.csv') and not check_file_processed(filename[:-4]):
	# 		print(f"Processing {filename}")
	# 		path = f"data/test/{filename}"
	# 		await main(path, filename[:-4])
	# 		print(f"Finished {filename}")
			
	# for filename in os.listdir('data/test'):
	# 	asyncio.run(process_file(filename))