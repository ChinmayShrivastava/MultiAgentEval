import asyncio
import uuid

import pandas as pd
import mlflow

from DUPagent import MMLU, arun_eval

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "DUP"
SUBJECT="high_school_physics"
PREFIX = "claude-3-5-sonnet-majority_vote"
SPLIT="test"
TOTAL_RUNS=1

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

async def main(path, filename, total_runs=1):
    
    # set up mlflow
	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

	asnyc_runs = []
	# questions, reasons, answers, hints = await eval(path)

	for runs in range(total_runs):
		asnyc_runs.append(asyncio.create_task(eval(path)))
	
	await asyncio.gather(*asnyc_runs)

	questions = asnyc_runs[0].result()[0]
	reasons = asnyc_runs[0].result()[1]
	hints = asnyc_runs[0].result()[3]

	all_answers = []

	# for _, _, answers, _ in asnyc_runs:
		# all_answers.append(answers)

	for run in asnyc_runs:
		_, _, answers, _ = run.result()
		all_answers.append(answers)

	# for answer, take a majority vote and make it the final answer, e.g. if 3/5 say A, then A is the final answer
	answers = []
	for i in range(len(all_answers[0])):
		_answer = []
		for j in range(len(all_answers)):
			_answer.append(all_answers[j][i])
		answers.append(max(set(_answer), key=_answer.count))

	with mlflow.start_run(run_name=f"{PREFIX}_{filename}_{uuid.uuid4()}") as _:
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
		# if filename in ["college_mathematics_test.csv", "machine_learning_test.csv", "college_computer_science_test.csv", "elementary_mathematics_test.csv", "security_studies_test.csv", "virology_test.csv", "global_facts_test.csv", "abstract_algebra_test.csv", "formal_logic_test.csv", "professional_law_test.csv", "college_chemistry_test.csv", "econometrics_test.csv", "high_school_statistics_test.csv", "electrical_engineering_test.csv"]:
		if filename in ["college_mathematics_test.csv", "machine_learning_test.csv", "college_computer_science_test.csv", "elementary_mathematics_test.csv", "security_studies_test.csv", "college_chemistry_test.csv", "econometrics_test.csv", "high_school_statistics_test.csv"]:
			print(f"Processing {filename} - {100*(i+1)/len(os.listdir(f'data/{SPLIT}')):.2f}%")
			path = f"data/{SPLIT}/{filename}"
			asyncio.run(main(path, filename[:-4], total_runs=TOTAL_RUNS))
			print(f"Finished {filename}")

	# async def process_file(filename):
	# 	if filename.endswith('.csv') and not check_file_processed(filename[:-4]):
	# 		print(f"Processing {filename}")
	# 		path = f"data/test/{filename}"
	# 		await main(path, filename[:-4])
	# 		print(f"Finished {filename}")
			
	# for filename in os.listdir('data/test'):
	# 	asyncio.run(process_file(filename))