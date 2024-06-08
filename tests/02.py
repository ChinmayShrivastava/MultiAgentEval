import pandas as pd
from agents.Agent import MMLU, arun_eval
import asyncio

LAST_5_SUBJECTS = [
	'global_facts_test.csv',
	'college_mathematics_test.csv',
	'abstract_algebra_test.csv',
	# 'moral_scenarios_test.csv', # too costly
	'high_school_physics_test.csv'
]

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
	return await arun_eval(questions)

async def main(path, filename, results_path='results/02'):
	responses, answers = await eval(path)
	# save responses and answers to file
	with open(f'{results_path}/{filename}_responses.txt', 'w') as f:
		for i, response in enumerate(responses):
			if response:
				f.write(f'{i+1}. {response}\n')
	with open(f'{results_path}/{filename}_answers.txt', 'w') as f:
		for i, answer in enumerate(answers):
			if answer:
				f.write(f'{i+1}. {answer}\n')

def check_file_processed(filename, results_path='results/02'):
	# if filename_responses.txt or filename_answers.txt exist, return True
	return os.path.exists(f'{results_path}/{filename}_responses.txt') or os.path.exists(f'{results_path}/{filename}_answers.txt')

if __name__ == "__main__":
	# filename = "clinical_knowledge_test"
	# path = f"data/test/{filename}.csv"
	# asyncio.run(main(path, filename))

	import os

	for i, filename in enumerate(LAST_5_SUBJECTS):
		if filename.endswith('.csv') and not check_file_processed(filename[:-4]):
			print(f"Processing {filename} - {100*(i+1)/len(LAST_5_SUBJECTS)}%")
			path = f"data/test/{filename}"
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