from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from .prompts import DEFAULT_AGENT_SPAWN_PROMPT, DEFAULT_AGENT_GENERATED_REASON, DEFAULT_ANSWER_PROMPT

import asyncio
import re
import random


DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_AGENTS_TO_SPAWN = 4
DEFAULT_BATCH_SIZE = 100

class MMLU(BaseModel):
	question: str
	answers: list[str]
	correct: str

class Agent:
	BATCH_SIZE = DEFAULT_BATCH_SIZE
	
	def __init__(
			self,
			llm: OpenAI = None,
			n_agents: int = None
			):
		self.llm = llm or OpenAI(model=DEFAULT_MODEL)
		self.n_agents = n_agents or DEFAULT_AGENTS_TO_SPAWN

		self._agent_tasks = []
		self._question = None
	
	@property
	def question(self):
		return self._question
	
	@question.setter
	def question(self, question: MMLU):
		self._question = question
	
	def get_response(self) -> str:
		if not self._question:
			raise ValueError("No question has been set.")
		mmlu_question = self._question
		self.spawn_agents(mmlu_question.question)
		reasonings = self.collect_reasonings()
		return self.get_answer(mmlu_question.question, mmlu_question.answers, reasonings)

	def get_answer(self, question: str, answers: list[str], reasonings: list[str]) -> str:
		_answers = ""
		for i, answer in enumerate(answers):
			_answers += f"{chr(65 + i)}. {answer}\n"
		_prompt = DEFAULT_ANSWER_PROMPT.format(
			question=question, options=_answers, 
			information="\n".join(reasonings))
		res = self.llm.complete(_prompt).text
		# print(res)
		_a = re.search(r"Answer: ([A-D])", res)
		try:
			return res, _a.group(1)
		except:
			return res, None

	def spawn_agents(self, question: str) -> None:
		_prompt = DEFAULT_AGENT_SPAWN_PROMPT.format(n=self.n_agents, question=question)
		res = self.llm.complete(_prompt).text
		self._agent_tasks = res.split("\n")
		# print(self._agent_tasks)

	def collect_reasonings(self) -> list[str]:
		_reasonings = []
		for task in self._agent_tasks:
			_prompt = DEFAULT_AGENT_GENERATED_REASON.format(question=self._question.question, perspective=task)
			res = self.llm.complete(_prompt).text
			_reasonings.append(res)
			# print(res)
		return _reasonings
	
	async def aget_response(self) -> str:
		if not self._question:
			raise ValueError("No question has been set.")
		mmlu_question = self._question
		await self.aspawn_agents(mmlu_question.question)
		reasonings = await self.acollect_reasonings()
		_answer = await self.aget_answer(mmlu_question.question, mmlu_question.answers, reasonings)
		return _answer
	
	async def aget_answer(self, question: str, answers: list[str], reasonings: list[str]) -> str:
		_answers = ""
		for i, answer in enumerate(answers):
			_answers += f"{chr(65 + i)}. {answer}\n"
		_prompt = DEFAULT_ANSWER_PROMPT.format(
			question=question, options=_answers, 
			information="\n".join(reasonings))
		res = await self.llm.acomplete(_prompt)
		res = res.text
		# print(res)
		_a = re.search(r"Answer: ([A-D])", res)
		try:
			return res, _a.group(1)
		except:
			return res, None
	
	async def aspawn_agents(self, question: str) -> None:
		_prompt = DEFAULT_AGENT_SPAWN_PROMPT.format(n=self.n_agents, question=question)
		res = await self.llm.acomplete(_prompt)
		res = res.text
		self._agent_tasks = res.split("\n")

	async def acollect_reasonings(self) -> list[str]:
		_reasonings = []
		atasks = []
		for task in self._agent_tasks:
			_prompt = DEFAULT_AGENT_GENERATED_REASON.format(question=self._question.question, perspective=task)
			atasks.append(asyncio.create_task(
				self.llm.acomplete(_prompt)
			))
		_res = await asyncio.gather(*atasks)
		for res in _res:
			_reasonings.append(res.text)
		return _reasonings

	def reset(self) -> None:
		self._agent_tasks = []
		self._question = None

def run_eval(
		agent: Agent,
		questions: list[MMLU],
) -> list[str]:
	import tqdm
	responses = []
	answers = []
	for question in tqdm.tqdm(questions):
		agent.reset()
		agent.question = question
		res, answer = agent.get_response()
		responses.append(res)
		answers.append(answer)
	return responses, answers

async def arun_eval(
		questions: list[MMLU],
		batchsize: int = DEFAULT_BATCH_SIZE
) -> list[str]:
    import tqdm
    responses = []
    answers = []
    for i in tqdm.tqdm(range(0, len(questions), batchsize)):
        batch = questions[i:i+batchsize]
        atasks = []
        for question in batch:
            a = Agent()
            a.reset()
            a.question = question
            atasks.append(asyncio.create_task(a.aget_response()))
        print("Awaiting responses...")
        _res = await asyncio.gather(*atasks)
        if (i+1)%4 == 0:
            print("sleeping for 60 seconds...")
            await asyncio.sleep(60)
        for res, answer in _res:
            responses.append(res)
            answers.append(answer)
    return responses, answers


# if __name__ == "__main__":
# 	agent = Agent()
# 	question = MMLU(
# 		question="""As Seller, an encyclopedia salesman, approached the grounds on which Hermit's house was situated,
# he saw a sign that said, "No salesmen. Trespassers will be prosecuted. Proceed at your own risk."
# Although Seller had not been invited to enter, he ignored the sign and drove up the driveway toward
# the house. As he rounded a curve, a powerful explosive charge buried in the driveway exploded, and
# Seller was injured. Can Seller recover damages from Hermit for his injuries?""",
# 		answers=["Yes, unless Hermit, when he planted the charge, intended only to deter, not harm, intruders.", 
# 		   "Yes, if Hermit was responsible for the explosive charge under the driveway.", 
# 		   "No, because Seller ignored the sign, which warned him against proceeding further.", 
# 		   "No, if Hermit reasonably feared that intruders would come and harm him or his family."],
# 		correct="B"
# 	)
# 	agent.question = question
# 	print(agent.get_response())