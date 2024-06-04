from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from agents.prompts import DEFAULT_AGENT_SPAWN_PROMPT, DEFAULT_AGENT_GENERATED_REASON, DEFAULT_ANSWER_PROMPT

import re


DEFAULT_MODEL = 'gpt-4o'
DEFAULT_AGENTS_TO_SPAWN = 4

class MMLU(BaseModel):
	question: str
	answers: list[str]
	correct: str

class Agent:
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
		self.spawn_agents(mmlu_question.question, self.n_agents)
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
		_a = re.search(r"Answer: ([A-D])", res)
		return _a.group(1)

	def spawn_agents(self, question: str) -> None:
		_prompt = DEFAULT_AGENT_SPAWN_PROMPT.format(n=self.n_agents, question=question)
		res = self.llm.complete(_prompt).text
		self._agent_tasks = res.split("\n")
		print(self._agent_tasks)

	def collect_reasonings(self) -> list[str]:
		_reasonings = []
		for task in self._agent_tasks:
			_prompt = DEFAULT_AGENT_GENERATED_REASON.format(question=self._question.question, perspective=task)
			res = self.llm.complete(_prompt).text
			_reasonings.append(res)
			print(res)
		return _reasonings

	def reset(self) -> None:
		self._agent_tasks = []
		self._question = None

if __name__ == "__main__":
	agent = Agent()
	question = MMLU(
		question="What is the capital of France?",
		answers=["Paris", "London", "Berlin", "Madrid"],
		correct="Paris"
	)
	agent.question = question
	print(agent.get_response())