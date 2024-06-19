import asyncio
import re
import uuid

import mlflow
import tqdm
from llama_index.llms.openai import OpenAI
from prompts import DUP_GENERATE_ANSWER, GENERATE_HINTS
from pydantic import BaseModel

DEFAULT_MODEL = 'gpt-3.5-turbo'
BATCH_SIZE = 100

class MMLU(BaseModel):
    question: str
    answers: list[str]
    correct: str

class DUPagent:
    def __init__(
        self,
        llm: OpenAI = None,
    ):
        self.llm = llm or OpenAI(model=DEFAULT_MODEL)

        self._question = None

        # # set up mlflow
        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    @property
    def question(self):
        return self._question
    
    @question.setter
    def question(self, question: MMLU):
        self._question = question
    
    async def generate_hints(self, question: MMLU) -> str:
        _prompt = GENERATE_HINTS.format(question=question.question)
        res = await self.llm.acomplete(_prompt)
        return res.text
    
    async def generate_answer(
        self,
        question: str,
        answers: list[str],
        hints: str,
    ) -> str:
        _answers = ""
        for i, answer in enumerate(answers):
            _answers += f"{chr(65+i)}. {answer}\n"
        _prompt = DUP_GENERATE_ANSWER.format(question=question, options=_answers, hints=hints)
        res = await self.llm.acomplete(_prompt)
        reason, answer = self.parse_response(res.text)
        return reason, answer
    
    def parse_response(self, response: str) -> str:
        r = re.search(r"(?s)REASONING:\s*(.*?)ANSWER", response).group(1)
        a = re.search(r"ANSWER:\s*([A-E])", response).group(1)
        return r, a
    
    async def get_answer(self, question: MMLU) -> str:
        hints = await self.generate_hints(question)
        reason, answer = await self.generate_answer(question.question, question.answers, hints)
        # with mlflow.start_run(run_name=f"random_{uuid.uuid4()}") as _:
        #     mlflow.log_param("question", question.question)
        #     mlflow.log_param("answers", question.answers)
        #     mlflow.log_param("hints", hints)
        #     mlflow.log_param("correct", question.correct)
        #     mlflow.log_param("reason", reason)
        #     mlflow.log_param("answer", answer)
        return reason, answer
    
    async def generate_answers(self, questions: list[MMLU], batch=BATCH_SIZE) -> tuple[list[str], list[str]]:
        answers = []
        reasons = []
        for i in tqdm.tqdm(range(0, len(questions), batch), desc="Generating answers", leave=False):
            batch_questions = questions[i:i+batch]
            batch_answers = await asyncio.gather(*[self.get_answer(question) for question in batch_questions])
            for reason, answer in batch_answers:
                reasons.append(reason)
                answers.append(answer)
        return reasons, answers
    
# def run_eval(
# 		agent: DUPagent,
# 		questions: list[MMLU],
# ) -> list[str]:
# 	import tqdm
# 	responses = []
# 	answers = []
# 	for question in tqdm.tqdm(questions):
# 		agent.reset()
# 		agent.question = question
# 		res, answer = agent.get_response()
# 		responses.append(res)
# 		answers.append(answer)
# 	return responses, answers

async def arun_eval(
		questions: list[MMLU]
) -> list[str]:
    answers = []
    reasons = []
    agent = DUPagent()
    res = await agent.generate_answers(questions)
    reasons.extend(res[0])
    answers.extend(res[1])
    return reasons, answers
    

if __name__ == "__main__":
    agent = DUPagent()
    questions = [MMLU(
		question="What is the capital of France?",
		answers=["Paris", "London", "Berlin", "Madrid"],
		correct="A"
	),
    MMLU(
        question="What is the capital of Germany?",
        answers=["Paris", "London", "Berlin", "Madrid"],
        correct="C"
    )]
    print(asyncio.run(arun_eval(questions)))