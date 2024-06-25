from llama_index.llms.openai import OpenAI

from textgradlite.llmactions.prompts import REASON_AND_ANSWER_PROMPT, EVALUATE_PROMPT_RESPONSE, UPDATE_PROMPT_FROM_EVALUATION
from textgradlite.state.StateManager import StateManager
from textgradlite.DEFAULTS import DEFAULT_OPENAI_MODEL, DEFAULT_EVALUATOR_MODEL, MAX_USER_AGENT_ITERATIONS

from pydantic import BaseModel

class MMLU(BaseModel):
	question: str
	answers: list[str]
	correct: str
    
class TextGradLiteAgent():
    def __init__(
        self,
        question: MMLU,
        llm = None,
        evaluator = None,
        state: dict = None,
        max_iterations: int = MAX_USER_AGENT_ITERATIONS,
        verbose: bool = False
    ):
        
        self.llm = llm \
            if llm \
            else OpenAI(model=DEFAULT_OPENAI_MODEL)
        self.evaluator = evaluator \
            if evaluator \
            else OpenAI(model=DEFAULT_EVALUATOR_MODEL)
        self.state_manager = StateManager(state)
        
        self.verbose = verbose
        self.max_iterations = max_iterations

        ###
        self._question = question.question
        self._answers = question.answers
        self._correct = question.correct
        self.state_manager.add_initial_prompt(REASON_AND_ANSWER_PROMPT.format(
            question=question.question,
            options=self._parse_options(question.answers)
        ))
        self._current_prompt = self.state_manager.current_iteration["initial_prompt"]
        self._current_response = None
        self._current_feedback = None

    def _parse_options(
            self, 
            options: list[str]
        ) -> str:
        return "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

    def generate_response(
            self,
        ) -> str:
        res = self.llm.complete(self._current_prompt).text
        return res

    async def agenerate_response(
            self,
        ) -> str:
        res = await self.llm.acomplete(self._current_prompt)
        return res.text

    def evaluate_response(
            self
        ) -> str:
        assert self._current_response, "Response is not set"
        return self.evaluator.complete(EVALUATE_PROMPT_RESPONSE.format(
            prompt=self._current_prompt,
            response=self._current_response
        )).text
            
    async def aevaluate_response(
            self, 
            response: str, 
        ) -> dict:
        assert response, "Response is not set"
        res = await self.evaluator.acomplete(EVALUATE_PROMPT_RESPONSE.format(
            prompt=self._current_prompt,
            response=response
        ))
        return res.text

    def update_prompt(
            self
        ) -> str:
        assert self._current_feedback, "Feedback is not set"
        return self.llm.complete(UPDATE_PROMPT_FROM_EVALUATION.format(
            prompt=self._current_prompt,
            feedback=self._current_feedback
        )).text

    async def aupdate_prompt(
            self
        ) -> str:
        assert self._current_feedback, "Feedback is not set"
        res = await self.llm.acomplete(UPDATE_PROMPT_FROM_EVALUATION.format(
            prompt=self._current_prompt,
            feedback=self._current_feedback
        ))
        return res.text
    
    def iterate(
            self
        ) -> dict:
        self._current_response = self.generate_response()
        self.state_manager.add_response(self._current_response)
        self._current_feedback = self.evaluate_response()
        self.state_manager.add_feedback(self._current_feedback)
        self._current_prompt = self.update_prompt()
        self.state_manager.add_updated_prompt(self._current_prompt)
        return self.state_manager.current_iteration
    
    async def aiterate(
            self
        ) -> dict:
        self._current_response = await self.agenerate_response()
        self.state_manager.add_response(self._current_response)
        self._current_feedback = await self.aevaluate_response(self._current_response)
        self.state_manager.add_feedback(self._current_feedback)
        self._current_prompt = await self.aupdate_prompt()
        self.state_manager.add_updated_prompt(self._current_prompt)
        return self.state_manager.current_iteration
    
    def run(
            self
        ) -> dict:
        for i in range(self.max_iterations):
            if self.verbose:
                print(f"Iteration {i}")
            self.iterate()
            if i == self.max_iterations - 1:
                break
            self.state_manager.next_iteration()
        return self.state_manager.current_iteration