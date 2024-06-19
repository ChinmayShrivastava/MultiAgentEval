import dspy
from modules.signatures import (AnswerQuestionWithSteps, CoreQuestion,
                                ProblemSolvingInfo, Steps)


STEPS_RATIONALE_TYPE = dspy.OutputField(
    prefix=("Given the question, think step by step. Strategize a list of steps to solve the problem. "
    "Let's think step by step to"),
    desc="${help solve the question}. We ...",
)

FINAL_ANSWER_RATIONALE_TYPE = dspy.OutputField(
    prefix=("Given the question, think step by step. Use the given steps to solve the problem. Add more steps if needed. "
    "Let's think step by step to"),
    desc="${produce the answer}. We ...",
)

class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)
        self.steps = dspy.ChainOfThought(Steps, rationale_type=STEPS_RATIONALE_TYPE)

        self.prog = dspy.ChainOfThought(AnswerQuestionWithSteps, rationale_type=FINAL_ANSWER_RATIONALE_TYPE)

        self.responses = []

    def forward(self, question, subject, a, b, c, d):

        self._core_question = self.core_question(question=question)['core_question']
        self._info = self.info(question=question)['info']

        self._steps = self.steps(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=self._core_question,
            info=self._info,
        )
        self._steps = self._steps['steps']

        self._answer = self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=self._core_question,
            info=self._info,
            steps=self._steps
        )

        self.responses.append({
                "question": question,
                "core_question": self._core_question,
                "info": self._info,
                "steps": self._steps,
                "rationale": self._answer['rationale'],
                "answer": self._answer['answer']
            })

        return self._answer