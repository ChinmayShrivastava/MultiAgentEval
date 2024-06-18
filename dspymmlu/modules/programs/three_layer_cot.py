import dspy
from modules.signatures import (AnswerQuestion, CoreQuestion,
                                ProblemSolvingInfo, TopTwoOptions)


def parseTuple(input_string):
    stripped_string = input_string.strip("()")
    split_string = stripped_string.split(",")
    tuple_elements = tuple(element.strip() for element in split_string)
    return tuple_elements

def formatTopTwoOptions(a, b, c, d, top_two_options: tuple):
    cleaned_options = ""
    for option in top_two_options:
        if option.lower() == 'a':
            cleaned_options += f"a. {a}\n"
        elif option.lower() == 'b':
            cleaned_options += f"b. {b}\n"
        elif option.lower() == 'c':
            cleaned_options += f"c. {c}\n"
        elif option.lower() == 'd':
            cleaned_options += f"d. {d}\n"
    return cleaned_options

TOP_TWO_OPTIONS_RATIONALE_TYPE = dspy.OutputField(
    prefix=("Given the question, think step by step. For each option, reason out why it might be the answer. "
    "Then, compare the two most likely options and return the top two most likely options. Let's think step by step to"),
    desc="${produce the answer}. We ...",
)

FINAL_ANSWER_RATIONALE_TYPE = dspy.OutputField(
    prefix=("Given the question, and the two most likely options, let's think step by step to find the correct answer. "
    "Let's think step by step to"),
    desc="${produce the answer}. We ...",
)

class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)
        self.toptwooptions = dspy.ChainOfThought(TopTwoOptions, rationale_type=TOP_TWO_OPTIONS_RATIONALE_TYPE)

        self.prog = dspy.ChainOfThought(AnswerQuestion)

        self.responses = []

    def forward(self, question, subject, a, b, c, d):

        self._core_question = self.core_question(question=question)['core_question']
        self._info = self.info(question=question)['info']

        self._toptwooptions = self.toptwooptions(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=self._core_question,
            info=self._info,
        )
        self._toptwooptions, self._twooptionsrationale = self._toptwooptions['toptwooptions'], self._toptwooptions['rationale']
        self._toptwooptions = parseTuple(self._toptwooptions)

        self._answer = self.prog(
            question=question,
            subject=subject,
            core_question=self._core_question,
            info=self._info,
            TopTwoOptions=formatTopTwoOptions(a, b, c, d, self._toptwooptions)
        )

        self.responses.append({
                "question": question,
                "core_question": self._core_question,
                "info": self._info,
                "formattedtoptwooptions": formatTopTwoOptions(a, b, c, d, self._toptwooptions),
                "twooptionsrationale": self._twooptionsrationale,
                "toptwooptions": str(self._toptwooptions),
                "rationale": self._answer['rationale'],
                "answer": self._answer['answer']
            })

        return self._answer