import dspy
from modules.signatures import CoreQuestion, ProblemSolvingInfo, QAset


class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)

        self.prog = dspy.ChainOfThought(QAset)

    def forward(self, question, subject, a, b, c, d):
        return self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=self.core_question(question=question)['core_question'],
            info=self.info(question=question)['info']
        )