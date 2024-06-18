import dspy
from modules.signatures.one_layer import QAset


class COT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QAset)

        self.responses = []

    def forward(self, question, subject, a, b, c, d):
        self._answer = self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d
        )

        self.responses.append({
            "question": question,
            "rationale": self._answer['rationale'],
            "answer": self._answer['answer'],
        })

        return self._answer