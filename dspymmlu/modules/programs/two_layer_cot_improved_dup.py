import dspy
from modules.signatures import DUPhint, QADUPset


class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.hints = dspy.Predict(DUPhint)

        self.prog = dspy.ChainOfThought(QADUPset)

        self.responses = []

    def forward(self, question, subject, a, b, c, d):
        self._hints = self.hints(question=question)['hints']

        self._answer = self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            hints=self._hints
        )

        self.responses.append({
            "question": question,
            "subject": subject,
            "hints": self._hints,
            "rationale": self._answer['rationale'],
            "answer": self._answer['answer']
        })

        return self._answer