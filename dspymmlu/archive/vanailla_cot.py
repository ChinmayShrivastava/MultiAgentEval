import json

import dspy
import tqdm
from datahandler import get_data
from dspy_helpers import dispatch_optmizer

# CONFIGURE DEFAULTs

SUBJECT = 'anatomy'
SAVE_PATH = 'testjson.json'
DEFAULT_MODEL_STRING = 'gpt-3.5-turbo-1106'
MAX_TOKENS = 256
# DEFAULT MODEL OBJECT

DEFAULT_MODEL_OBJECT = dspy.OpenAI(
    model=DEFAULT_MODEL_STRING,
    max_tokens=512
)

dspy.configure(
    lm=DEFAULT_MODEL_OBJECT
)

# SIGATURES

class QAset(dspy.Signature):
    """
    Given a multiple choice question, the subject, and 4 options, return the alphabetical letter of the correct answer.
    """

    question = dspy.InputField()
    
    subject = dspy.InputField()

    a = dspy.InputField()
    b = dspy.InputField()
    c = dspy.InputField()
    d = dspy.InputField()

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`.")

# METRICS

def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# TRAINSET

trainset, _, _ = get_data(SUBJECT)

# PROGRAM

# cot = dspy.ChainOfThought(QAset)

class COT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QAset)

    def forward(self, question, subject, a, b, c, d):
        return self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d
        )

# OPTIMIZER

# config = dict(
#     max_bootstrapped_demos=4,
#     max_labeled_demos=4,
#     # num_candidate_programs=10,
#     # num_threads=4
# )

# teleprompter = BootstrapFewShot(
#     metric=validate_answer,
#     **config
# )

# optimized_program = teleprompter.compile(
#     COT(),
#     trainset=trainset
# )

# while True:
#     try:
#         optimized_program.save(SAVE_PATH)
#     except:
#         SAVE_PATH = input('Enter a valid save path: ')

# optimized_program.save(SAVE_PATH)

class DSPYpipeline:
    def __init__(
            self,
            model=DEFAULT_MODEL_STRING,
            save_path=SAVE_PATH,
            max_tokens=512
    ):
        self.model = model
        self.save_path = save_path
        self.lm = dspy.OpenAI(
            model=model,
            max_tokens=max_tokens
        )
        dspy.configure(lm=self.lm)

    def optimize(self, subject, optimizer=None):
        assert optimizer is not None

        devset, valset, _ = get_data(subject)
        optimized_model = dispatch_optmizer(optimizer)(
            model=COT(),
            trainset=devset,
            metric=validate_answer,
            valset=valset
        )
        # save optimized model
        optimized_model.save(self.save_path)

    def load(self, path):
        p = COT()
        p.load(path=path)
        return p

    def test(self, subject):
        _, _, testset = get_data(subject)
        model = self.load(self.save_path)
        responses = {}
        for example in tqdm.tqdm(testset):
            answer = model.forward(
                question=example.question,
                subject=example.subject,
                a=example.a,
                b=example.b,
                c=example.c,
                d=example.d
            )
            responses[example.question] = {
                "answer": answer['answer'],
                "rationale": answer['rationale'],
                "correct": validate_answer(example, answer, trace=None)
            }
        return responses

if __name__ == '__main__':

    optimizer = "BootstrapFewShot"
    subject = "college_mathematics"

    _save_path = "runs/"
    save_path = _save_path+subject+"_"+optimizer+".json"
    pipeline = DSPYpipeline(
        model=DEFAULT_MODEL_STRING,
        save_path=save_path,
        max_tokens=MAX_TOKENS
    )
    # pipeline.optimize(subject, optimizer=optimizer)
    # test
    responses = pipeline.test(subject)
    # pring the score
    correct = sum([1 for k, v in responses.items() if v['correct']])
    total = len(responses)
    print(f"Accuracy: {correct/total}")
    # save responses
    with open(f"{_save_path}{subject}_{optimizer}_responses.json", 'w') as f:
        json.dump(responses, f)