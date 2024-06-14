import dspy
from datahandler import get_examples, get_subjects, get_test_data
from dspy.teleprompt import BootstrapFewShot



# CONFIGURE DEFAULTs

SUBJECT = 'anatomy'
SAVE_PATH = 'testjson.json'
DEFAULT_MODEL_STRING = 'gpt-3.5-turbo-1106'

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

trainset = get_examples(SUBJECT)

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

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    # num_candidate_programs=10,
    # num_threads=4
)

teleprompter = BootstrapFewShot(
    metric=validate_answer,
    **config
)

optimized_program = teleprompter.compile(
    COT(),
    trainset=trainset
)

# while True:
#     try:
#         optimized_program.save(SAVE_PATH)
#     except:
#         SAVE_PATH = input('Enter a valid save path: ')

optimized_program.save(SAVE_PATH)