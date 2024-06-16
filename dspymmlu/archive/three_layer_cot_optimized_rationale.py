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

class CoreQuestion(dspy.Signature):
    """Given a question, output a crisp and consize question free of any unnecessary information."""

    question = dspy.InputField()

    core_question = dspy.OutputField(desc="The one liner core question.")

class ProblemSolvingInfo(dspy.Signature):
    """Extract and list the available information from the question that can be used to solve it"""

    question = dspy.InputField()

    info = dspy.OutputField(desc="The list of atomic information available in the question.")

class QAset(dspy.Signature):
    """
    Given a multiple choice question, the subject, 4 options, and some supplemental information, return the alphabetical letter of the correct answer.
    """

    question = dspy.InputField(desc="The multiple choice question.")
    
    subject = dspy.InputField(desc="The subject of the question.")

    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`.")

# METRICS

def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# TRAINSET

trainset, _, _ = get_data(SUBJECT)

# PROGRAM

# cot = dspy.ChainOfThought(QAset)

# RATIONALE_TYPE_1 = dspy.OutputField(
#     prefix=("Given the question, you will answer a reasoning question from high school physics. You will have to reason by using "
#             "high school physics concepts and then generating the step towards solving the question. You will have to think step by step in order to reach the answer. "
#             "Reasoning: Let's think step by step in order to"),
#     desc="${produce the answer}. We ...",
# )

# RATIONALE_TYPE_2 = dspy.OutputField(
#     prefix="""You are given a high school physics problem. To solve it, use high school physics concepts and break down the solution step by step. Carefully analyze the problem and think through each step logically to reach the correct answer. 

# For the question:    
# 1. Identify the key concepts and quantities, if present, given in the problem.
# 2. Recall relevant physics principles and equations.
# 3. Apply the concepts and/or equations to the given quantities.
# 4. Solve for the unknowns systematically.
# 5. Double-check calculations and concepts.

# Reasoning: Let's think through the problem step by step to""",
#     desc="${produce the answer}. We ...",
# )

# RATIONALE_TYPE_3 = dspy.OutputField(
#     prefix="""You are given a high school physics problem. To solve it, use high school physics concepts and break down the solution step by step. Carefully analyze the problem and think through each step logically to reach the correct answer. 

# For the question:    
# 1. Identify the key concepts and quantities.
# 2. Recall relevant physics principles and equations.
# 3. Apply the concepts and/or equations to the given quantities.
# 4. Solve for the unknowns systematically.

# Reasoning: Let's think through the problem step by step to""",
#     desc="${produce the answer}. We ...",
# )


def majority_vote(p):
    answers = p.completions.answer
    a = max(set(answers), key=answers.count)
    p.answer = a

class MajorityVote(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question, votes -> majorityanswer")

    def forward(self, votes):
        r = self.prog(votes=votes)
        return r

class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)

        # self.mv = MajorityVote()

        self.prog = dspy.ChainOfThought(QAset, rationale_type=RATIONALE_TYPE)

    def forward(self, question, subject, a, b, c, d):
        r = self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=self.core_question(question=question)['core_question'],
            info=self.info(question=question)['info']
        )
        # _votes = r.completions.answer
        # votes = ""
        # for i in range(len(_votes)):
        #     votes += _votes[i] + " "
        # r = self.mv(votes=votes)
        return r

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
        correct_count = 0
        total_count = 0
        progress_bar = tqdm.tqdm(testset)
        for example in progress_bar:
            answer = model.forward(
                question=example.question,
                subject=example.subject,
                a=example.a,
                b=example.b,
                c=example.c,
                d=example.d
            )
            correct = validate_answer(example, answer, trace=None)
            if correct:
                correct_count += 1
            total_count += 1
            progress_bar.set_postfix({
                "accuracy": correct_count / total_count
            })
            responses[example.question] = {
                "answer": answer['answer'],
                "rationale": answer['rationale'],
                "correct": validate_answer(example, answer, trace=None)
            }
        return responses

if __name__ == '__main__':

    optimizer = "BootstrapFewShot"
    subject = "high_school_physics"

    _save_path = "runs/"
    save_path = _save_path+subject+"_"+optimizer+".json"
    pipeline = DSPYpipeline(
        model=DEFAULT_MODEL_STRING,
        save_path=save_path,
        max_tokens=MAX_TOKENS
    )
    # pipeline.optimize(subject, optimizer=optimizer)
    # # test
    responses = pipeline.test(subject)
    # pring the score
    correct = sum([1 for k, v in responses.items() if v['correct']])
    total = len(responses)
    print(f"Accuracy: {correct/total}")
    # save responses
    with open(f"{_save_path}{subject}_{optimizer}_responses.json", 'w') as f:
        json.dump(responses, f)