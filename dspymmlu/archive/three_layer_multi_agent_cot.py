import json

import dspy
import tqdm
from dspymmlu.modules.datahandler.datahandler import get_data
from dspymmlu.modules.optimizers.dspy_helpers import dispatch_optmizer

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

class Steps(dspy.Signature):
    """Given a question, output the steps to solve it."""

    question = dspy.InputField()

    steps = dspy.OutputField(desc="The steps to solve the question. Each step is separated by a newline.")

class Hint(dspy.Signature):
    """For the given question and one of the steps from its solution, generate the hint for the step specific to the question. The hint should contain how to approach the step."""

    question = dspy.InputField()
    step = dspy.InputField()

    hint = dspy.OutputField(desc="The hint for the step.")

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
    steps = dspy.InputField(desc="The steps and hints to solve the question.")

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

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)
        self.steps = dspy.ChainOfThought(Steps)
        self.hint = dspy.ChainOfThought(Hint)

        self.prog = dspy.ChainOfThought(QAset)

    def get_steps(self, question):
        steps = self.steps(question=question)['steps']
        steps = steps.split('\n')
        return steps
    
    def get_hint(self, question, step):
        return self.hint(question=question, step=step)['hint']

    def forward(self, question, subject, a, b, c, d):
        core_question = self.core_question(question=question)['core_question']
        info = self.info(question=question)['info']
        steps = self.get_steps(question=question)
        hints = [self.get_hint(question=question, step=step) for step in steps]
        # join steps and hints
        steps = [f"STEP: {step}\nHINT FOR THE STEP: {hint}" for step, hint in zip(steps, hints)]
        return self.prog(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=core_question,
            info=info,
            steps='\n'.join(steps)
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
    subject = "high_school_physics"

    _save_path = "runs/"
    save_path = _save_path+subject+"_"+optimizer+".json"
    pipeline = DSPYpipeline(
        model=DEFAULT_MODEL_STRING,
        save_path=save_path,
        max_tokens=MAX_TOKENS
    )
    pipeline.optimize(subject, optimizer=optimizer)
    # # test
    # responses = pipeline.test(subject)
    # # pring the score
    # correct = sum([1 for k, v in responses.items() if v['correct']])
    # total = len(responses)
    # print(f"Accuracy: {correct/total}")
    # # save responses
    # with open(f"{_save_path}{subject}_{optimizer}_responses.json", 'w') as f:
    #     json.dump(responses, f)