import json
import os
import dspy
import tqdm
from datahandler import get_data
from dspy_helpers import dispatch_optmizer
import re

# CONFIGURE DEFAULTs

# Idk why, but I needed this:
os.environ["OPENAI_API_KEY"] = ""

SUBJECT = 'anatomy'
SAVE_PATH = 'testjson.json'
DEFAULT_MODEL_STRING = 'gpt-4o'
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

class Reminders(dspy.Signature):
    """Given ther question, output three reminders that will help the user remember concepts and better answer the question."""

    question = dspy.InputField()

    reminders = dspy.OutputField(desc="A reminder that will help the user remember concepts and better answer the question.")

class QAset(dspy.Signature):
    """
    Given a question, the subject, 2 answer choices, and some supplemental information, return the alphabetical letter of the correct answer choice.
    """

    question = dspy.InputField(desc="The multiple choice question.")
    
    subject = dspy.InputField(desc="The subject of the question.")

    possibility = dspy.InputField(desc="Possible answer of the question, labeled with the letter it should always be referred to `a`, `b`, `c` or `d`.")
    other_possibility = dspy.InputField(desc="Another possible answer of the question, labeled with the letter it should always be referred to `a`, `b`, `c` or `d`.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")
    reminders = dspy.InputField(desc="The list of three reminders that will help the user remember concepts and better answer the question.")

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`")

class NarrowDownOptions(dspy.Signature):
    """
    Given a multiple choice question, the subject, 4 answer choices, and some supplemental information, choose the two best options.
    """

    question = dspy.InputField(desc="The multiple choice question.")
    
    subject = dspy.InputField(desc="The subject of the question.")

    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")
    reminders = dspy.InputField(desc="The list of three reminders that will help the user remember concepts and better answer the question.")

    answer = dspy.OutputField(desc="The best two answer choices with their letters, for example [a, c]")

class CleanUpOptions(dspy.Signature):
    """
    From text, return an array of size 2 whose values are one of `a`, `b`, `c` or `d`. Or make a guess.
    """
    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    prev_answers = dspy.InputField(desc="The previously decided best two answer choices.")

    answer = dspy.OutputField(desc="Nothing but an array of the two best answer choice letters. IT IS CRUCIAL YOU OUTPUT SOLELY AN ARRAY OF LENGTH 2 WITH VALUES BEING ONE OF `a`, `b`, `c` or `d`. Examples: [a, c], [b, d], [a, b]")

# METRICS

def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# TRAINSET

trainset, _, _ = get_data(SUBJECT)

# PROGRAM

def parse_array_from_string(input_string):
    match = re.search(r'\[(.*?)\]', input_string)
    if match:
        array_content = match.group(1)
        array_elements = [element.strip().strip("'\"`") for element in array_content.split(',')]
        return array_elements
    else:
        return None

class COT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.core_question = dspy.ChainOfThought(CoreQuestion)
        self.info = dspy.ChainOfThought(ProblemSolvingInfo)
        self.reminders = dspy.ChainOfThought(Reminders)
        self.narrow_down_options = dspy.ChainOfThought(NarrowDownOptions)
        self.clean_up_options = dspy.ChainOfThought(CleanUpOptions)

        self.prog = dspy.ChainOfThought(QAset)

    def forward(self, question, subject, a, b, c, d):
        core_question = self.core_question(question=question)['core_question']
        info = self.info(question=question)['info']
        reminders = self.reminders(question=question)['reminders']
        answer_map = {'a': a, 'b': b, 'c': c, 'd': d}
        options = self.narrow_down_options(
            question=question,
            subject=subject,
            a=a,
            b=b,
            c=c,
            d=d,
            core_question=core_question,
            info=info,
            reminders=reminders
        )['answer']

        cleaned_options = self.clean_up_options(
            a=a,
            b=b,
            c=c,
            d=d,
            prev_answers=options
        )['answer']

        remaining = parse_array_from_string(cleaned_options)
        option_1 = remaining[0]
        option_2 = remaining[1]

        one_answer = option_1 + ": " + answer_map[option_1]
        two_answer = option_2 + ": " + answer_map[option_2]

        return self.prog(
            question=question,
            subject=subject,
            possibility=one_answer,
            other_possibility=two_answer,
            core_question=core_question,
            info=info,
            reminders=reminders
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
        #p.load(path=path)
        return p

    def test(self, subject):
        _, _, testset = get_data(subject)
        model = self.load(self.save_path)
        responses = {}
        for example in tqdm.tqdm(testset[:20]):
            answer = model.forward(
                question=example.question,
                subject=example.subject,
                a=example.a,
                b=example.b,
                c=example.c,
                d=example.d
            )
            # self.lm.inspect_history(n=5)
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
    # test
    responses = pipeline.test(subject)
    correct = sum([1 for k, v in responses.items() if v['correct']])
    total = len(responses)
    print(f"Accuracy: {correct/total}")
    # save responses
    with open(f"{_save_path}{subject}_{optimizer}_responses.json", 'w') as f:
        json.dump(responses, f)