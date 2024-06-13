import dspy

# CONFIGURE DEFAULT MODEL

DEFAULT_MODEL_STRING = 'gpt-3.5-turbo-1106'

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