import dspy

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
