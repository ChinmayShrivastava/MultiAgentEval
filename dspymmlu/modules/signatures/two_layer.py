import dspy

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