import re
import dspy

class Reminders(dspy.Signature):
    """Given ther question, output three reminders that will help the user remember concepts and better answer the question."""

    question = dspy.InputField()

    reminders = dspy.OutputField(desc="A reminder that will help the user remember concepts and better answer the question.")

class TopTwoOptions(dspy.Signature):
    """
    Given a multiple choice question, the subject, 4 answer choices, and some supplemental information, choose the two most likely answer choices.
    """

    question = dspy.InputField(desc="The multiple choice question.")

    subject = dspy.InputField(desc="The subject of the question.")

    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")
    # reminders = dspy.InputField(desc="The list of three reminders that will help the user remember concepts and better answer the question.")

    toptwooptions = dspy.OutputField(desc="A tuple of the two most likely answer choices. Examples: (x, y) where x and y are one of `a`, `b`, `c` or `d`.")

class AnswerQuestion(dspy.Signature):
    """
    Given a multiple choice question, the subject, and the two most likely answer choices, return the alphabetical letter of the correct answer.
    """

    question = dspy.InputField(desc="The multiple choice question.")

    subject = dspy.InputField(desc="The subject of the question.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")
    TopTwoOptions = dspy.InputField(desc="The previously decided best two answer choices.")

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`.")