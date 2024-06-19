import re

import dspy


class Steps(dspy.Signature):
    """
    Given a multiple choice question, the subject, 4 answer choices, and some supplemental information, generate a step by step guide to solving the question.
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

    steps = dspy.OutputField(desc="The step by step guide to solving the question to find the correct answer.")

class AnswerQuestionWithSteps(dspy.Signature):
    """
    Given a multiple choice question, the subject, 4 answer choices, the core question, the list of atomic information available in the question, and the step by step guide to solving the question, generate the correct answer.
    """

    question = dspy.InputField(desc="The multiple choice question.")

    subject = dspy.InputField(desc="The subject of the question.")

    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    core_question = dspy.InputField(desc="The one liner core question.")
    info = dspy.InputField(desc="The list of atomic information available in the question.")

    steps = dspy.InputField(desc="The step by step guide to solving the question to find the correct answer.")

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`.")