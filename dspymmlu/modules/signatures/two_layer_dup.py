import dspy

class DUPhint(dspy.Signature):
    """Extract and list the essential information needed to solve the given question using the following structure. Ensure the information is concise and to the point. Handle missing or incomplete data by noting its absence only if it impacts the solution. Simplify formulas and steps to focus on the key elements.

Template:

Task Statement:

[Explicit statement of the task or question to be solved]
Key Details:

[Bullet points of essential information]
[Include relevant context]
Relevant Formulas:

[Simplified formulas needed for the solution]
Steps to Solve:

[Concise steps highlighting the essential information]
Missing Data:

[Note any missing or incomplete data and its impact on the solution, if applicable]
Example:

Task Statement:

Calculate the area of a triangle given its base and height.
Key Details:

Base (b): 5 cm
Height (h): 10 cm
Relevant Formulas:

Area = 0.5 * base * height
Steps to Solve:

Substitute the values into the formula: Area = 0.5 * 5 cm * 10 cm
Calculate the result: Area = 25 cm²
Missing Data:

None"""

    question = dspy.InputField(desc="The multiple choice question.")

    hints = dspy.OutputField(desc="The essential information needed to solve the given question using the specified structure.")

class QADUPset(dspy.Signature):
    """Please understand the question information and the hint and then find the correct answer to the question."""

    question = dspy.InputField(desc="The multiple choice question.")
    
    subject = dspy.InputField(desc="The subject of the question.")

    a = dspy.InputField(desc="Option `a` of the question.")
    b = dspy.InputField(desc="Option `b` of the question.")
    c = dspy.InputField(desc="Option `c` of the question.")
    d = dspy.InputField(desc="Option `d` of the question.")

    hints = dspy.InputField(desc="The essential information needed to solve the given question using the specified structure.")

    answer = dspy.OutputField(desc="The alphabetical letter of the correct answer; `a`, `b`, `c` or `d`.")