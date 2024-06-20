AUTO_REASON_PROMPT = """Generate a step-by-step reasoning process to reach the correct answer for the given multiple choice question. Follow these instructions:

Identify the question and list the options.
Compare each option against the correct answer, explaining why each incorrect option is eliminated.
Address any potential misconceptions or common errors related to the question.
Conclude by reiterating the correct answer after the reasoning process.
State any assumptions made if the question is ambiguous or lacks context.

INPUT:
{input}

OUTPUT:
{expected_output}

For the given input and output, generate reasoning steps that explain how the correct answer was reached.

REASONING:"""