REASON_AND_ANSWER_PROMPT=(
	"For the given multiple choice question with the options, reason step by step to get to the answer.\n"
	"For the output follow the format: \n"
	"Reasoning:\n"
	"$reasoning\n"
	"Answer:\n"
	"$answer\n"
	"Where `$reasoning` is the step by step reasoning to get to the answer and `$answer` is the letter from `ABCD` corresponding to the answer.\n"
	"Question:\n"
	"{question}\n"
	"Options:\n"
	"{options}"
)

EVALUATE_PROMPT_RESPONSE=(
	"As an advance evaluator, you are tasked to evaluate a prompt used to generate an answer to a question and the response to a multiple choice question. \n"
	"You are also tasked with providing feedback which will be used to improve the prompt to perform better. \n"
	"Please use the Question, and the options provided to deeply understand the problem. \n"
	"Then judge the rationale and the final answer in teh response provided. \n"
	"Generate a list of criterion specific to the question, with feedback on each criteria to improve the robustness of the prompt. \n"
	"Please provide a detailed feedback on the prompt and the response. \n"
	"Prompt:\n"
	"{prompt}\n"
	"Generated Response with rationale and the final answer:\n"
	"{response}\n"
	"Detailed Feedback:\n"
)

UPDATE_PROMPT_FROM_EVALUATION=(
	"Based on the feedback generated for the prompt and the response it generated, update the prompt to improve the robustness and performance of the prompt. \n"
	"Please make sure to address and incorporate all the feedback provided. \n"
    "Ensure that the prompt is also configured to output reasoning as well as a letter for an answer, not the answer itself. \n"
	"Initial Prompt:\n"
	"{prompt}\n"
	"Feedback:\n"
	"{feedback}\n"
	"Updated Prompt:\n"
)