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

_EVALUATE_PROMPT_RESPONSE=(
	"As an advance evaluator, you are tasked to evaluate a prompt used to generate an answer to a multiple choice question. \n"
	"You are also tasked with providing critical feedback which will be used to improve the prompt to perform better. \n"
	"Please use the Question, and the options provided to deeply understand the problem. \n"
	"Then judge the rationale and the final answer in the response provided. \n"
	"Generate a list of criterion specific to the question, with feedback on each criteria to improve the robustness of the prompt. \n"
	"Please provide a detailed feedback on the prompt and the response. \n"
	"Prompt:\n"
	"{prompt}\n"
	"Generated Response with rationale and the final answer:\n"
	"{response}\n"
	"Detailed Feedback:\n"
)

EVALUATE_PROMPT_RESPONSE=(
	"We have an opportunity to provide critical feedback to the prompt designed to generate an answer to a multiple choice question. \n"
	"Generate a structured feedback with criterion specific to the question. Provide at max of 5 criterion. Each line should have a new criteria. \n"
	"The feedback should be constructive while pointing the shortcomings and teh inaccuracies in the prompt through the response generated. \n"
	"Through the feedback, guide the prompt to be be able to generate a correct answer with a rationale. \n"
	"Do not pass the answer in the feedback, rather provide a detailed feedback on how to improve the prompt to generate a better response. \n"
	"Prompt:\n"
	"{prompt}\n"
	"Generated Response with rationale and the final answer:\n"
	"{response}\n"
	"Detailed Feedback:\n"
)

_UPDATE_PROMPT_FROM_EVALUATION=(
	"Based on the feedback generated for the prompt and the response it generated, update the prompt to improve the robustness and performance of the prompt. \n"
	"Please make sure to address and incorporate all the feedback provided. \n"
    "Ensure that the prompt is also configured to output reasoning as well as a letter for an answer, not the answer itself. \n"
	"Initial Prompt:\n"
	"{prompt}\n"
	"Feedback:\n"
	"{feedback}\n"
	"Updated Prompt:\n"
)

UPDATE_PROMPT_FROM_EVALUATION=(
	"Use the detailed feedback to update the old prompt and make it more robust adn better suited to answer the question. \n"
	"Please make sure to address and incorporate all the feedback provided. \n"
    "Ensure that the prompt is also configured to output reasoning as well as a letter for an answer, not the answer itself. \n"
	"Do not include any recommended answer to the question, instead focus on improving the prompt to reason the answer better. \n"
	"Modify the old Prompt withoput changing the structure of the prompt. \n"
	"You are allowed to rephrase the question for clarity, add additional instructions on how to approach the question, and provide additional context to the question. \n"
	"You can also provide additional hints to the question to guide the model to generate a better response. \n"
	"**Old Prompt**:\n"
	"{prompt}\n"
	"**Feedback**:\n"
	"{feedback}\n"
	"**Updated Prompt**:\n"
)