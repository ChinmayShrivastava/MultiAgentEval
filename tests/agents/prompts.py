DEFAULT_AGENT_PROMPT = (
	"As an advanced language model, answer the questions given to you.\n"
	"Each question has 4 possible answers, where 3 are incorrect and 1 is correct.\n"
	"Your task is to identify the correct answer and return the corresponding letter; A, B, C, or D.\n"
	"You will get additional information that might help you determine the correct answer.\n"
	"Use this information to your advantage and answer the questions to the best of your ability.\n"
	"For each question, just return the letter of the correct answer. No Preambles or Postscripts. Just the letter.\n"
)

DEFAULT_AGENT_SPAWN_PROMPT = (
	"As an advanced language model, you are required to break down the given question into {n} unique perspectives.\n"
	"You will get a question to answer on which we need information.\n"
	"Your task is to break down the question into long tailed queries representing unique perspectives "
	"that we need to further research to provide a correct answer.\n"
	"Output each perspective as a separate line.\n"
	"Try to rank the perspectives based on how important they are for the question.\n"
	"Generate {n} unique perspectives for the given question.\n"
	
	"Question: {question}\n"
	"Output: \n"
)

DEFAULT_AGENT_GENERATED_REASON = (
	"As an advanced language model, your job is to provide best reasing for "
	"the given question from the required perspective.\n"
	"The user needs information from different angles to make a final decision on the question.\n"
	"The original question is a multiple choice question so don't provide the answer.\n"
	"Provide instead, the characteristics of the correct answer and what is the reasoning behind it.\n"
	"Generate your reasoning in the following format:\n"
	"------------------------\n"
	"For the given question, when looking at it from the perspective of <perspective>, "
	"the correct answer should have the following characteristics: <characteristics> "
	"because <reasoning>.\n"
	"------------------------\n"
	"Replace <perspective> with the perspective you are generating reasoning for.\n"
	"Replace <characteristics> with the characteristics using your own words.\n"
	"Replace <reasoning> with the reasoning behind the characteristics.\n"
	"Generate reasoning for the given question from the required perspective.\n"
	"Question: {question}\n"
	"Perspective: {perspective}\n"
	"Output: \n"
)

DEFAULT_ANSWER_PROMPT = (
	"You are instructed to answer the multiple choice question given to you.\n"
	"Each question has 4 possible answers, where 3 are incorrect and 1 is correct.\n"
	"Your task is to identify the correct answer and return the corresponding letter; A, B, C, or D.\n"
	"Use the relevant information attached to the question to determine the correct answer.\n"
	"The information is collected from different perspectives to help you make an informed decision.\n"
	"Answer the question to the best of your ability.\n"
	"Return the response in the following format:\n"
	"------------------------\n"
	"Reasoning: <reasoning>\n"
	"Answer: <answer>\n"
	"------------------------\n"
	"Replace <reasoning> with the reasoning, why you think the answer is correct.\n"
	"Replace <answer> with the letter of the correct answer.\n"
	"Answer the question with the correct letter.\n"
	"Question: {question}\n"
	"Options: {options}\n"
	"Information: {information}\n"
	"Output: \n"
)

# if __name__ == "__main__":
	# print(DEFAULT_AGENT_PROMPT)
	# print(DEFAULT_AGENT_SPAWN_PROMPT)
	# print(DEFAULT_AGENT_GENERATED_REASON)