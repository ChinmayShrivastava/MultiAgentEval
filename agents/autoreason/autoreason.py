from autoreason.prompt import AUTO_REASON_PROMPT

class ReasoningGenerator:
    def __init__(
        self,
        llm
    ):
        self.llm = llm

    async def generate_reasoning(
        self,
        input,
        expected_output
    ):
        prompt = AUTO_REASON_PROMPT.format(
            input=input,
            expected_output=expected_output
        )
        reasoning = await self.llm.acomplete(prompt)
        return reasoning.text