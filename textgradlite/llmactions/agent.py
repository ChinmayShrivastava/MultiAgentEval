from llama_index.core.agent import ReActAgent
from llama_index.core.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from textgradlite.llmactions.prompts import MONITOR_AGENT_PROMPT, USER_AGENT_PROMPT
from textgradlite.state.StateManager import StateManager
from textgradlite.DEFAULTS import DEFAULT_OPENAI_MODEL, MAX_MONITOR_AGENT_ITERATIONS, MAX_USER_AGENT_ITERATIONS

class MonitorAgent:
    def __init__(
        self,
        llm = None,
        sys_prompt = None,
        state = None,
        max_iterations: int = MAX_MONITOR_AGENT_ITERATIONS,
        verbose: bool = False
    ):
        
        if state is None:
            raise ValueError("`state` cannot be None")
        
        self.llm = llm \
            if llm \
            else OpenAI(model=DEFAULT_OPENAI_MODEL)
        self.sys_prompt = sys_prompt \
            if sys_prompt \
            else MONITOR_AGENT_PROMPT
        self.state_manager = StateManager(state)

        self.verbose = verbose
        self.max_iterations = max_iterations

        # TODO: update agent histories as the conversation progresses
        self._agent_history = []
        self._agent = None
        self._chat_end = False

    def _get_reAct_agent(self):

        if self._agent:
            return self._agent
        
        _content = self.sys_prompt

        _history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=_content
            )
        ]

        _agent = ReActAgent.from_tools( 
            llm=self.llm, 
            verbose=self.verbose, 
            chat_history=_history,
            max_iterations=self.max_iterations
        )

        self._agent_history = _history
        self._agent = _agent
        
        return self._agent

    
    def _query(self, input):

        if self._chat_end:
            return None
        
        _agent = self._get_reAct_agent()

        _response = str(_agent.chat(input))

        # TODO
        # TEST RESPONSE, GENERATE LOSS, UPDATE STATE
        # test loss on all examples, if it doesn't work on a single example, fail the test and ask to update the step
        # if loss is zero, finish the conversation.

        return _response

    def query(self, input):
        return self._query(input)
    
class UserAgent:
    def __init__(
        self,
        llm = None,
        state: dict = None,
        monitor: MonitorAgent = None,
        max_iterations: int = MAX_USER_AGENT_ITERATIONS,
        verbose: bool = False
    ):
        if state is None:
            raise ValueError("`state` cannot be None")
        
        self.llm = llm \
            if llm \
            else OpenAI(model=DEFAULT_OPENAI_MODEL)
        self.state_manager = StateManager(state)
        self.monitor = monitor \
            if monitor \
            else MonitorAgent(state=state)
        
        self.verbose = verbose
        self.max_iterations = max_iterations

        self._agent = self._get_reAct_agent()
        
    def _get_reAct_agent(self):
        
        _content = USER_AGENT_PROMPT.format(info=self.state_manager._tostring())

        _history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=_content
            )
        ]

        _agent = ReActAgent.from_tools(
            llm=self.llm,
            verbose=self.verbose,
            chat_history=_history,
            max_iterations=self.max_iterations
        )

        return _agent
        
    def query(self, input):
        instruction = self.monitor.query(input)
        if instruction:
            _instruction = f"NEXT INSTRUCTION: {instruction}"
            return str(self._agent.chat(_instruction))
        else:
            return None