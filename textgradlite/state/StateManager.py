from textgradlite.utils.formatters import iterative_formatter
from textgradlite.state.examplestate import eg_state

import datetime
import json

class StateManager:
    def __init__(
        self,
        state = None
    ):
        self.state = state if state else eg_state

        self._current_iteration = 0

    def _tostring(self):
        return iterative_formatter(self.state)
    
    def __str__(self):
        return self._tostring()
    
    def add_core_question(self, question):
        self.state["corequestion"] = question

    def add_hint(self, hint):
        self.state["hints"].append(hint)

    def add_hints(self, hints):
        self.state["hints"] += hints

    def core_question(self):
        return self.state["corequestion"]
    
    def hints(self):
        return "\n".join([f"{i+1}. {hint}" for i, hint in enumerate(self.state["hints"])])
    
    @property
    def current_iteration(self):
        if self._current_iteration in self.state["iterations"]:
            return self.state["iterations"][self._current_iteration]
        else:
            self.add_iteration(None, None)
            return self.state["iterations"][self._current_iteration]
    
    def add_iteration(self):
        self.state["iterations"][self._current_iteration] = {
            "initial_prompt": "",
            "response": "",
            "feedback": "",
            "updated_prompt": "",
        }

    def add_initial_prompt(self, prompt):
        self.state["iterations"][self._current_iteration]["initial_prompt"] = prompt

    def add_response(self, response):
        self.state["iterations"][self._current_iteration]["response"] = response

    def add_feedback(self, feedback):
        self.state["iterations"][self._current_iteration]["feedback"] = feedback

    def add_updated_prompt(self, prompt):
        self.state["iterations"][self._current_iteration]["updated_prompt"] = prompt

    def next_iteration(self):
        self._current_iteration += 1
        return self.current_iteration