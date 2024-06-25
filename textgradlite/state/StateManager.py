from textgradlite.utils.formatters import iterative_formatter
from textgradlite.state.examplestate import eg_state

import datetime
import json

class StateManager:
    def __init__(
        self,
        state
    ):
        self.state = state

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
        return self._current_iteration
    
    def add_iteration(self, output, evaluation):
        self.state["iterations"][self._current_iteration] = {
            "output": output,
            "evaluation": evaluation
        }
        self._current_iteration += 1