import mlflow

class DSPYPyFunc(mlflow.pyfunc.PythonModel):

    def __init__(
        self,
        program=None
    ):
        assert program is not None, "program must be provided"

        super.__init__()
        self.program = program

        self._loaded_model = None
    
    def predict(
        self, 
        context, 
        model_input
    ):
        pass

    def load_context(self, context):
        self._loaded_model = self.program.load(path=context.artifacts["model_path"])