import dspy
import mlflow
import pandas as pd
import tqdm
from modules.datahandler import get_data
from modules.metrics import validate_answer
from modules.optimizers import dispatch_optmizer
from modules.programs.two_layer_cot import COT


class DSPYpipeline:
    def __init__(
        self,
        model=None,
        save_path=None,
        program=None,
        max_tokens=None,
        mlflow_tracking_uri=None,
        mlflow_experiment_name=None
    ):
        assert model is not None, "model is required"
        assert save_path is not None, "save_path is required"
        assert program is not None, "program is required"
        assert max_tokens is not None, "max_tokens is required"
        assert mlflow_tracking_uri is not None, "mlflow_tracking_uri is required"
        assert mlflow_experiment_name is not None, "mlflow_experiment_name is required"

        self.model = model
        self.save_path = save_path
        self.program = program
        self.lm = dspy.OpenAI(
            model=model,
            max_tokens=max_tokens
        )
        dspy.configure(lm=self.lm)

        # set up MLFlow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

    def optimize(self, subject, optimizer=None):
        assert optimizer is not None

        devset, valset, _ = get_data(subject)
        optimized_model = dispatch_optmizer(optimizer)(
            model=self.program(),
            trainset=devset,
            metric=validate_answer,
            valset=valset
        )
        # save optimized model
        optimized_model.save(self.save_path)

    def load(self, path):
        p = self.program()
        p.load(path=path)
        return p

    def test(self, subject):
        _, _, testset = get_data(subject)
        model = self.load(self.save_path)
        responses = {}
        correct_count = 0
        total_count = 0
        progress_bar = tqdm.tqdm(enumerate(testset), total=len(testset), desc="Testing", leave=False)

        output_df = pd.DataFrame(columns=["qid", "rationale", "correct_answer", "answer"])
        for i, example in progress_bar:
            answer = model.forward(
                question=example.question,
                subject=example.subject,
                a=example.a,
                b=example.b,
                c=example.c,
                d=example.d
            )
            
            output_df = output_df.append({
                "question": example.question,
                "rationale": answer['rationale'],
                "correct_answer": example.correct, 
                "answer": answer['answer']
            }, ignore_index=True)

            correct = validate_answer(example, answer, trace=None)
            if correct:
                correct_count += 1
            total_count += 1
            progress_bar.set_postfix({
                "accuracy": correct_count / total_count
            })
            responses[example.question] = {
                "answer": answer['answer'],
                "rationale": answer['rationale'],
                "correct": validate_answer(example, answer, trace=None)
            }

        with mlflow.start_run():
            # Log metrics and parameters to mlflow
            mlflow.log_metric("accuracy", correct_count / total_count)
            mlflow.log_param("total_count", total_count)

            dict_to_table = {
                "question": [x for x in output_df["question"]],
                "rationale": [x for x in output_df["rationale"]],
                "correct_answer": [x for x in output_df["correct_answer"]],
                "answer": [x for x in output_df["answer"]]
            }
            mlflow.log_table(data=dict_to_table, artifact_file="eval_results.json")

        return responses