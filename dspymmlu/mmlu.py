import json
import os
from concurrent.futures import ThreadPoolExecutor

from modules.datahandler.datahandler import SUBJECTS
from modules.pipelines.dspypipeline import DSPYpipeline
from modules.programs.one_layer_cot import COT

################ Constants #################

DEFAULT_MODEL_STRING = 'gpt-3.5-turbo'
MAX_TOKENS = 256
OPTIMIZER = "BootstrapFewShot"
SUBJECT = "high_school_physics"
PROGRAM_NAME = "final_vainlla_cot"
SAVE_DIR = "runs/"+PROGRAM_NAME
SAVE_PATH = SAVE_DIR+"/"+SUBJECT+"_"+OPTIMIZER+".json"
PROGRAM=COT

################ MLFlow ####################

# set the MLFlow tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = PROGRAM_NAME

############################################

# ensure that the SAVE_DIR exists, if not create it
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

############################################

def main():
    pipeline = DSPYpipeline(
        model=DEFAULT_MODEL_STRING,
        save_path=SAVE_PATH,
        program=PROGRAM,
        max_tokens=MAX_TOKENS,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME
    )
    # # optimize
    pipeline.optimize(SUBJECT, optimizer=OPTIMIZER)
    # test
    pipeline.test(SUBJECT)

def gen_optimizers_for_all_subjects():
    for i, subject in enumerate(SUBJECTS):
        if os.path.exists(SAVE_DIR+"/"+subject+"_"+OPTIMIZER+".json"):
            continue
        print(f"Optimizing for subject: {subject} - {i+1}/{len(SUBJECTS)}")
        pipeline = DSPYpipeline(
            model=DEFAULT_MODEL_STRING,
            save_path=SAVE_DIR+"/"+subject+"_"+OPTIMIZER+".json",
            program=PROGRAM,
            max_tokens=MAX_TOKENS,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME
        )
        # optimize
        pipeline.optimize(subject, optimizer=OPTIMIZER)

def test_all_subjects():
    for i, subject in enumerate(SUBJECTS):
        print(f"Testing for subject: {subject} - {i+1}/{len(SUBJECTS)}")
        pipeline = DSPYpipeline(
            model=DEFAULT_MODEL_STRING,
            save_path=SAVE_DIR+"/"+subject+"_"+OPTIMIZER+".json",
            program=PROGRAM,
            max_tokens=MAX_TOKENS,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME
        )
        # test
        pipeline.test(subject)

# def gen_optimizers_for_subject(subject, i, total):
#     print(f"Optimizing for subject: {subject} - {i+1}/{total}")
#     pipeline = DSPYpipeline(
#         model=DEFAULT_MODEL_STRING,
#         save_path=SAVE_DIR+"/"+subject+"_"+OPTIMIZER+".json",
#         program=PROGRAM,
#         max_tokens=MAX_TOKENS,
#         mlflow_tracking_uri=MLFLOW_TRACKING_URI,
#         mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME
#     )
#     # optimize
#     pipeline.optimize(subject, optimizer=OPTIMIZER)
#     # test
#     pipeline.test(subject)

# def gen_optimizers_for_all_subjects():
#     total = len(SUBJECTS)
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         for i, subject in enumerate(SUBJECTS):
#             executor.submit(gen_optimizers_for_subject, subject, i, total)

if __name__ == '__main__':
    # main()
    gen_optimizers_for_all_subjects()