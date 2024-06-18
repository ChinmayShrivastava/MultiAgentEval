import json
import os

from modules.pipelines.dspypipeline import DSPYpipeline
from modules.programs.three_layer_cot import COT

################ Constants #################

DEFAULT_MODEL_STRING = 'gpt-3.5-turbo-1106'
MAX_TOKENS = 256
OPTIMIZER = "BootstrapFewShot"
SUBJECT = "high_school_physics"
PROGRAM_NAME = "three_layer_cot_with_halving"
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
    pipeline.optimize(SUBJECT, optimizer=OPTIMIZER)
    # test
    pipeline.test(SUBJECT)

if __name__ == '__main__':
    main()