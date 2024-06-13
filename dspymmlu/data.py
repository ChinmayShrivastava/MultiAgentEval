import os
import pandas as pd

# ENSURE THAT THE DATA DIRECTORY EXISTS

if not os.path.exists('data'):
    assert False, 'Data directory not found!'

# TYPES

def get_subjects():
    return [f.split('.')[0] for f in os.listdir('data/test') if f.endswith('.csv')]

def get_examples(subject) -> pd.DataFrame:
    return pd.read_csv(f'data/val/{subject}.csv', names=['question', 'a', 'b', 'c', 'd', 'answer'])

def get_test_data(subject) -> pd.DataFrame:
    return pd.read_csv(f'data/test/{subject}.csv', names=['question', 'a', 'b', 'c', 'd', 'answer'])