import csv
import os

import dspy

# ENSURE THAT THE DATA DIRECTORY EXISTS

if not os.path.exists('data'):
    assert False, 'Data directory not found!'

# TYPES

def get_subjects(type='val'):
    return [f.split('_')[0] for f in os.listdir(f'data/{type}') if f.endswith('.csv')]

def get_examples(subject) -> list[dspy.Example]:
    examples = []
    with open(f'data/val/{subject}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0]:  # Ensure the question is not empty
                examples.append(dspy.Example(
                    question=str(row[0]),
                    subject=subject,
                    a=str(row[1]),
                    b=str(row[2]),
                    c=str(row[3]),
                    d=str(row[4]),
                    answer=str(row[5])
                ).with_inputs("question", "subject", "a", "b", "c", "d"))
    return examples

def get_test_data(subject) -> list:
    data = []
    with open(f'data/test/{subject}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0]:  # Ensure the question is not empty
                data.append([str(r) for r in row])
    return data

# if __name__ == '__main__':
#     print(len(get_examples('abstract_algebra_val')))