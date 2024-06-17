import csv
import os

import dspy

# ENSURE THAT THE DATA DIRECTORY EXISTS

if not os.path.exists('data'):
    assert False, 'Data directory not found!'

# MMLU SUBJECTS

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# TYPES

def get_subjects() -> list[str]:
    return SUBJECTS

def get_dev_data(subject) -> list[dspy.Example]:
    examples = []
    with open(f'data/dev/{subject}_dev.csv', 'r') as file:
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

def get_val_data(subject) -> list[dspy.Example]:
    examples = []
    with open(f'data/val/{subject}_val.csv', 'r') as file:
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

def get_test_data(subject) -> list[dspy.Example]:
    examples = []
    with open(f'data/test/{subject}_test.csv', 'r') as file:
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

def get_data(subject) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    return get_dev_data(subject), get_val_data(subject), get_test_data(subject)

# if __name__ == '__main__':
#     print(len(get_examples('abstract_algebra_val')))