def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()