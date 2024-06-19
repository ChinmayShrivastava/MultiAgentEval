from textgrad.tasks import Dataset

class MMLUdataset(Dataset):
    def __init__(
        self,
        subject: str = "machine_learning",
        split: str = "train",
        *args,
        **kwargs
    ):
        super().__init__()

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    def get_default_task_instruction(self):
        pass