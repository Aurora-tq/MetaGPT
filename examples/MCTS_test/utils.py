import yaml
from examples.test_experiments.dataset_load import dataload


TASK_PROMPT = """\
# User requirement
{user_requirement}

# Data dir
training: {train_path}
testing: {test_path}
"""

def load_data_config(file_path="data.yaml"):
    with open(file_path, 'r') as stream:
        data_config = yaml.safe_load(stream)
    return data_config


def generate_task_requirement(task_name, data_config):
    data_path, user_requirement = dataload(task_name, data_config)
    train_path = f"{data_path}/split_train.csv"
    test_path = f"{data_path}/split_eval.csv"
    user_requirement = TASK_PROMPT.format(user_requirement=user_requirement, train_path=train_path, test_path=test_path)
    return user_requirement

