import yaml


def load_data_config(file_path="data.yaml"):
    with open(file_path, 'r') as stream:
        data_config = yaml.safe_load(stream)
    return data_config