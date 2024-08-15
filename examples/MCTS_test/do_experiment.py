import os

import fire

from examples.di.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.tools.tool_recommend import TypeMatchToolRecommender
import asyncio
from examples.MCTS_test.utils import load_data_config
from examples.test_experiments.dataset_load import dataload



data_config = load_data_config()

prompt = """\
user requirement:
{user_requirement}

data dir
training: {train_path}
testing: {test_path}
"""

# Ensure ML-Benchmark dataset has been downloaded before using these example.
async def main(task_name, use_reflection=True):
    data_dir, user_requirement = dataload(task_name, data_config)
    train_path = os.path.join(data_dir, "split_train.csv")
    test_path = os.path.join(data_dir, "split_eval.csv")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ML-Benchmark dataset not found in {data_dir}.")
    user_requirement = prompt.format(user_requirement=user_requirement, train_path=train_path, test_path=test_path)
    di = DataInterpreter(use_reflection=use_reflection, tool_recommender=TypeMatchToolRecommender(tools=["<all>"]))
    await di.run(user_requirement)


if __name__ == "__main__":
    asyncio.run(main("house_prices"))
