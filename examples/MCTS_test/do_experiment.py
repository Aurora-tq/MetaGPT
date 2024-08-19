import os

import fire

from examples.di.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.roles.di.research_assistant import ResearchAssistant
from metagpt.tools.tool_recommend import TypeMatchToolRecommender
import asyncio
from examples.MCTS_test.utils import load_data_config, generate_task_requirement
from examples.test_experiments.dataset_load import dataload
from metagpt.utils.save_code import DATA_PATH, save_code_file



data_config = load_data_config()


# Ensure ML-Benchmark dataset has been downloaded before using these example.
async def main(task_name, use_reflection=True):
    user_requirement = generate_task_requirement(task_name, data_config)
    di = ResearchAssistant(use_reflection=use_reflection)
    await di.run(user_requirement)

if __name__ == "__main__":
    asyncio.run(main("house_prices"))
