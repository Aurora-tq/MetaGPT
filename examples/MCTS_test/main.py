import shutil
import asyncio
import pytest
import logging
from metagpt.logs import logger 
from metagpt.schema import Message
from examples.MCTS_test.MCTS import MCTS
from examples.test_experiments.dataset_load import dataload
from metagpt.utils.recovery_util import save_history
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.const import MESSAGE_ROUTE_TO_ALL, SERDESER_PATH
from metagpt.actions.add_requirement import UserRequirement
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.utils.common import write_json_file,read_json_file,format_trackback_info
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def get_role_json(context):
    # 实例化 DataInterpreter 对象
    data_interpreter = DataInterpreter()
    print(data_interpreter.planner.plan.goal)
    await data_interpreter.run(with_message=Message(content=context, cause_by=UserRequirement))
    # save_history(role=data_interpreter)

async def use_role_json(role_path):
    # 从 JSON 文件中读取对象状态，并创建新的 DataInterpreter 对象
    role_dict = read_json_file(role_path)
    if role_dict.get('tool_recommender') is None:
        role_dict['tool_recommender'] = ToolRecommender() 
    # if isinstance(role_dict.get('tool_recommender', {}).get('tools'), dict):
    #     role_dict['tool_recommender']['tools'] = list(role_dict['tool_recommender']['tools'].keys())
    new_data_interpreter = DataInterpreter(**role_dict)
    await new_data_interpreter.run(with_message=Message(content='continue', cause_by=UserRequirement))

async def main():
    root_path = "/Users/aurora/Desktop/MCTS_test"
    dataset_name = "House Price"
    data_path, user_requirement = dataload(dataset_name)
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    task_type = 'EDA'
    user_requirement = (
        f"Make sure to generate only one task each time. At this stage,your task is {task_type}. Don't need to plot. "
        )
    mcts = MCTS(max_depth=4)
    # role = DataInterpreter(use_reflection=True)
    #task = "This is a house price dataset, your goal is to predict the sale price of a property based on its features." +f"Train data path: {train_path}', eval data path: '{eval_path}'." 
    query = "Analyze the 'load_wine' dataset from sklearn to predict wine quality. Visualize relationships between features, use machine learning for classification, and report model accuracy. Include analysis and prediction visualizations. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Don't need to plot!"
    # initial_state = await role.run(query)
    # logger.info(initial_state)
    await get_role_json(query)
    role_path = "/Users/aurora/Desktop/metaGPT_new/MetaGPT/workspace/storage/team/environment/roles/DataInterpreter_David/role.json"
    await use_role_json(role_path)
    # # 执行MCTS搜索
    # best_node = await mcts.search(initial_state,task)

    # # 输出最优路径的状态和行动
    # path = []
    # node = best_node
    # while node.parent is not None:
    #     path.append((node.state, node.action))
    #     node = node.parent

    # path.reverse()
    # for state, action in path:
    #     print(f"Action: {action}\nState:\n{state}\n")

asyncio.run(main())
