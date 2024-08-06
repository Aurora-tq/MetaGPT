import asyncio
from examples.MCTS_test.MCTS import MCTS
from examples.test_experiments.dataset_load import dataload
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger 
import nest_asyncio
nest_asyncio.apply()
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
    role = DataInterpreter(use_reflection=True)
    task = "This is a house price dataset, your goal is to predict the sale price of a property based on its features." +f"Train data path: {train_path}', eval data path: '{eval_path}'." 
    query = task + user_requirement
    initial_state = await role.run(query)
    logger.info(initial_state)

    # 执行MCTS搜索
    best_node = await mcts.search(initial_state,task)

    # 输出最优路径的状态和行动
    path = []
    node = best_node
    while node.parent is not None:
        path.append((node.state, node.action))
        node = node.parent

    path.reverse()
    for state, action in path:
        print(f"Action: {action}\nState:\n{state}\n")

asyncio.run(main())
# if __name__ == "__main__":
#     main()

#整理一下思路：
# 每个节点的定义：
#   - state：截止到当前阶段生成的代码片段
#   - action：当前阶段所采取的行动
#   - value：LLM对当前行动进行预测的得分
#   - vistited：记录被访问次数
#   - id：每个节点都有一个独立的id
#step1：调用DI，让DI只生成EDA任务的代码。
#step2：将EDA的代码存为root节点，然后对该root节点进行expand
#Step3：需要定义一个评估LLM，对当前节点的state和action进行打分；
#step3：扩展后随机选择一个或者多个子节点进行模拟和反向传播
#Step4：最终返回最优路径，得到效果最好的节点。



#_expand实现的流程：
#step1：需要限制任务去action space中进行生成，迭代生成5次，可以按照current_depth来进行确认当前应该抽取的是哪个task_type的任务。
#step2：还需要DI去生成截止到当前深度的代码


#需要改insight Generator的流程：
#不同的深度应该抽取不同的task