import random
import math
import numpy as np
import os
from collections import defaultdict
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.roles.di.research_assistant import ResearchAssistant
from metagpt.logs import logger
from InsightGenerate import InsightGenerator
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.utils.common import write_json_file, read_json_file, format_trackback_info
from examples.MCTS_test.utils import load_data_config, change_plan, load_notebook, execute_notebook, load_execute_notebook
from metagpt.actions.di.execute_nb_code import ExecuteNbCode
import asyncio



def initialize_di_root_node(requirement, data_config):
    start_task_id = 1
    role = ResearchAssistant(node_id="0", start_task_id=start_task_id)
    state = dict(
        node_dir=os.path.join(data_config["work_dir"], data_config["role_dir"]),
        notebook_path=os.path.join(data_config["work_dir"], data_config["role_dir"]),
        requirement=requirement,
        has_run=False,
        start_task_id=start_task_id,
    )
    return role, Node(parent=None, state=state, action=None, value=0) 



class Node:
    
    def __init__(self, parent=None, state = None, action=None, value = 0, **kwargs):
        self.state = state
        self.action = action
        self.value = value
        self.visited = 0
        self.parent = parent
        self.children = []
        self.depth = self.generate_depth()
        self.id = self.generate_id()

    def get_depth(self):
        return self.depth

    def generate_depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.depth + 1

    def generate_id(self):
        if self.parent is None:
            return "0"
        else:
            num_sibling = len(self.parent.children)
            return f"{self.parent.id}-{num_sibling}"

    def is_terminal(self, depth):
        return depth == 4
    
    def is_fully_expanded(self):
        return len(self.children) == 4
    
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visited) + c_param * math.sqrt((2 * math.log(self.visited) / child.visited))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, reward):
        self.value += reward
        self.visited += 1

    def get_role_path(self):
        fname = f"Node-{self.id}.json"
        role_path = os.path.join(self.state["node_dir"], fname)
        return role_path
    
    def load_node(self):
        role_dict = read_json_file(self.get_role_path())
        if role_dict.get('tool_recommender') is None:
            role_dict['tool_recommender'] = ToolRecommender() 
        elif isinstance(role_dict.get('tool_recommender', {}).get('tools'), dict):
            role_dict['tool_recommender']['tools'] = list(role_dict['tool_recommender']['tools'].keys())
        return ResearchAssistant(**role_dict)
    
    def expand(self):
        insight_geneartor = InsightGenerator()
        insights = insight_geneartor.generate_new_plan()
        for insight in insights:
            role = self.load_node()
            change_plan(role, insight)
            node = Node(parent=self, state=None, action=insight, value=0)
            self.add_child(node)
        
    
    async def run_node(self):
        role = self.load_node()
        execute_nb_code = await load_execute_notebook(role)
        # role.execute_code = execute_nb_code
        
        await role.run(with_message='continue')

class MCTS:
    #data_path
    def __init__(self,max_depth):
        self.children = {}
        self.max_depth = max_depth

    def select(self, node):
        while node.is_fully_expanded():
            node = node.best_child()
        return node

    async def expand(self, node : Node):
        node.expand()
        self.children[node] = node.children
        return node.children
    
    async def simulate(self, node, current_depth):
        simulation_depth = self.max_depth - current_depth
        "Returns the reward for a random simulation (to completion) of `node`"
        #如果当前节点是终止节点，那么应该DI会有一套完整的流程，所以会得到一个强score
        if node.is_terminal():
            return self.evaluate_node(node)
        
        # Randomly select a child node if it exists
        if node in self.children and len(self.children[node]) > 0:
            node = np.random.choice(self.children[node])
        
        # Perform simulation up to simulation_depth
        if simulation_depth == 1:
            reward = self.evaluate_node(node)
            return reward
        else:
            # Get a candidate node and simulate recursively
            # 应该是从action space中随机抽取动作，所以更新动作后再重新生成状态
            candidate = await self.expand(node, current_depth) #这里存疑，就是获取候选集，是怎么样获取的？
            return await self.simulate(candidate, simulation_depth=simulation_depth-1)


    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_path(self, root):
        best_child = root.best_child(c_param=0)  # c_param=0 means we choose based on value only
        return best_child
    
    async def search(self, requirement, data_config):
        root = initialize_di_root_node(requirement, data_config)
        value = self.evaluate_node(root)
        # root = Node(parent=None, state = initial_state, action=None, value=value)
        print("根节点的状态为:",root.state)
        print("根节点的得分为:",root.value)
        children = await self.expand(root)
        #目前是随机选择1个，后续可以改成多个
        first_leaf = random.choice(children)
        reward = await self.simulate(first_leaf)
        self.backpropagate(first_leaf, reward)
        # 后续迭代：使用UCT/UCB进行选择，expand并模拟和反向传播
        for _ in range(1):  # 999次迭代，加上第一次，共计1000次
            leaf = self.select(children)
            if leaf.visited > 0:
                children = await self.expand(leaf)
                leaf = random.choice(children)
            reward = await self.simulate(leaf)
            self.backpropagate(leaf, reward)
            
        return self.best_path(root)
    
    def get_task_type(self, depth):
        if depth == 0:
            return "EDA"
        elif depth == 1:
            return "data processing"
        elif depth == 2:
            return "feature engineering"
        elif depth == 3:
            return "model training"
        elif depth == 4:
            return "model eval"
        else:
            return "other_task"
    
    #这里可以使用LLM来评估得分
    def evaluate_node(self, node):
        # 评估节点状态和行动的得分
        return random.random()  # 这里可以使用LLM来评估得分
