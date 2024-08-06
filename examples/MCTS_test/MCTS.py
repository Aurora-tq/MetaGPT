import random
import math
import numpy as np
from collections import defaultdict
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger
from InsightGenerate import InsightGenerator

class Node:
    def __init__(self, parent=None, state = None, action=None,value = 0 ,depth = 0):
        self.state = state
        self.action = action
        self.value = value
        self.depth = depth
        self.visited = 0
        self.id = id(self)
        self.parent = parent
        self.children = []

    def is_terminal(self,depth):
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

class MCTS:
    #data_path
    def __init__(self,max_depth):
        self.children = {}
        self.max_depth = max_depth

    def select(self, node):
        while node.is_fully_expanded():
            node = node.best_child()
        return node

    async def expand(self, node, current_depth,task):
        task_type = self.get_task_type(current_depth)
        print("现在的task_type是:",task_type)
        for _ in range(3):
            new_state, action = await self.generate_code(node.state,task_type,task)
            value = self.evaluate_node(node)
            print("该节点的评分为:",value)
            child_node = Node(parent=node,state=new_state, action=action, value = value ,depth=current_depth)
            node.add_child(child_node)
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
            candidate = await self.expand(node,current_depth) #这里存疑，就是获取候选集，是怎么样获取的？
            return await self.simulate(candidate, simulation_depth=simulation_depth-1)


    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_path(self, root):
        best_child = root.best_child(c_param=0)  # c_param=0 means we choose based on value only
        return best_child

    # def search(self, initial_state):
    #     root = Node(initial_state,action=None,value = 0,depth=0)
    #     self.expand(root, 1)
    #     for _ in range(1000):  # 设置最大迭代次数
    #         leaf = self.select(root)
    #         if leaf.visited > 0:
    #             children = self.expand(leaf, self.get_current_depth(leaf))
    #             leaf = random.choice(children)
    #         reward = self.simulate(leaf)
    #         self.backpropagate(leaf, reward)
    #     return self.best_path(root)
    
    async def search(self, initial_state,task):
        root = Node(parent=None,state = initial_state, action=None, value=0, depth=0)
        value = self.evaluate_node(root)
        root = Node(parent=None,state = initial_state, action=None, value=value, depth=0)
        print("根节点的状态为:",root.state)
        print("根节点的得分为:",root.value)
        await self.expand(root, 1,task)
        # 首次迭代：随机选择根节点的一个子节点进行模拟和反向传播
        first_children = self.children[root]
        #目前是随机选择1个，后续可以改成多个
        first_leaf = random.choice(first_children)
        reward = await self.simulate(first_leaf)
        self.backpropagate(first_leaf, reward)
        # 后续迭代：使用UCT/UCB进行选择，expand并模拟和反向传播
        for _ in range(999):  # 999次迭代，加上第一次，共计1000次
            leaf = self.select(first_children)
            if leaf.visited > 0:
                children = await self.expand(leaf, self.get_current_depth(leaf),task)
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

    def get_current_depth(self, node):
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth
    
    #这里需要改
    async def generate_code(self, state,task_type,task):
        # 调用DI生成代码，返回新状态和行动
        # print("type of state:"*10,type(state.content))
        # print("state的content:",state.content)
        user_requirement = (
        f"At the current stage, your task is {task_type}, please combine the tasks and code snippets of the parent node to generate a new plan. The requirements include the tasks involved in the parent node, and the code remains unchanged as much as possible."
        )
        #根据task_type去action space中随机采样得到的insight
        insight = InsightGenerator()
        action = await insight.generate_insights(state,task_type) #InsightGenerator得到的结果为包含user_requriement的insight
        print("当前的行动为:",action)
        query = task + "\n"+"Here are some insights:"+str(action) + "\n"+"Here is parent node's state:"+ str(state)+ "\n"+user_requirement 
        print("当前的用户询问为:",query)
        role = DataInterpreter(use_reflection=True)
        rsp = await role.run(query)
        logger.info(rsp)
        new_state = rsp
        return new_state, action
    
    #这里可以使用LLM来评估得分
    def evaluate_node(self,node):
        # 评估节点状态和行动的得分
        return random.random()  # 这里可以使用LLM来评估得分
