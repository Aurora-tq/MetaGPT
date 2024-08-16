#调用LLM生成进化点--待改
STRUCTUAL_PROMPT = """
[Original experience]
Below are randomly sampled experience from a previous dataset, each labeled with an experience, a corresponding score, and a unique ID:
{experience}
[Baseline code]
{code}
**Task**:
Please according to the different {task_type} to generate the corresponding insights:
- Reasoning: Analyze the provided experience based on the variation in their scores to identify which experience are most likely to improve baseline code's performance. 
- Reference: Connect each key point with the corresponding experience ID.
- Insights: Based on the given reasons, referenced experience, provide an as specific and actionable as possible insight you believe can enhance the baseline code's performance. It is best if this insight is based on the provided past experience and are different from the baseline code. ** Your insights must be listed in bullet points, with a minimum of 2 points(e.g.,1.).**

**Expected Output Hard Format**:
```json
{{
    "{task_type}": {{
        "Source": ["List all experience IDs related to {task_type}."],
        "Reference IDs": ["List of IDs that you mainly reference or choose from this stage source. "],
        "Reasoning(yes)": "Reasons for selecting these experience, including comparisons and key takeaways.",
        "Not Reference IDs": ["List of IDs that you did not reference or choose from this stage source."],
        "Reasoning(No)": "Reasons for not selecting these experience, including comparisons and limitations.",
        "Insights": "Based on the reasons and experience, propose an as specific and actionable as possible insight for improving baseline code's performance. If you need to create new features, please provide specific feature names. "
    }}
}}

"""
REFLECTION_SYSTEM_MSG = "As a Kaggle grandmaster participating in a competition, you need to analyze your experience and propose evolutionary points that are more likely to improve the performance of baseline code."

import re
import random
import json
from metagpt.llm import LLM
from metagpt.schema import Message
from metagpt.logs import logger
from examples.MCTS_test.utils import load_data_config
DATA_CONFIG = load_data_config()

class InsightGenerator:
    data_config = DATA_CONFIG

    @staticmethod
    def load_json_data(json_dir):      
        with open(json_dir, "r") as file:
            json_data = json.load(file)
            return json_data

    @staticmethod
    def _random_sample(analysis, num_samples):
        return random.sample(analysis, num_samples)

    @staticmethod
    def clean_json_from_rsp(text):
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            json_str = "\n".join(matches)
            return json_str
        else:
            return ""  
    
    @staticmethod
    def format_output(rsp):
        rsp_list = []
        new_data = [] 
        rsp_list.append(rsp)              
        for item in rsp_list:
            item_dict = json.loads(item)
            data = {
                "Insights": item_dict,
            }
            new_data.append(data)
        return new_data

    @staticmethod
    def load_analysis_pool(file_path):
        data = InsightGenerator.load_json_data(file_path)
        return data

    # @staticmethod
    # #这里需要改正，传入的参数为父节点而不是文件path
    # def load_baseline_code(file_path):
    #     code_str = ""
    #     with open(file_path, "r") as file:
    #         for line in file:
    #             code_str += line
    #     return code_str 

    @staticmethod
    def load_baseline_code(parent_node):
        code_str = parent_node.state['code']  # 获取父节点的state，假设state包含代码
        return code_str 
    
    @staticmethod
    def load_insight(data, task_type):
        new_data = []
        data = data[0]["Insights"]
        _data ={
            f'{task_type}':{
                data[f'{task_type}']['Insights']
            }
        }    
        new_data.append(_data)
        return new_data
    
    async def summarize_insights(self,experience, code,task_type):
        llm = LLM()
        
        structual_prompt = STRUCTUAL_PROMPT.format(
            experience=experience, 
            code=code,  
            task_type=task_type
        )
        context = llm.format_msg([Message(content=structual_prompt, role="user")]) 
        llm_response = await llm.aask(
            context, system_msgs=[REFLECTION_SYSTEM_MSG]
        )
        logger.info(llm_response)
        rsp = InsightGenerator.clean_json_from_rsp(llm_response)
        format_rsp = InsightGenerator.format_output(rsp)
        return format_rsp
        
        
    
    #这里还需要改正
    async def generate_insights(self, baseline_code, task_type):
        # Step 1: 得到父节点的code
        # code = self.load_baseline_code(parent_node)
        #score = parent_node.score
        # Step 2: 从经验池中抽取经验
        experience_pool_path = self.data_config["analysis_pool_dir"] #目前是直接传入一个经验池的path，后续可能需要更改
        experience_pool = self.load_analysis_pool(experience_pool_path)
        selected_experiences = self._random_sample(experience_pool, max(25,int(len(experience_pool)/2))) #随机采样
        insight = await self.summarize_insights(selected_experiences, baseline_code, task_type)  # Add the score if applicable
        return self.load_insight(insight, task_type)
        # 注意: 返回的为一个list，然后这个list的形式为
        # [{"Data_process insight:","...."},{"Feature Engineering insight:","...."},{"Model Training insight:","...."}]
