#调用DI生成code--待改
import os
import re
import json

from metagpt.llm import LLM
from metagpt.schema import Message
from metagpt.logs import logger
from metagpt.roles.di.data_interpreter import DataInterpreter
from examples.test_experiments import save_plan
STRUCTURAL_PROMPT = """
1.If the generated code matches the insights and the score is reasonable, output the final score. 
2.If the generated code does not match the insights or the score is not reasonable, output the text: 
3. "The generated candidate code score is not reasonable according to the self-checker."
"""

REFLECTION_SYSTEM_MSG = """ 
"""
class CodeGenerator:
    @staticmethod
    def load_baseline_code(parent_node):
        code_str = parent_node.state  # 获取父节点的state，假设state包含代码
        return code_str 
    @staticmethod
    def load_json_data(json_dir):
        with open(json_dir, "r") as file:
            json_data = json.load(file)
            return json_data
        
    def extract_code(
        self,
        plan_path: str
        ):
        code_pairs = []
        json_data = self.load_json_data(plan_path)
        code = [json_data["task_map"][str(j)]["code"] for j in range(1, len(json_data["task_map"]) + 1)]
        code_pairs.append(code)
        return code_pairs
    
    def extract_highest_score(
        self,
        file_path:str,
        low_is_better:bool
        ):
        """
        Extract the highest scores from a list of JSON file paths.

        Parameters:
        file_paths (list of str): List of paths to JSON files.

        Returns:
        list of float: List of highest scores from each file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            # Extract the result field from the JSON data
            result_text = json_data['task_map'][str(len(json_data["task_map"]))]["result"]
            number_pattern = re.compile(r'\d+\.\d+')
            # 提取所有数字
            numbers = number_pattern.findall(result_text)
            if low_is_better:
                # 查找当前文件中的最低分数
                current_best = float('inf')
                for number in numbers:
                    accuracy_value = float(number)
                    # 确保数值在0到1之间
                    if 0 <= accuracy_value <= 1:
                        if accuracy_value < current_best:
                            current_best = accuracy_value
            else:
                # 查找当前文件中的最高分数
                current_best = 0.0
                for number in numbers:
                    accuracy_value = float(number)
                    # 确保数值在0到1之间
                    if 0 <= accuracy_value <= 1:
                        if accuracy_value > current_best:
                            current_best = accuracy_value
                            
            # 将最佳分数存储到 scores 列表中，保留4位小数
            if current_best != float('inf') and current_best != 0.0:
                score = round(current_best, 4) 
        return score
    #这里还要改，增加checkLLM的prompt

    async def checkLLM(self,history, score):
        llm = LLM()
        structural_prompt = STRUCTURAL_PROMPT.format(
            history = history,  
            score = score
        )
        context = llm.format_msg([Message(content=structural_prompt, role="user")]) 
        check_rsp = await llm.aask(
            context, system_msgs=[REFLECTION_SYSTEM_MSG]
        )
        logger.info(check_rsp)
        #注意: 检查生成的code是否符合insight的描述，以及生成的代码和分数是否合理，如果合理则输出最终分数，如果不合理则输出“分数不合理，暂无score”；
        # rsp = InsightGenerator.clean_json_from_rsp(llm_response)
        # format_rsp = InsightGenerator.format_output(rsp)
        feedback = check_rsp
        return feedback
    
    async def generate_code(self,root_path,insight, parent_node,user_requriment,data_path,low_is_better,depth):
        # 根据insight和父节点生成代码和分数
        train_path = f"{data_path}/split_train.csv"
        eval_path = f"{data_path}/split_eval.csv"
        baseline_code = self.load_baseline_code(parent_node)
        role = DataInterpreter(use_reflection=True)
        output_requirement = "Please generate new code based on the user requirement, provided insights, and the best-performing baseline code from the previous round. The new code should include both the new changes and the original unchanged code to ensure completeness. The insights consist of three parts: data preprocessing, feature engineering, and model training. Ensure that only the part mentioned in the insight is modified in the newly generated code, while the rest parts should remain unchanged and generate code exactly the same as the baseline code for their respective tasks."
        query = (
                    f"{user_requriment}\n{insight}\nHere is baseline code:\n{baseline_code}\n{output_requirement} "
                    f"Please make sure you have loaded the data and some basic libraries before starting data preprocessing(In eda task): Train data path: {train_path}, eval data path: {eval_path}."
                )
        rsp = await role.run(query)
        logger.info(rsp)
        #step1:先保存history
        output_path = save_plan.save_history(role=role,save_dir = root_path,round =str(depth))
        #step2:从history中提取plan
        plan_path = output_path/ "plan.json"
        # plan_to_code.rename_plan(save_path,new_plan_path)
        #plan_to_code
        code = self.extract_code(plan_path)
        #code中选择最高分
        score = self.extract_highest_score(plan_path,low_is_better)
        #检查这个分数和代码是否正确；错误的话请改正
        history = rsp #暂时先定义为rsp
        # feedback = await self.checkLLM(history,score) ##先不加checkLLM
        # 调用checkLLM方法来检查生成的代码
        return code, score
    
 
   

