from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, model_validator

from metagpt.actions.di.ask_review import ReviewConst
from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from metagpt.actions.di.write_analysis_code import CheckData, WriteAnalysisCode
from metagpt.logs import logger
from metagpt.prompts.di.write_analysis_code import DATA_INFO
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.schema import Message, Task, TaskResult
from metagpt.strategy.task_type import TaskType
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.utils.common import CodeParser
from metagpt.utils.common import write_json_file,read_json_file,format_trackback_info
from metagpt.const import MESSAGE_ROUTE_TO_ALL, SERDESER_PATH
from metagpt.utils.recovery_util import save_history
from metagpt.actions import Action, ActionOutput


class ResearchAssistant(DataInterpreter):
    node_id: str = "0"
    start_task_id: int = 1
    state_saved : bool = False

    def get_node_name(self):
        return f"Node-{self.node_id}"

    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """Useful in 'plan_and_act' mode. Wrap the output in a TaskResult for review and confirmation."""
        print("The current_task is:", current_task)

        # 执行任务的代码
        code, result, is_success = await self._write_and_exec_code()
        task_result = TaskResult(code=code, result=result, is_success=is_success)
        # 只在任务类型为 'feature engineering' 时保存状态
        if int(current_task.task_id) == self.start_task_id + 1:
            # fe_id = current_task.dependent_task_ids
            self.save_state()
        return task_result
    
    async def _write_code(
        self,
        counter: int,
        plan_status: str = "",
        tool_info: str = "",
    ):  
          # todo is WriteAnalysisCode
        # if self.rc.todo is None:
        #     self.rc.todo = WriteAnalysisCode()
        todo = self.rc.todo
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection  # only use reflection after the first trial

        user_requirement = self.get_memories()[0].content

        code = await todo.run(
            user_requirement=user_requirement,
            plan_status=plan_status,
            tool_info=tool_info,
            working_memory=self.working_memory.get(),
            use_reflection=use_reflection,
        )

        return code, todo

    def save_state(self):
        if self.state_saved:
            return
        self.state_saved = True
        print(f"Saving state at task {self.start_task_id}")
        stg_path = SERDESER_PATH.joinpath("team", "environment", "roles", f"{self.__class__.__name__}_{self.name}")
        name = self.get_node_name()
        role_path = stg_path.joinpath(f"{name}.json")
        # 将状态保存为 JSON 文件
        write_json_file(role_path, self.model_dump())
        save_history(role=self, save_dir=stg_path, name=name)
        

    def remap_tasks(self):
        self.planner.plan.tasks = [self.planner.plan.task_map[task_id] for task_id in sorted(self.planner.plan.task_map.keys())]


    async def run(self, with_message=None) -> Message | None:
        """Observe, and think and act based on the results of the observation"""
        if with_message == "continue":
            # self.set_todo(None)
            # working_memory = self.working_memory
            self.remap_tasks()
            logger.info("Continue to run")
            self.rc.working_memory.clear()
            self.working_memory.clear()
            # self.rc.todo = WriteAnalysisCode()
            rsp = await self.react()            
            # 发送响应消息给 Environment 对象，以便它将消息传递给订阅者
            self.set_todo(None)
            self.publish_message(rsp)
            return rsp
        return await super().run(with_message)

    

    