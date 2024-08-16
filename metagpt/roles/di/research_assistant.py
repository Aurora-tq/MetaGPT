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




class ResearchAssistant(DataInterpreter):
    node_id: str = "0"
    start_task_id: int = 1



    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """Useful in 'plan_and_act' mode. Wrap the output in a TaskResult for review and confirmation."""
        print("The current_task is:", current_task)

        # 执行任务的代码
        code, result, is_success = await self._write_and_exec_code()
        task_result = TaskResult(code=code, result=result, is_success=is_success)
        # 只在任务类型为 'feature engineering' 时保存状态
        if int(current_task.task_id) == self.start_task_id + 1:
            # fe_id = current_task.dependent_task_ids
            print(f"Saving state at task {current_task.task_id}")
            self.save_state()
        return task_result

    def save_state(self):
        stg_path = SERDESER_PATH.joinpath("team", "environment", "roles", f"{self.__class__.__name__}_{self.name}")
        role_path = stg_path.joinpath(f"Node-{self.node_id}.json")
        # 将状态保存为 JSON 文件
        write_json_file(role_path, self.model_dump())
        save_history(role=self)


    async def run(self, with_message=None) -> Message | None:
        """Observe, and think and act based on the results of the observation"""
        if with_message == "continue":
            rsp = await self.react()
            # 重置下一步行动
            self.set_todo(None)
            # 发送响应消息给 Environment 对象，以便它将消息传递给订阅者
            self.publish_message(rsp)
            return rsp
        await super().run(with_message)

    