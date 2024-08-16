# -*- coding: utf-8 -*-
# @Date    : 12/20/2023 11:07 AM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import json
from datetime import datetime
from pathlib import Path

import nbformat

from metagpt.const import DATA_PATH
from metagpt.roles.role import Role
from metagpt.utils.common import read_json_file
from metagpt.utils.save_code import save_code_file



def save_history(role: Role, save_dir: str = "", name: str = "") -> Path:
    """
    Save plan and code execution history to the specified directory.

    Args:
        role (Role): The role containing the plan and execute_code attributes.
        save_dir (str): The directory to save the history.

    Returns:
        Path: The path to the saved history directory.
    """
    record_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_dir:
        save_path = Path(save_dir)
    else:
        save_path = DATA_PATH / "output" / f"{record_time}"

    # overwrite exist trajectory
    save_path.mkdir(parents=True, exist_ok=True)

    plan = role.planner.plan.dict()

    with open(save_path / "plan.json", "w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=4, ensure_ascii=False)

    if not name:
        name = Path(record_time)
    save_code_file(name=name, code_context=role.execute_code.nb, file_format="ipynb", save_dir=save_path)
    return save_path
