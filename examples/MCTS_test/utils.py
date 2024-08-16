import yaml
from examples.test_experiments.dataset_load import dataload
from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from nbclient import NotebookClient
from nbformat.notebooknode import NotebookNode
import nbformat


TASK_PROMPT = """\
# User requirement
{user_requirement}

# Data dir
training: {train_path}
testing: {test_path}
"""

def load_data_config(file_path="data.yaml"):
    with open(file_path, 'r') as stream:
        data_config = yaml.safe_load(stream)
    return data_config


def generate_task_requirement(task_name, data_config):
    data_path, user_requirement = dataload(task_name, data_config)
    train_path = f"{data_path}/split_train.csv"
    test_path = f"{data_path}/split_eval.csv"
    user_requirement = TASK_PROMPT.format(user_requirement=user_requirement, train_path=train_path, test_path=test_path)
    return user_requirement

def change_plan(role, plan):
    print(f"Change next plan to: {plan}")
    tasks = role.planner.plan.tasks
    finished = True
    for i, task in enumerate(tasks):
        if not task.code:
            finished = False
            break
    if not finished:
        tasks[i].plan = plan
    return finished
            


def is_cell_to_delete(cell: NotebookNode) -> bool:
    if "outputs" in cell:
        for output in cell["outputs"]:
            if output and "traceback" in output:
                return True
    return False


def process_cells(nb: NotebookNode) -> NotebookNode:
    new_cells = []
    i = 1
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and not is_cell_to_delete(cell):
            cell["execution_count"] = i
            new_cells.append(cell)
            i = i + 1
    nb["cells"] = new_cells
    return nb

def load_notebook(fpath, fname):
    fpath = f"{fpath}/{fname}.ipynb"
    nb = nbformat.read(open(fpath, "r", encoding="utf-8"), as_version=nbformat.NO_CONVERT)
    nb = process_cells(nb)
    execute_nb_code = ExecuteNbCode(nb)
    return execute_nb_code

async def load_execute_notebook(role):
    tasks = role.planner.plan.tasks
    codes = [task.code for task in tasks if task.code]
    executor = role.execute_code
    # await executor.build()
    for code in codes:
        outputs, success = await executor.run(code)
        print(f"Execution success: {success}, Output: {outputs}")
    print("Finish executing the loaded notebook")
    return executor

# 执行 notebook 中的所有代码单元
async def execute_notebook(executor:ExecuteNbCode):
    # 遍历并执行每个代码单元
    notebook = executor.nb
    await executor.build()
    num_cells = len(notebook.cells)
    print(f"Number of cells: {num_cells}")
    for i in range(num_cells):
        cell = notebook.cells[i]
        if cell.cell_type == 'code':  # 只执行代码单元格
            outputs, success = await executor.run_cell(cell, i)
            print(f"Execution success: {success}, Output: {outputs}")
    # executor.add_code_cell(code="train_data")
    # cell_index = len(executor.nb.cells) - 1
    # success, outputs = await executor.run_cell(executor.nb.cells[-1], cell_index)
    # print(f"End Execution success: {success}, Output: {outputs}")
