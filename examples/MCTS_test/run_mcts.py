from examples.MCTS_test.MCTS import MCTS, Node, initialize_di_root_node
from examples.MCTS_test.utils import load_data_config, generate_task_requirement
import asyncio

data_config = load_data_config()


requirement = generate_task_requirement("house_prices", data_config)
print(requirement)

role, root_node = initialize_di_root_node(requirement, data_config)
# asyncio.run(role.run(requirement))

asyncio.run(root_node.run_node())

