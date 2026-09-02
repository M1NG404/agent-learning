
# 给 Runtime 用的 Tool 注册表

from functools import partial

from memory.store import MemoryStore
from models.order import OrderArgs
from tools.memory_tools import save_memory
from tools.order_tools import cancel_order, get_order
from models.memory_model import SaveMemoryArgs  

def create_tool_registry(memory_store: MemoryStore):
    return {
        "get_order": {
                "function": get_order,
                "args_model": OrderArgs
            },
            "cancel_order": {
                "function": cancel_order,
                "args_model": OrderArgs
            },
            "save_memory": {
                "function": partial(
                    save_memory, 
                    store=memory_store
                ),   
                "args_model": SaveMemoryArgs  # 这里可以放置实际的保存内存的参数模型
            }
    }
