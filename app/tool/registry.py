# 给 Runtime 用的 Tool 注册表

from functools import partial

from app.service.embedding_service import EmbeddingService
from app.pojo.order import OrderArgs
from app.pojo.memory import SaveMemoryArgs

from app.tool.order_tools import cancel_order, get_order
from app.tool.memory_tools import save_memory
from app.mapper.interface.vector_store_interface import VectorStoreInterface

def create_tool_registry(
        embedding_service: EmbeddingService,
        vector_store: VectorStoreInterface   
        ):
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
                embedding_service=embedding_service,
                vector_store=vector_store
            ),
            "args_model": SaveMemoryArgs
        }
    }