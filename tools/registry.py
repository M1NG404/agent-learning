# 给 Runtime 用的 Tool 注册表

from functools import partial

from memory.embedding import EmbeddingService
from memory.store import MemoryStore
from models.order import OrderArgs
from models.memory_model import SaveMemoryArgs

from tools.order_tools import cancel_order, get_order
from tools.memory_tools import save_memory
from memory.vector_store import InMemoryVectorStore

def create_tool_registry(
        memory_store: MemoryStore,
        embedding_service: EmbeddingService,
        vector_store: InMemoryVectorStore   
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
                store=memory_store,
                embedding_service=embedding_service,
                vector_store=vector_store
            ),
            "args_model": SaveMemoryArgs
        }
    }