import os

from dependency_injector import containers, providers
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from memory.embedding import EmbeddingService
from memory.impl.qdrant_vector_store import QdrantVectorStore
from memory.manager import MemoryManager
from tools.registry import create_tool_registry

# 先加载环境变量
load_dotenv()

class Container(
    containers.DeclarativeContainer
):

    openai_client = providers.Singleton(
        OpenAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL")
    )

    embedding_service = providers.Singleton(
        EmbeddingService,
        client=openai_client
    )

    qdrant_client = providers.Singleton(
        QdrantClient,
        url="http://localhost:6333",
        check_compatibility=False
    )

    vector_store = providers.Singleton(
        QdrantVectorStore,
        client=qdrant_client,
        collection_name="agent_memory",
    )

    memory_manager = providers.Singleton(
        MemoryManager,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    tool_registry=providers.Callable(
        create_tool_registry,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
