from memory.store import MemoryStore
from memory.embedding import EmbeddingService
from memory.interface.vector_store_interface import VectorStoreInterface

class MemoryManager:
    def __init__(
            self,
            store: MemoryStore,
            embedding_service:EmbeddingService,
            vector_store: VectorStoreInterface
        ):
        self.store = store
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
            self,
            query:str,
            top_k:int=1
        )->dict:

        # 1. 调用 EmbeddingService 将 query 转换为向量
        query_vector = self.embedding_service.embed(query)

        # 2. 调用 VectorStore 的 search 方法进行语义检索
        result = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        # 3. 将检索结果转换为字典形式，方便后续使用
        results={}

        # 4. 遍历检索结果，将每个结果的 key 和 value 存入 results 字典中
        for item in result:
            results[item["key"]] = item["value"]

        return results



    def build_context(self, query: str) -> str:
        memory = self.retrieve(
            query=query,
            top_k=1
        )

        if not memory:
            return "没有找到与当前问题相关的长期记忆"

        return f"与当前问题相关的长期记忆：{memory}"