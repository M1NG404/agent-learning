from app.service.similarity import cosine_similarity
from app.mapper.interface.vector_store_interface import VectorStoreInterface

# 内存版向量存储/检索
class InMemoryVectorStore(VectorStoreInterface):

    # 这个类是一个简单的内存向量存储器，用于存储和检索向量化的记忆数据。
    def __init__(self):
        self.items=[]

    # 添加一个新的记忆项到向量存储器中。
    def upsert(
            self,
            key:str,
            value:str,
            vector:list[float]
    ) -> None:

        for item in self.items:
            if item["key"] == key:
                item["value"] = value
                item["vector"] = vector
                return
        self.items.append(
            {
                "key":key,
                "value":value,
                "vector":vector
            }
        )

    def search(
            self,
            query_vector: list[float],
            top_k: int = 1
    ) -> list[dict]:
        # 计算每个存储项与查询向量的相似度
        scored_items = []
        for item in self.items:

            score = cosine_similarity(
                query_vector,
                item["vector"]
            )
            scored_items.append(
                {
                    "key": item["key"],
                    "value": item["value"],
                    "score": score
                }
            )

        # 按相似度排序并返回 top_k 个结果
        scored_items.sort(
            key=lambda x: x["score"],
            reverse=True
        )
        return scored_items[:top_k]