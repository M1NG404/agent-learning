from memory.store import MemoryStore
from memory.embedding import EmbeddingService
from memory.similarity import cosine_similarity

class MemoryManager:
    def __init__(
            self,
            store: MemoryStore,
            embedding_service:EmbeddingService
        ):
        self.store = store
        self.embedding_service = embedding_service

    def retrieve(
            self,
            query:str,
            top_k:int=1
        )->dict:
        memory = self.store.load()

        if not memory:
            return {}

        #1.用户问题转成向量
        query_vector = self.embedding_service.embed(query)
        scored_memories=[]
        #2.计算每条记忆的向量与用户问题向量的相似度
        for key, value in memory.items():
            # 把一条memory转换成自然语言文本
            memory_text=f"{key}:{value}"

            # 3.memory转向量
            memory_vector=self.embedding_service.embed(memory_text)

            # 4.计算query和这条memory的相似度
            score=cosine_similarity(
                query_vector,
                memory_vector
            )

            # 保存：key、value、score
            scored_memories.append(
                (key, value, score)
            )

        # 5.按相似度排序并返回top_k条记忆
        scored_memories.sort(
            key=lambda item: item[2], 
            reverse=True
        )

        # 6.返回 top_k 条记忆
        top_memories=scored_memories[:top_k]   

        # 7.转回dict
        result={} 

        for key, value, score in top_memories:
            result[key]=value

        return result



    def build_context(self, query: str) -> str:
        memory = self.retrieve(
            query=query,
            top_k=1
        )

        if not memory:
            return "没有找到与当前问题相关的长期记忆"

        return f"与当前问题相关的长期记忆：{memory}"