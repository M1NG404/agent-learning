from memory.embedding import EmbeddingService
from memory.store import MemoryStore

memory_store=MemoryStore()

# save_memory(key, value)
#         ↓
# 先 load 现有 memory
#         ↓
# memory[key] = value
#         ↓
# save 回 memory.json
#         ↓
# 返回执行结果
def save_memory(
        store:MemoryStore,
        embedding_service: EmbeddingService,
        key: str,
        value: str,
    ):
    # 1.读取现有的memory
    memory = store.load()

    # 2.把现有的memory转化成适合做语义检索的文本
    memory_text=f"{key}:{value}"

    # 3.调EmbeddingService把memory_text转成向量
    memory_vector=embedding_service.embed(
        memory_text
    )

    # 4.保存value+vector到memory中
    memory[key] = {
        "value":value,
        "embedding":memory_vector
    }
    # 5.持久化到memory.json
    memory_store.save(memory)

    return {
        "success": True,
        "key": key,
        "value": value
    }