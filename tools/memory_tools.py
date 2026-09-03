from memory.embedding import EmbeddingService
from memory.interface.vector_store_interface import VectorStoreInterface


def save_memory(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreInterface,
    key: str,
    value: str,
):
    # 将 Memory 文本转换成向量
    memory_text = f"{key}:{value}"
    memory_vector = embedding_service.embed(memory_text)

    # 写入向量数据库
    vector_store.upsert(
        key=key,
        value=value,
        vector=memory_vector,
    )

    # 返回 Tool 执行结果
    return {
        "success": True,
        "key": key,
        "value": value,
    }