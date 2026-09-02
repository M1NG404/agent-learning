from memory.embedding import EmbeddingService
from memory.store import MemoryStore
from memory.vector_store import InMemoryVectorStore


def save_memory(
        store: MemoryStore,
        embedding_service: EmbeddingService,
        vector_store: InMemoryVectorStore,
        key: str,
        value: str,
    ):

    # =========================
    # 1. 读取已有长期记忆
    # =========================
    # MemoryStore 负责持久化存储。
    # 当前底层是 memory.json。
    #
    # 例如：
    # {
    #   "name": {
    #       "value": "小明",
    #       "embedding": [...]
    #   }
    # }
    memory = store.load()


    # =========================
    # 2. 构造用于 Embedding 的文本
    # =========================
    # 不只对 value 做 Embedding，
    # 而是把 key 和 value 一起作为语义信息。
    #
    # 例如：
    # key   = "language"
    # value = "Java"
    #
    # memory_text:
    # "language:Java"
    memory_text = f"{key}:{value}"


    # =========================
    # 3. 生成 Memory Vector
    # =========================
    # EmbeddingService：
    #
    # text
    #   ↓
    # Embedding Model
    #   ↓
    # vector
    #
    # 这个 vector 后面会同时：
    # 1. 持久化到 memory.json
    # 2. 写入 VectorStore 用于语义搜索
    memory_vector = embedding_service.embed(
        memory_text
    )


    # =========================
    # 4. 更新 Memory 数据
    # =========================
    # 把原来的：
    #
    # key -> value
    #
    # 升级为：
    #
    # key -> {
    #     value,
    #     embedding
    # }
    #
    # embedding 在写入时提前计算，
    # 查询时就不需要重新为每条 Memory 调 Embedding API。
    memory[key] = {
        "value": value,
        "embedding": memory_vector
    }


    # =========================
    # 5. 持久化 Memory
    # =========================
    # 保存到 memory.json。
    #
    # 作用：
    # 即使程序关闭 / 重启，
    # Memory 仍然存在。
    store.save(memory)


    # =========================
    # 6. 同步写入 VectorStore
    # =========================
    # MemoryStore 和 VectorStore 职责不同：
    #
    # MemoryStore
    # → 负责持久化
    #
    # VectorStore
    # → 负责向量搜索
    #
    # 因此新 Memory 保存后，
    # 当前运行中的 VectorStore 也必须同步更新。
    vector_store.add(
        key=key,
        value=value,
        vector=memory_vector
    )


    # =========================
    # 7. 返回 Tool 执行结果
    # =========================
    # 这个结果最终会返回给 Agent Runtime，
    # 再作为 Tool Result 写回 messages。
    return {
        "success": True,
        "key": key,
        "value": value
    }