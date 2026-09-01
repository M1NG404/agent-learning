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
def save_memory(key: str, value: str):
    memory = memory_store.load()

    print("保存前:", memory)

    memory[key] = value

    print("保存后:", memory)
    print("保存路径:", memory_store.file_path.resolve())

    memory_store.save(memory)

    return {
        "success": True,
        "key": key,
        "value": value
    }