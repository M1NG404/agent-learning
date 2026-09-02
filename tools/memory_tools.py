from memory.store import MemoryStore

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
        store: MemoryStore,
        key: str, 
        value: str
        ):
    memory = store.load()

    print("保存前:", memory)

    memory[key] = value

    print("保存后:", memory)
    print("保存路径:", store.file_path.resolve())

    store.save(memory)

    return {
        "success": True,
        "key": key,
        "value": value
    }