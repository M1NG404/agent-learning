from memory.store import MemoryStore

store=MemoryStore()

store.save(
    {
        "user_name":"小明"
    }
)

memory=store.load()
print(memory)