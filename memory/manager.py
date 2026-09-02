from memory.store import MemoryStore


class MemoryManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self)->dict:
        return self.store.load()

    def build_context(self) -> str:
        memory = self.retrieve()

        if not memory:
            return "当前没有用户长期记忆。"

        return f"用户长期记忆：{memory}"