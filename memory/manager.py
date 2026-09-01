from memory.store import MemoryStore


class MemoryManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    def build_context(self) -> str:
        memory = self.store.load()

        if not memory:
            return "当前没有用户长期记忆。"

        return f"用户长期记忆：{memory}"