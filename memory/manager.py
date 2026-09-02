from memory.store import MemoryStore


class MemoryManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self,query:str)->dict:
        memory = self.store.load()

        # Memory 字段和自然语言关键词之间的映射
        keyword_map = {
            "user_name": ["名字", "叫什么", "是谁", "我是谁"],
            "favorite_language": ["编程语言", "语言", "Java", "Python"],
            "favorite_food": ["食物", "吃", "喜欢吃", "菜"]
        }

        result = {}

        # 遍历所有Memory字段
        for memory_key,keywords in keyword_map.items():

            # 当前memory中没有这个字段，直接跳过
            if memory_key not in memory:
                continue

            # 只要query命中任意关键字
            # 就把对应的Memory放进检索结果
            for keyword in keywords:
                if keyword in query:
                    result[memory_key] = memory[memory_key]
                    break  

        return result

    

    def build_context(self, query: str) -> str:
        memory = self.retrieve(query=query)

        if not memory:
            return "没有找到与当前问题相关的长期记忆"

        return f"与当前问题相关的长期记忆：{memory}"