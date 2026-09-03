import logging

from agent.runtime import run_agent
from config.container import Container
from models.agent_state import AgentState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# DI 容器
container = Container()

client = container.openai_client()
memory_store = container.memory_store()
vector_store = container.vector_store()
memory_manager = container.memory_manager()
tool_registry = container.tool_registry()

# 启动时同步持久化 Memory 到 Qdrant
memory = memory_store.load()

for key, item in memory.items():
    vector_store.upsert(
        key=key,
        value=item["value"],
        vector=item["embedding"],
    )

user_input = "记住，我最喜欢的编程语言是 Python"

memory_context = memory_manager.build_context(
    query=user_input
)

state = AgentState(
    messages=[
        {
            "role": "system",
            "content": memory_context,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]
)

final_state = run_agent(
    client=client,
    state=state,
    tool_registry=tool_registry,
)

print("Agent 最终状态:", final_state.status.value)
print("Agent 执行轮次:", final_state.iteration_count)

if final_state.final_answer:
    print("Agent 最终回答:", final_state.final_answer)