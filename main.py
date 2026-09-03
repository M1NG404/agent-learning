import logging

from app.core.runtime import run_agent
from app.config.container import Container
from app.pojo.agent_state import AgentState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# DI 容器
container = Container()

client = container.openai_client()
vector_store = container.vector_store()
memory_manager = container.memory_manager()
tool_registry = container.tool_registry()


user_input = "我最喜欢什么编程语言？"
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