import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.runtime import run_agent
from models.agent_state import AgentState
import logging

from memory.store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

load_dotenv()

memory_store = MemoryStore()
memory = memory_store.load()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


state = AgentState(
    messages=[
    {
            "role": "system",
            "content": f"用户长期记忆：{memory}"
        },
        {
            "role": "user",
            "content": "我是谁"
        }
    ]
)


final_state = run_agent(
    client=client,
    state=state
)


print("Agent 最终状态:", final_state.status.value)
print("Agent 执行轮次:", final_state.iteration_count)

if final_state.final_answer:
    print("Agent 最终回答:", final_state.final_answer)