import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.runtime import run_agent
from models.agent_state import AgentState
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


state = AgentState(
    messages=[
        {
            "role": "user",
            "content": "帮我取消订单1234567890"
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