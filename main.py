import os
import logging

from dotenv import load_dotenv
from openai import OpenAI

from agent.runtime import run_agent
from models.agent_state import AgentState

from memory.store import MemoryStore
from memory.manager import MemoryManager
from memory.embedding import EmbeddingService
from tools.registry import create_tool_registry


# 配置日志
# Runtime 中的 logger.info / logger.error 会按照这个格式输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


# 加载 .env 文件中的环境变量
# 例如：
# OPENAI_API_KEY
# OPENAI_API_BASE_URL
load_dotenv()

# =========================
# 5. 创建 LLM Client
# =========================
# 当前使用 OpenAI-compatible SDK 调用 Qwen
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# =========================
# 1. 创建 MemoryStore
# =========================
# MemoryStore 负责长期记忆的持久化读写
# 当前底层存储是 memory.json
memory_store = MemoryStore()


# =========================
# 2. 创建 MemoryManager
# =========================
# 把同一个 memory_store 注入给 MemoryManager
#
# MemoryManager 负责：
# Memory -> LLM Context
#
# 例如：
# {"user_name": "小明"}
#        ↓
# "用户长期记忆：{'user_name': '小明'}"
embedding_service=EmbeddingService(
    client=client
)

memory_manager = MemoryManager(
    store=memory_store,
    embedding_service=embedding_service
)


# =========================
# 3. 创建 Tool Registry
# =========================
# 把同一个 memory_store 传给 Tool Registry
#
# create_tool_registry 内部会把 memory_store
# 注入 / 绑定给 save_memory Tool
#
# 最终效果类似：
#
# save_memory(
#     store=memory_store,
#     key="user_name",
#     value="小明"
# )
tool_registry = create_tool_registry(
    memory_store=memory_store,
    embedding_service=embedding_service
)


# =========================
# 4. 构建 Memory Context
# =========================
# 从长期 Memory 中读取数据，
# 然后转换成可以放进 messages 的文本 Context
user_input="我是谁"

memory_context = memory_manager.build_context(
    query=user_input
)





# =========================
# 6. 创建 AgentState
# =========================
# AgentState 保存当前这一次 Agent Run 的运行状态
#
# 目前包括：
# - messages
# - iteration_count
# - status
# - final_answer
state = AgentState(
    messages=[
        {
            # system message 中注入长期 Memory
            # 这样 LLM 才能真正看到 Memory
            "role": "system",
            "content": memory_context
        },
        {
            # 当前用户请求
            "role": "user",
            "content": user_input
        }
    ]
)


# =========================
# 7. 启动 Agent Runtime
# =========================
# Runtime 负责：
#
# while True
#   ↓
# 调用 LLM
#   ↓
# 判断 tool_calls
#   ↓
# Tool Registry 查找工具
#   ↓
# Pydantic 参数校验
#   ↓
# 执行 Tool
#   ↓
# Tool Result 写回 messages
#   ↓
# 下一轮 LLM
#
# tool_registry 是由 main.py 创建后注入 Runtime 的，
# Runtime 本身不再负责创建 Registry
final_state = run_agent(
    client=client,
    state=state,
    tool_registry=tool_registry
)


# =========================
# 8. 输出最终运行结果
# =========================
print(
    "Agent 最终状态:",
    final_state.status.value
)

print(
    "Agent 执行轮次:",
    final_state.iteration_count
)

if final_state.final_answer:
    print(
        "Agent 最终回答:",
        final_state.final_answer
    )