import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tools.registry import tool_registry

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "根据订单ID查询订单信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "根据订单ID取消订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]

messages = [
    {
       "role": "user",
        "content": "帮我取消订单1234567890"
    }
]

while True:

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # 记录 LLM 本轮输出
    messages.append(
        message.model_dump(exclude_none=True)
    )

    # 没有工具调用，说明任务完成
    if not message.tool_calls:
        print("最终回答：")
        print(message.content)
        break

    # 执行所有工具调用
    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        arguments = json.loads(
            tool_call.function.arguments
        )

        print("调用工具:", tool_name)
        print("参数:", arguments)

        tool = tool_registry.get(tool_name)

        if tool:
            result = tool(**arguments)
        else:
            result = {
                "error": f"未知工具: {tool_name}"
             }
       

        # 把工具结果写回 State
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False
                )
            }
        )