# llm/qwen.py

from openai import OpenAI


client = OpenAI(
    api_key="xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def chat(messages, tools):
    return client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )