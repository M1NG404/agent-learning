import json

from openai import OpenAI
from pydantic import ValidationError

from models.agent_state import AgentState, AgentStatus
from tools.definitions import tools
from tools.registry import tool_registry

MAX_ITERATIONS = 10

def run_agent(client: OpenAI, state: AgentState):
    while True:

        state.iteration_count += 1

        if state.iteration_count > MAX_ITERATIONS:
            state.status = AgentStatus.MAX_ITERATIONS_REACHED
            break

        print("Agent 当前轮次:", state.iteration_count)

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=state.messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        state.messages.append(
            message.model_dump(exclude_none=True)
        )

        if not message.tool_calls:
            state.status = AgentStatus.COMPLETED

            print("最终回答：")
            print(message.content)
            break

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                print("调用工具:", tool_name)
                print("参数:", arguments)

                tool_info = tool_registry.get(tool_name)

                if not tool_info:
                    result = {
                        "error": f"未知工具: {tool_name}"
                    }

                else:
                    tool = tool_info["function"]
                    args_model = tool_info["args_model"]

                    validated_args = args_model(**arguments)

                    result = tool(
                        **validated_args.model_dump()
                    )

            except json.JSONDecodeError as e:
                result = {
                    "error": "工具参数 JSON 解析失败",
                    "details": str(e)
                }

            except ValidationError as e:
                result = {
                    "error": "工具参数校验失败",
                    "details": e.errors()
                }

            except Exception as e:
                result = {
                    "error": "工具执行失败",
                    "details": str(e)
                }

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                }
            )

    return state