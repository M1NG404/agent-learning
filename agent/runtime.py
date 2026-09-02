import json

import logging

from openai import OpenAI
from pydantic import ValidationError

from models.agent_state import AgentState, AgentStatus
from tools.definitions import tools

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 10

def run_agent(
        client: OpenAI, 
        state: AgentState,
        tool_registry
        ):
    while True:

        state.iteration_count += 1

        if state.iteration_count > MAX_ITERATIONS:
            state.status = AgentStatus.MAX_ITERATIONS_REACHED
            break

        logger.info(
            "Agent 当前轮次: %s",
            state.iteration_count
        )

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
            state.final_answer = message.content

            break

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                logger.info("调用工具: %s", tool_name)
                logger.info("工具参数: %s", arguments)

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
                    logger.info("工具执行结果: %s", result)

            except json.JSONDecodeError as e:
                result = {
                    "error": "工具参数 JSON 解析失败",
                    "details": str(e)
                }
                logger.error("工具参数 JSON 解析失败: %s", e)
            except ValidationError as e:
                result = {
                    "error": "工具参数校验失败",
                    "details": e.errors()
                }
                logger.error("工具参数校验失败: %s", e)
            except Exception as e:
                result = {
                    "error": "工具执行失败",
                    "details": str(e)
                }
                logger.error("工具执行失败: %s", e)
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