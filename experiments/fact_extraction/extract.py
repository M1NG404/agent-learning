import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 读取 .env
load_dotenv()


# 当前脚本目录
BASE_DIR = Path(__file__).parent

# 输入 Markdown 目录
INPUT_DIR = BASE_DIR / "input"

# Prompt 文件
PROMPT_FILE = BASE_DIR / "prompt.txt"

# 实验结果输出目录
OUTPUT_DIR = BASE_DIR / "output"


# 初始化 LLM Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


def add_line_numbers(text: str) -> str:
    """
    给 Markdown 每一行增加行号。
    """

    lines = text.splitlines()

    result = []

    for index, line in enumerate(lines, start=1):
        result.append(f"[L{index:04d}] {line}")

    return "\n".join(result)


def extract_facts(file_path: Path):
    """
    把一个 Markdown 文件发送给 LLM，
    让模型返回结构化事实。
    """

    # 1. 读取 Markdown 原文
    text = file_path.read_text(encoding="utf-8")

    # 2. 给原文增加行号
    numbered_text = add_line_numbers(text)

    # 3. 读取事实抽取 Prompt
    prompt = PROMPT_FILE.read_text(encoding="utf-8")

    # 4. 调用 LLM
    response = client.chat.completions.create(
        model=os.getenv("FACT_EXTRACT_MODEL", "qwen-plus"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"""
文件名：

{file_path.name}

文档内容：

{numbered_text}
""",
            },
        ],
        # 要求模型返回 JSON
        response_format={
            "type": "json_object"
        },
    )

    # 5. 取得模型返回的文本
    content = response.choices[0].message.content

    # 6. JSON 字符串 -> Python dict
    result = json.loads(content)

    return result

def main():
    # 找 input 下所有 Markdown
    files = list(INPUT_DIR.glob("*.md"))

    print(f"发现 Markdown 文件：{len(files)}")

    if not files:
        print("没有找到 Markdown 文件")
        return

    # 暂时仍然只测试第一个文件
    file_path = files[0]

    print()
    print("=" * 60)
    print(f"正在处理：{file_path.name}")

    # 调用 LLM 抽取事实
    result = extract_facts(file_path)

    print()
    print("=" * 60)
    print("模型返回结果：")

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    # 确保 output 目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存本次实验结果
    output_file = OUTPUT_DIR / "result.json"

    output_file.write_text(
        json.dumps(
            {
                "model": os.getenv("FACT_EXTRACT_MODEL", "qwen-plus"),
                "prompt_version": "v2",
                "file": file_path.name,
                "facts": result.get("facts", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"实验结果已保存：{output_file}")


if __name__ == "__main__":
    main()