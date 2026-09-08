import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 读取 .env
load_dotenv()


# 当前实验目录
BASE_DIR = Path(__file__).parent

# 输入目录
INPUT_DIR = BASE_DIR / "input"

# Prompt 文件
PROMPT_FILE = BASE_DIR / "prompt.txt"


# 初始化模型客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


def resolve_entities(file_path: Path):
    """
    对一个 Markdown 文档进行实体指代消解。

    例如：
    “本公司”
        ↓
    “江苏海达科技集团有限公司”
    """

    # 1. 读取测试文档
    text = file_path.read_text(encoding="utf-8")

    # 2. 读取 Prompt
    prompt = PROMPT_FILE.read_text(encoding="utf-8")

    # 3. 调用 LLM
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
                "content": text,
            },
        ],
        response_format={
            "type": "json_object"
        },
    )

    # 4. 拿到模型返回的 JSON 字符串
    content = response.choices[0].message.content

    # 5. JSON 字符串 -> Python dict
    return json.loads(content)


def main():
    # 查找测试 Markdown
    files = list(INPUT_DIR.glob("*.md"))

    print(f"发现测试文件：{len(files)}")

    if not files:
        print("没有找到测试文件")
        return

    # 暂时只跑第一份
    file_path = files[0]

    print()
    print("=" * 60)
    print(f"正在处理：{file_path.name}")

    result = resolve_entities(file_path)

    print()
    print("=" * 60)
    print("实体消解结果：")

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()