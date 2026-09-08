import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 读取项目根目录 .env
load_dotenv()


# 当前实验目录：
# experiments/entity_resolution/
BASE_DIR = Path(__file__).parent

# 测试 Markdown 输入目录
INPUT_DIR = BASE_DIR / "input"

# Entity Resolution Prompt
PROMPT_FILE = BASE_DIR / "prompt.txt"


# 初始化 LLM 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


def resolve_entities(file_path: Path):
    """
    对一个 Markdown 文档进行实体指代消解。

    例如：

    江苏海达科技集团有限公司（以下简称本公司）
        ↓

    “本公司”
        ↓

    江苏海达科技集团有限公司
    """

    # 1. 读取 Markdown 文档
    text = file_path.read_text(encoding="utf-8")

    # 2. 读取实体消解 Prompt
    prompt = PROMPT_FILE.read_text(encoding="utf-8")

    # 3. 调用 LLM
    response = client.chat.completions.create(
        model=os.getenv(
            "FACT_EXTRACT_MODEL",
            "qwen-plus",
        ),
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

    # 4. 获取模型输出
    content = response.choices[0].message.content

    # 5. JSON 字符串 -> Python dict
    result = json.loads(content)

    return result


def main():
    # 查找 input 目录下全部 Markdown
    files = list(INPUT_DIR.glob("*.md"))

    # 为了每次实验顺序一致
    files.sort()

    print(f"发现测试文件：{len(files)}")

    if not files:
        print("没有找到测试文件")
        return

    # 逐个处理测试文件
    #
    # 例如：
    #
    # sample_local.md
    # sample_whole.md
    #
    for index, file_path in enumerate(files, start=1):

        print()
        print("=" * 60)
        print(f"[{index}/{len(files)}] 正在处理：{file_path.name}")

        try:
            # 执行 Entity Resolution
            result = resolve_entities(file_path)

            print()
            print("实体消解结果：")

            # 格式化打印 JSON
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            # 简单打印本次解析出的 mention 数量
            mentions = result.get("mentions", [])

            print()
            print(f"解析指代数量：{len(mentions)}")

        except Exception as e:
            print()
            print(f"处理失败：{e}")


if __name__ == "__main__":
    main()