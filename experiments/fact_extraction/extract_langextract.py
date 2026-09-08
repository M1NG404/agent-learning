import os
from pathlib import Path

import langextract as lx
from dotenv import load_dotenv
from langextract.factory import ModelConfig


# 读取项目根目录中的 .env
load_dotenv()


# 当前实验目录：
# experiments/fact_extraction/
BASE_DIR = Path(__file__).parent

# Markdown 输入目录
INPUT_DIR = BASE_DIR / "input"


def main():
    # 1. 找 input 目录下的 Markdown
    files = list(INPUT_DIR.glob("*.md"))

    print(f"发现 Markdown 文件：{len(files)}")

    if not files:
        print("没有找到 Markdown 文件")
        return

    # 暂时仍然只测试第一份
    file_path = files[0]

    print()
    print("=" * 60)
    print(f"正在处理：{file_path.name}")

    # 2. 读取完整 Markdown
    text = file_path.read_text(encoding="utf-8")

    # 3. 定义 LangExtract 的抽取任务
    #
    # 这里重点解决上次实验中的三个问题：
    #
    # ① extraction_text 被模型改写，导致 char_interval=None
    # ② value 被自动标准化
    # ③ subject_type 自己创造 ASSET / DATE 等类型
    prompt = """
你是一个不良资产尽调资料事实抽取器。

你的任务是从输入文档中提取明确存在于原文中的业务事实。

一个事实表示：

某个主体
+
某个属性
+
某个值

必须严格遵守以下规则：

1. 只能提取原文明确表达的事实。
2. 禁止根据常识或推测补充原文不存在的信息。
3. 不要为了完整性而猜测。

4. extraction_text 必须逐字复制原文中的一段连续文本。
   禁止改写。
   禁止总结。
   禁止重新组织语句。
   禁止把不同位置的文字拼接成一句话。

5. extraction_text 的主要作用是原文溯源。
   必须保证这段文字能够在原始文档中直接找到。

6. subject_name 可以根据明确的上下文指代进行解析。
   例如：
   “江苏海达科技集团有限公司（以下简称本公司）”
   后文出现“本公司”时，
   subject_name 可以填写“江苏海达科技集团有限公司”。

7. 如果无法明确判断主体是谁：
   subject_name 必须填写 UNKNOWN。

8. value 必须保持原文表达形式。
   不允许自行计算、转换或标准化。

   例如：

   原文：
   贰仟万

   正确：
   value = 贰仟万

   错误：
   value = 20000000

9. 数字也必须以字符串形式放入 value。

10. 一个文本片段中包含多个独立事实时，可以拆成多条 Extraction。

11. 不要判断事实之间是否冲突。

12. subject_type 只能使用以下值：

COMPANY
PERSON
PROPERTY
CLAIM
CONTRACT
CASE
OTHER
UNKNOWN

禁止创建新的 subject_type，例如：
ASSET
DATE
ORGANIZATION

13. 不要仅仅因为文档出现了一个公司名称、人员姓名或日期，
    就生成“公司名称”“人员名称”“日期”这种没有业务意义的事实。

14. attributes 必须包含：

subject_type
subject_name
predicate
value
unit
"""


    # 4. Few-shot Example
    #
    # 特别注意：
    # extraction_text 直接复制输入 text 的完整原句，
    # 不能自己重新组织。
    examples = [
        lx.data.ExampleData(
            text="江阴东华铝材科技有限公司尚欠贷款本金11521035.88元。",
            extractions=[
                lx.data.Extraction(
                    extraction_class="fact",

                    # 必须是上面 text 中真实、连续出现的原文
                    extraction_text=(
                        "江阴东华铝材科技有限公司尚欠贷款本金11521035.88元。"
                    ),

                    attributes={
                        "subject_type": "COMPANY",
                        "subject_name": "江阴东华铝材科技有限公司",
                        "predicate": "贷款本金",

                        # 保留原文，不做数字转换
                        "value": "11521035.88",

                        "unit": "元",
                    },
                )
            ],
        )
    ]


    # 5. 配置 Qwen
    #
    # LangExtract 负责抽取流程和 grounding，
    # 底层模型仍然使用我们 baseline 中的 Qwen。
    config = ModelConfig(
        model_id=os.getenv(
            "FACT_EXTRACT_MODEL",
            "qwen-plus",
        ),
        provider="openai",
        provider_kwargs={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_API_BASE_URL"),
        },
    )


    # 6. 执行 LangExtract
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        config=config,
    )


    # 7. 打印结果
    print()
    print("=" * 60)
    print(f"抽取数量：{len(result.extractions)}")

    # 顺便统计 grounding 成功率
    grounded_count = 0

    for extraction in result.extractions:
        print()
        print("类型：", extraction.extraction_class)
        print("原文：", extraction.extraction_text)
        print("属性：", extraction.attributes)
        print("位置：", extraction.char_interval)

        # 有 char_interval 表示成功定位回原文
        if extraction.char_interval is not None:
            grounded_count += 1

    print()
    print("=" * 60)
    print("Grounding 统计：")
    print(f"成功：{grounded_count}")
    print(f"失败：{len(result.extractions) - grounded_count}")
    print(f"总数：{len(result.extractions)}")


if __name__ == "__main__":
    main()