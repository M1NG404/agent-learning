import json
import os
from pathlib import Path

import langextract as lx
from dotenv import load_dotenv
from langextract.factory import ModelConfig
from openai import OpenAI


# ============================================================
# 基础配置
# ============================================================

load_dotenv()


# 当前目录：
# experiments/fact_pipeline/
BASE_DIR = Path(__file__).parent

# 实验 1 的原始 Markdown
FACT_EXTRACTION_DIR = BASE_DIR.parent / "fact_extraction"
INPUT_DIR = FACT_EXTRACTION_DIR / "input"

# 实验 2 的 Entity Resolution Prompt
ENTITY_RESOLUTION_DIR = BASE_DIR.parent / "entity_resolution"
ENTITY_PROMPT_FILE = ENTITY_RESOLUTION_DIR / "prompt.txt"

# 实验 3 输出目录
OUTPUT_DIR = BASE_DIR / "output"


# 普通 LLM Client
# 专门用于 Entity Resolution
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# ============================================================
# 1. 行号处理
# ============================================================

def add_line_numbers(text: str) -> str:
    """
    给原始 Markdown 添加逻辑行号。

    原文：
        第一行
        第二行

    转换后：
        [L0001] 第一行
        [L0002] 第二行

    注意：
    带行号版本只用于 Entity Resolution。

    LangExtract 始终处理原始 raw_text。
    """

    lines = text.splitlines()

    result = []

    for index, line in enumerate(lines, start=1):
        result.append(f"[L{index:04d}] {line}")

    return "\n".join(result)


def char_position_to_line(
    text: str,
    char_pos: int,
) -> int:
    """
    把 LangExtract 的字符位置转换成 Markdown 行号。

    原理：

    字符位置之前存在几个换行符
        +
    1
        =
    当前行号
    """

    return text[:char_pos].count("\n") + 1


# ============================================================
# 2. Entity Resolution
# ============================================================

def resolve_entities(raw_text: str) -> dict:
    """
    专门解决实体指代。

    例如：

        江苏海达科技集团有限公司（以下简称本公司）

        ...

        同意本公司提供担保

    输出：

        本公司
            ->
        江苏海达科技集团有限公司
    """

    # 给同一份原始 Markdown 临时增加行号
    numbered_text = add_line_numbers(raw_text)

    # 使用实验 2 已经验证过的 Prompt
    prompt = ENTITY_PROMPT_FILE.read_text(
        encoding="utf-8",
    )

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
                "content": numbered_text,
            },
        ],
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    return json.loads(content)


def build_entity_map(result: dict) -> dict:
    """
    把 Entity Resolution 输出转换成：

        {
            10: "江阴利泰装饰材料有限公司",
            42: "江苏海达科技集团有限公司"
        }

    Java 类比：

        Map<Integer, String>
    """

    entity_map = {}

    mentions = result.get("mentions", [])

    for mention in mentions:
        mention_line = mention.get("mention_line")

        resolved_entity = mention.get(
            "resolved_entity"
        )

        if (
            mention_line is not None
            and resolved_entity
            and resolved_entity != "UNKNOWN"
        ):
            entity_map[mention_line] = resolved_entity

    return entity_map


# ============================================================
# 3. LangExtract
# ============================================================

def extract_facts(raw_text: str):
    """
    LangExtract 负责：

        1. 找到业务事实
        2. 提取 predicate
        3. 提取 value
        4. Grounding 到原文

    LangExtract 不负责最终解决：

        “本公司到底是谁”

    所以允许：

        subject_name = 本公司
    """

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

2. extraction_text 必须逐字复制原文中的连续文本。

3. 禁止改写、总结或把不同位置的文本拼接起来。

4. value 必须保持原文表达形式。

禁止自行计算、换算或者标准化。

例如：

原文：
壹亿元

正确：

value = 壹亿元

错误：

value = 100000000

5. 数字也必须以字符串形式输出。

6. 不需要负责解析“本公司”具体是哪家公司。

如果事实主体在当前原文中表达为：

本公司

则：

subject_name = 本公司

7. 不要因为前文出现过公司名称，就擅自把后文的“本公司”
转换成某个公司名称。

Entity Resolution 会在后续步骤专门解决这个问题。

8. 如果原文明确定义了完整实体名称，并且当前 extraction_text
中直接包含这个完整实体名称，则 subject_name 可以直接使用完整名称。

9. 一个文本片段包含多个独立事实，可以拆分成多条 Extraction。

10. 不要进行冲突判断。

11. 不要为了完整性猜测原文没有表达的信息。

12. attributes 必须包含：

subject_name
predicate
value
unit
"""

    # Few-shot example
    examples = [
        lx.data.ExampleData(
            text=(
                "江阴东华铝材科技有限公司"
                "尚欠贷款本金11521035.88元。"
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="fact",
                    extraction_text=(
                        "江阴东华铝材科技有限公司"
                        "尚欠贷款本金11521035.88元。"
                    ),
                    attributes={
                        "subject_name":
                            "江阴东华铝材科技有限公司",
                        "predicate": "贷款本金",
                        "value": "11521035.88",
                        "unit": "元",
                    },
                )
            ],
        )
    ]

    config = ModelConfig(
        model_id=os.getenv(
            "FACT_EXTRACT_MODEL",
            "qwen-plus",
        ),
        provider="openai",
        provider_kwargs={
            "api_key":
                os.getenv("OPENAI_API_KEY"),
            "base_url":
                os.getenv("OPENAI_API_BASE_URL"),
        },
    )

    return lx.extract(
        text_or_documents=raw_text,
        prompt_description=prompt,
        examples=examples,
        config=config,
    )


# ============================================================
# 4. Entity Resolution + LangExtract 合并
# ============================================================

def build_final_facts(
    raw_text: str,
    extraction_result,
    entity_map: dict,
) -> list:
    """
    把两个阶段合并成最终 Fact。

    ------------------------------------------------

    LangExtract：

        subject_name = 本公司
        predicate = 担保方式
        value = 连带保证责任担保
        char_interval = 802 ~ 898

    ↓

    char_position_to_line()

    ↓

        L0042

    ↓

    Entity Map：

        L0042
        ->
        江苏海达科技集团有限公司

    ↓

    Final Fact：

        江苏海达科技集团有限公司
        担保方式
        连带保证责任担保
    """

    final_facts = []

    for extraction in extraction_result.extractions:
        attributes = extraction.attributes or {}

        original_subject = attributes.get(
            "subject_name",
            "UNKNOWN",
        )

        predicate = attributes.get(
            "predicate",
            "",
        )

        value = attributes.get(
            "value",
            "",
        )

        unit = attributes.get(
            "unit",
            "",
        )

        char_interval = extraction.char_interval

        # ====================================================
        # 情况 1：
        # LangExtract 没有成功 Grounding
        # ====================================================

        if char_interval is None:
            fact = {
                "subject_name": original_subject,

                # 保证所有 Fact 都拥有同样的数据结构
                "original_subject_name": original_subject,

                "predicate": predicate,
                "value": value,
                "unit": unit,

                "line_start": None,
                "line_end": None,

                "char_start": None,
                "char_end": None,

                "evidence":
                    extraction.extraction_text,

                "grounded": False,

                # Grounding 都失败了，
                # 暂时不认为 subject 已经可靠验证
                "subject_resolved": False,
            }

            final_facts.append(fact)

            continue

        # ====================================================
        # 情况 2：
        # LangExtract 成功 Grounding
        # ====================================================

        start_pos = char_interval.start_pos
        end_pos = char_interval.end_pos

        # 字符位置 -> 起始行
        start_line = char_position_to_line(
            raw_text,
            start_pos,
        )

        # end_pos 通常是 extraction 结束后的下一个位置
        # 所以转换结束行时使用 end_pos - 1
        end_char_for_line = max(
            start_pos,
            end_pos - 1,
        )

        end_line = char_position_to_line(
            raw_text,
            end_char_for_line,
        )

        # ====================================================
        # Subject Resolution
        # ====================================================

        final_subject = original_subject

        subject_resolved = False

        # ----------------------------------------
        # LangExtract 返回的是“本公司”
        # ----------------------------------------

        if original_subject == "本公司":

            # extraction 本身必须真的包含“本公司”
            #
            # 否则可能是模型自己猜出来的 subject，
            # 这种情况不允许强行套 Entity Map。
            if "本公司" in extraction.extraction_text:

                resolved_entity = entity_map.get(
                    start_line
                )

                if resolved_entity:
                    final_subject = resolved_entity
                    subject_resolved = True

                else:
                    # 当前行没有找到明确 Entity Mention
                    #
                    # 实验阶段宁可 UNKNOWN，
                    # 不做猜测。
                    final_subject = "UNKNOWN"
                    subject_resolved = False

            else:
                # attributes 写了“本公司”，
                # 但证据里不存在“本公司”。
                #
                # 不允许推测。
                final_subject = "UNKNOWN"
                subject_resolved = False

        # ----------------------------------------
        # LangExtract 已经直接返回完整实体名称
        # ----------------------------------------

        else:
            final_subject = original_subject

            # 当前先认为明确实体名称已经 resolved
            #
            # 后续如果需要严格验证，
            # 可以继续增加“实体名称是否直接存在于 evidence”
            # 的检查。
            subject_resolved = (
                original_subject != "UNKNOWN"
            )

        # ====================================================
        # Final Fact
        # ====================================================

        fact = {
            "subject_name": final_subject,

            # 保存 LangExtract 原始输出，
            # 方便我们做实验对比。
            "original_subject_name":
                original_subject,

            "predicate": predicate,
            "value": value,
            "unit": unit,

            "line_start": start_line,
            "line_end": end_line,

            "char_start": start_pos,
            "char_end": end_pos,

            "evidence":
                extraction.extraction_text,

            "grounded": True,

            "subject_resolved":
                subject_resolved,
        }

        final_facts.append(fact)

    return final_facts


# ============================================================
# 5. 输出 Fact
# ============================================================

def print_fact(
    index: int,
    fact: dict,
):
    """
    格式化打印单个 Fact。

    使用 dict.get()，
    避免因为某个实验字段不存在导致程序整个崩掉。
    """

    print()
    print("-" * 60)
    print(f"Fact #{index}")

    subject_name = fact.get(
        "subject_name",
        "UNKNOWN",
    )

    original_subject_name = fact.get(
        "original_subject_name",
        subject_name,
    )

    print(
        "Subject：",
        subject_name,
    )

    if original_subject_name != subject_name:
        print(
            "原 Subject：",
            original_subject_name,
        )

    print(
        "Predicate：",
        fact.get("predicate", ""),
    )

    print(
        "Value：",
        fact.get("value", ""),
    )

    print(
        "Unit：",
        fact.get("unit", ""),
    )

    line_start = fact.get("line_start")

    line_end = fact.get("line_end")

    if line_start is None:
        print("行号：None")

    elif (
        line_end is None
        or line_start == line_end
    ):
        print(
            f"行号：L{line_start:04d}"
        )

    else:
        print(
            f"行号："
            f"L{line_start:04d}"
            f"~"
            f"L{line_end:04d}"
        )

    print(
        "Evidence：",
        fact.get("evidence", ""),
    )

    print(
        "Grounding：",
        "成功"
        if fact.get("grounded")
        else "失败",
    )

    print(
        "主体解析：",
        "成功"
        if fact.get("subject_resolved")
        else "失败",
    )


# ============================================================
# 6. main
# ============================================================

def main():
    # ========================================================
    # 找原始 Markdown
    # ========================================================

    files = list(
        INPUT_DIR.glob("*.md")
    )

    files.sort()

    if not files:
        print(
            f"没有找到原始 Markdown："
            f"{INPUT_DIR}"
        )
        return

    # 实验阶段仍然只处理第一份
    file_path = files[0]

    print("=" * 60)
    print(
        f"实验文件：{file_path.name}"
    )

    # ========================================================
    # 整个 Pipeline 只读取一次原始 Markdown
    # ========================================================

    raw_text = file_path.read_text(
        encoding="utf-8"
    )

    print()
    print(
        f"原始字符数：{len(raw_text)}"
    )

    print(
        f"原始行数："
        f"{len(raw_text.splitlines())}"
    )

    # ========================================================
    # Stage 1：
    # Entity Resolution
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Stage 1：Entity Resolution"
    )

    entity_result = resolve_entities(
        raw_text
    )

    entity_map = build_entity_map(
        entity_result
    )

    print()
    print("Entity Map：")

    for line, entity in sorted(
        entity_map.items()
    ):
        print(
            f"L{line:04d} -> {entity}"
        )

    # ========================================================
    # Stage 2：
    # LangExtract
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Stage 2：LangExtract"
    )

    extraction_result = extract_facts(
        raw_text
    )

    print(
        f"LangExtract Facts："
        f"{len(extraction_result.extractions)}"
    )

    # ========================================================
    # Stage 3：
    # Merge
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Stage 3：Merge"
    )

    final_facts = build_final_facts(
        raw_text=raw_text,
        extraction_result=extraction_result,
        entity_map=entity_map,
    )

    # ========================================================
    # 打印最终 Facts
    # ========================================================

    for index, fact in enumerate(
        final_facts,
        start=1,
    ):
        print_fact(
            index,
            fact,
        )

    # ========================================================
    # 统计
    # ========================================================

    total_facts = len(final_facts)

    grounded_count = sum(
        1
        for fact in final_facts
        if fact.get("grounded")
    )

    ungrounded_count = (
        total_facts - grounded_count
    )

    resolved_count = sum(
        1
        for fact in final_facts
        if fact.get("subject_resolved")
    )

    unresolved_count = (
        total_facts - resolved_count
    )

    # 统计真正由 Entity Resolution
    # 把“本公司”转换成实体名称的数量
    replaced_subject_count = sum(
        1
        for fact in final_facts
        if (
            fact.get("original_subject_name")
            == "本公司"
            and fact.get("subject_name")
            not in (
                "本公司",
                "UNKNOWN",
                None,
            )
        )
    )

    print()
    print("=" * 60)
    print("Pipeline 统计")

    print(
        f"最终 Fact：{total_facts}"
    )

    print(
        f"Grounding 成功："
        f"{grounded_count}"
    )

    print(
        f"Grounding 失败："
        f"{ungrounded_count}"
    )

    print(
        f"主体解析成功："
        f"{resolved_count}"
    )

    print(
        f"主体解析失败："
        f"{unresolved_count}"
    )

    print(
        f"本公司 -> 实体成功替换："
        f"{replaced_subject_count}"
    )

    # ========================================================
    # 保存结果
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "pipeline_result.json"
    )

    output_data = {
        "file": file_path.name,

        "model": os.getenv(
            "FACT_EXTRACT_MODEL",
            "qwen-plus",
        ),

        "entity_resolution": {
            "mentions":
                entity_result.get(
                    "mentions",
                    [],
                ),
            "entity_map":
                entity_map,
        },

        "facts":
            final_facts,

        "summary": {
            "total_facts":
                total_facts,

            "grounded":
                grounded_count,

            "ungrounded":
                ungrounded_count,

            "resolved_subjects":
                resolved_count,

            "unresolved_subjects":
                unresolved_count,

            "replaced_company_mentions":
                replaced_subject_count,
        },
    }

    output_file.write_text(
        json.dumps(
            output_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"结果已保存："
        f"{output_file}"
    )


if __name__ == "__main__":
    main()