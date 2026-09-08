import json
import os
from dataclasses import dataclass
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

# 原始 Markdown
FACT_EXTRACTION_DIR = BASE_DIR.parent / "fact_extraction"
INPUT_DIR = FACT_EXTRACTION_DIR / "input"

# Entity Resolution Prompt
ENTITY_RESOLUTION_DIR = BASE_DIR.parent / "entity_resolution"
ENTITY_PROMPT_FILE = ENTITY_RESOLUTION_DIR / "prompt.txt"

# Pipeline 输出
OUTPUT_DIR = BASE_DIR / "output"


# Entity Resolution 使用的 LLM Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EntityBlock:
    """
    表示文档中属于某个主体的一段连续区间。

    例如：

    L0007 ~ L0038
        ->
    江阴利泰装饰材料有限公司
    """

    start_line: int
    end_line: int
    entity: str


# ============================================================
# 1. 行号工具
# ============================================================

def add_line_numbers(text: str) -> str:
    """
    给 Markdown 添加行号。

    原文：

        第一行
        第二行

    转换：

        [L0001] 第一行
        [L0002] 第二行

    带行号文本只给 Entity Resolution 使用。
    LangExtract 始终处理原始 raw_text。
    """

    lines = text.splitlines()

    result = []

    for index, line in enumerate(lines, start=1):
        result.append(
            f"[L{index:04d}] {line}"
        )

    return "\n".join(result)


def char_position_to_line(
    text: str,
    char_pos: int,
) -> int:
    """
    把 LangExtract 的字符位置转换成 Markdown 行号。

    原理：

    char_pos 前面有多少个换行符
                +
                1
                =
              行号
    """

    return text[:char_pos].count("\n") + 1


# ============================================================
# 2. Entity Resolution
# ============================================================

def resolve_entities(
    raw_text: str,
) -> dict:
    """
    专门解决：

        本公司
          ↓
        到底是谁？

    不负责 Fact Extraction。
    """

    numbered_text = add_line_numbers(
        raw_text
    )

    prompt = ENTITY_PROMPT_FILE.read_text(
        encoding="utf-8"
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

    content = (
        response
        .choices[0]
        .message
        .content
    )

    return json.loads(content)


def build_entity_map(
    result: dict,
) -> dict:
    """
    把：

        {
            "mention_line": 42,
            "resolved_entity": "江苏海达科技集团有限公司"
        }

    转换成：

        {
            42: "江苏海达科技集团有限公司"
        }

    用于精确解析：

        当前行的“本公司”是谁？
    """

    entity_map = {}

    mentions = result.get(
        "mentions",
        [],
    )

    for mention in mentions:

        mention_line = mention.get(
            "mention_line"
        )

        resolved_entity = mention.get(
            "resolved_entity"
        )

        if (
            mention_line is not None
            and resolved_entity
            and resolved_entity != "UNKNOWN"
        ):
            entity_map[
                mention_line
            ] = resolved_entity

    return entity_map


# ============================================================
# 3. Block Ownership
# ============================================================

def build_entity_blocks(
    entity_result: dict,
    total_lines: int,
) -> list[EntityBlock]:
    """
    根据 definition_line 自动构建主体区间。

    例如：

        L0007 -> 江阴利泰
        L0039 -> 江苏海达
        L0076 -> 海达特种人革
        L0113 -> 海达彩涂

    自动生成：

        L0007 ~ L0038 -> 江阴利泰
        L0039 ~ L0075 -> 江苏海达
        L0076 ~ L0112 -> 海达特种人革
        L0113 ~ 文档末尾 -> 海达彩涂
    """

    definitions = {}

    mentions = entity_result.get(
        "mentions",
        [],
    )

    for mention in mentions:

        definition_line = mention.get(
            "definition_line"
        )

        resolved_entity = mention.get(
            "resolved_entity"
        )

        if (
            definition_line is not None
            and resolved_entity
            and resolved_entity != "UNKNOWN"
        ):
            # 同一个 definition_line 可能重复出现
            # 用 dict 自动去重
            definitions[
                definition_line
            ] = resolved_entity

    sorted_definitions = sorted(
        definitions.items()
    )

    blocks = []

    for index, (
        start_line,
        entity,
    ) in enumerate(
        sorted_definitions
    ):

        if (
            index + 1
            < len(sorted_definitions)
        ):
            next_start_line = (
                sorted_definitions[
                    index + 1
                ][0]
            )

            end_line = (
                next_start_line - 1
            )

        else:
            end_line = total_lines

        blocks.append(
            EntityBlock(
                start_line=start_line,
                end_line=end_line,
                entity=entity,
            )
        )

    return blocks


def find_block_owner(
    line: int,
    blocks: list[EntityBlock],
) -> str:
    """
    根据行号查询所在主体块。
    """

    for block in blocks:

        if (
            block.start_line
            <= line
            <= block.end_line
        ):
            return block.entity

    return "UNKNOWN"


# ============================================================
# 4. LangExtract
# ============================================================

def extract_facts(
    raw_text: str,
):
    """
    LangExtract 负责：

    1. 找事实
    2. predicate
    3. value
    4. Grounding

    不要求它解决：

        本公司到底是谁
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

3. 禁止改写、总结或拼接不同位置的文本。

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

7. 不要根据长距离上下文擅自把“本公司”
转换成某个公司名称。

Entity Resolution 会在后续步骤专门处理。

8. 如果当前 extraction_text 中直接包含完整实体名称，
subject_name 可以直接使用该完整实体名称。

9. 一个文本片段包含多个独立事实时，
可以拆分成多条 Extraction。

10. 不要进行冲突判断。

11. 不要为了完整性猜测原文没有表达的信息。

12. attributes 必须包含：

subject_name
predicate
value
unit
"""

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
                        "predicate":
                            "贷款本金",
                        "value":
                            "11521035.88",
                        "unit":
                            "元",
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
                os.getenv(
                    "OPENAI_API_KEY"
                ),
            "base_url":
                os.getenv(
                    "OPENAI_API_BASE_URL"
                ),
        },
    )

    return lx.extract(
        text_or_documents=raw_text,
        prompt_description=prompt,
        examples=examples,
        config=config,
    )


# ============================================================
# 5. 主体解析
# ============================================================

def resolve_fact_subject(
    original_subject: str,
    evidence: str,
    start_line: int,
    entity_map: dict,
    entity_blocks: list[EntityBlock],
) -> tuple[str, bool, str]:
    """
    最终主体解析规则。

    优先级：

    1. 明确 subject 确实存在于 evidence
       -> DIRECT

    2. evidence 中有“本公司”
       -> Entity Resolution

    3. 上面都不成立
       -> Block Ownership

    4. 全失败
       -> UNKNOWN
    """

    # ========================================================
    # 1. LangExtract 给出了明确 subject
    # ========================================================

    if (
        original_subject
        not in (
            "本公司",
            "UNKNOWN",
            "",
            None,
        )
    ):
        # ----------------------------------------------------
        # 只有主体真的出现在证据里，
        # 才允许 DIRECT。
        #
        # 例如：
        #
        # Evidence:
        # 澄土国用（2008）第3884号项下...
        #
        # Subject:
        # 澄土国用（2008）第3884号
        #
        # => DIRECT
        # ----------------------------------------------------

        if original_subject in evidence:

            return (
                original_subject,
                True,
                "DIRECT",
            )

        # ----------------------------------------------------
        # 模型给了明确主体，
        # 但证据根本没有这个主体。
        #
        # 不相信模型。
        #
        # 改用 Block Ownership。
        # ----------------------------------------------------

        block_owner = find_block_owner(
            start_line,
            entity_blocks,
        )

        if block_owner != "UNKNOWN":

            return (
                block_owner,
                True,
                "BLOCK_OWNERSHIP",
            )

        return (
            "UNKNOWN",
            False,
            "NONE",
        )

    # ========================================================
    # 2. subject = 本公司
    # ========================================================

    if original_subject == "本公司":

        # ----------------------------------------------------
        # Evidence 确实包含“本公司”
        #
        # 优先精确 Entity Resolution
        # ----------------------------------------------------

        if "本公司" in evidence:

            resolved_entity = entity_map.get(
                start_line
            )

            if resolved_entity:

                return (
                    resolved_entity,
                    True,
                    "ENTITY_RESOLUTION",
                )

        # ----------------------------------------------------
        # 精确 Entity Resolution 没匹配到，
        # 或 evidence 本身没有“本公司”
        #
        # 回退到 Block Ownership
        # ----------------------------------------------------

        block_owner = find_block_owner(
            start_line,
            entity_blocks,
        )

        if block_owner != "UNKNOWN":

            return (
                block_owner,
                True,
                "BLOCK_OWNERSHIP",
            )

        return (
            "UNKNOWN",
            False,
            "NONE",
        )

    # ========================================================
    # 3. subject = UNKNOWN / 空
    # ========================================================

    block_owner = find_block_owner(
        start_line,
        entity_blocks,
    )

    if block_owner != "UNKNOWN":

        return (
            block_owner,
            True,
            "BLOCK_OWNERSHIP",
        )

    return (
        "UNKNOWN",
        False,
        "NONE",
    )


# ============================================================
# 6. Merge
# ============================================================

def build_final_facts(
    raw_text: str,
    extraction_result,
    entity_map: dict,
    entity_blocks: list[EntityBlock],
) -> list:
    """
    LangExtract
        +
    Entity Resolution
        +
    Block Ownership
        ↓
    Final Facts
    """

    final_facts = []

    for extraction in (
        extraction_result.extractions
    ):

        attributes = (
            extraction.attributes
            or {}
        )

        original_subject = (
            attributes.get(
                "subject_name",
                "UNKNOWN",
            )
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

        evidence = (
            extraction.extraction_text
            or ""
        )

        char_interval = (
            extraction.char_interval
        )

        # ====================================================
        # Grounding 失败
        # ====================================================

        if char_interval is None:

            final_facts.append(
                {
                    "subject_name":
                        original_subject,

                    "original_subject_name":
                        original_subject,

                    "predicate":
                        predicate,

                    "value":
                        value,

                    "unit":
                        unit,

                    "line_start":
                        None,

                    "line_end":
                        None,

                    "char_start":
                        None,

                    "char_end":
                        None,

                    "evidence":
                        evidence,

                    "grounded":
                        False,

                    "subject_resolved":
                        False,

                    "resolution_method":
                        "NONE",
                }
            )

            continue

        # ====================================================
        # char interval -> line
        # ====================================================

        start_pos = (
            char_interval.start_pos
        )

        end_pos = (
            char_interval.end_pos
        )

        start_line = (
            char_position_to_line(
                raw_text,
                start_pos,
            )
        )

        end_char_for_line = max(
            start_pos,
            end_pos - 1,
        )

        end_line = (
            char_position_to_line(
                raw_text,
                end_char_for_line,
            )
        )

        # ====================================================
        # 解析最终 Subject
        # ====================================================

        (
            final_subject,
            subject_resolved,
            resolution_method,
        ) = resolve_fact_subject(
            original_subject=
                original_subject,

            evidence=
                evidence,

            start_line=
                start_line,

            entity_map=
                entity_map,

            entity_blocks=
                entity_blocks,
        )

        # ====================================================
        # Final Fact
        # ====================================================

        final_facts.append(
            {
                "subject_name":
                    final_subject,

                "original_subject_name":
                    original_subject,

                "predicate":
                    predicate,

                "value":
                    value,

                "unit":
                    unit,

                "line_start":
                    start_line,

                "line_end":
                    end_line,

                "char_start":
                    start_pos,

                "char_end":
                    end_pos,

                "evidence":
                    evidence,

                "grounded":
                    True,

                "subject_resolved":
                    subject_resolved,

                "resolution_method":
                    resolution_method,
            }
        )

    return final_facts


# ============================================================
# 7. 打印 Fact
# ============================================================

def print_fact(
    index: int,
    fact: dict,
):
    """
    格式化输出单个 Fact。
    """

    print()
    print("-" * 60)
    print(
        f"Fact #{index}"
    )

    subject_name = fact.get(
        "subject_name",
        "UNKNOWN",
    )

    original_subject = fact.get(
        "original_subject_name",
        subject_name,
    )

    print(
        "Subject：",
        subject_name,
    )

    if (
        original_subject
        != subject_name
    ):
        print(
            "原 Subject：",
            original_subject,
        )

    print(
        "Predicate：",
        fact.get(
            "predicate",
            "",
        ),
    )

    print(
        "Value：",
        fact.get(
            "value",
            "",
        ),
    )

    print(
        "Unit：",
        fact.get(
            "unit",
            "",
        ),
    )

    line_start = fact.get(
        "line_start"
    )

    line_end = fact.get(
        "line_end"
    )

    if line_start is None:

        print(
            "行号：None"
        )

    elif (
        line_end is None
        or line_start == line_end
    ):

        print(
            f"行号："
            f"L{line_start:04d}"
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
        fact.get(
            "evidence",
            "",
        ),
    )

    print(
        "Grounding：",
        (
            "成功"
            if fact.get(
                "grounded"
            )
            else "失败"
        ),
    )

    print(
        "主体解析：",
        (
            "成功"
            if fact.get(
                "subject_resolved"
            )
            else "失败"
        ),
    )

    print(
        "解析方式：",
        fact.get(
            "resolution_method",
            "NONE",
        ),
    )


# ============================================================
# 8. main
# ============================================================

def main():

    # ========================================================
    # 找 Markdown
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

    # 实验阶段仍然只跑第一份
    file_path = files[0]

    print("=" * 60)

    print(
        f"实验文件："
        f"{file_path.name}"
    )

    # ========================================================
    # 读取原文
    # ========================================================

    raw_text = (
        file_path.read_text(
            encoding="utf-8"
        )
    )

    total_lines = len(
        raw_text.splitlines()
    )

    print()

    print(
        f"原始字符数："
        f"{len(raw_text)}"
    )

    print(
        f"原始行数："
        f"{total_lines}"
    )

    # ========================================================
    # Stage 1：Entity Resolution
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 1：Entity Resolution"
    )

    entity_result = (
        resolve_entities(
            raw_text
        )
    )

    entity_map = (
        build_entity_map(
            entity_result
        )
    )

    print()
    print(
        "Entity Map："
    )

    for line, entity in sorted(
        entity_map.items()
    ):

        print(
            f"L{line:04d}"
            f" -> "
            f"{entity}"
        )

    # ========================================================
    # Stage 1.5：Block Ownership
    # ========================================================

    entity_blocks = (
        build_entity_blocks(
            entity_result=
                entity_result,

            total_lines=
                total_lines,
        )
    )

    print()
    print(
        "Entity Blocks："
    )

    for block in entity_blocks:

        print(
            f"L{block.start_line:04d}"
            f" ~ "
            f"L{block.end_line:04d}"
            f" -> "
            f"{block.entity}"
        )

    # ========================================================
    # Stage 2：LangExtract
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 2：LangExtract"
    )

    extraction_result = (
        extract_facts(
            raw_text
        )
    )

    print(
        f"LangExtract Facts："
        f"{len(extraction_result.extractions)}"
    )

    # ========================================================
    # Stage 3：Merge
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 3：Merge"
    )

    final_facts = (
        build_final_facts(
            raw_text=
                raw_text,

            extraction_result=
                extraction_result,

            entity_map=
                entity_map,

            entity_blocks=
                entity_blocks,
        )
    )

    # ========================================================
    # 打印 Facts
    # ========================================================

    for index, fact in enumerate(
        final_facts,
        start=1,
    ):

        print_fact(
            index=index,
            fact=fact,
        )

    # ========================================================
    # 统计
    # ========================================================

    total_facts = len(
        final_facts
    )

    grounded_count = sum(
        1
        for fact in final_facts
        if fact.get(
            "grounded"
        )
    )

    resolved_count = sum(
        1
        for fact in final_facts
        if fact.get(
            "subject_resolved"
        )
    )

    direct_count = sum(
        1
        for fact in final_facts
        if (
            fact.get(
                "resolution_method"
            )
            == "DIRECT"
        )
    )

    entity_resolution_count = sum(
        1
        for fact in final_facts
        if (
            fact.get(
                "resolution_method"
            )
            == "ENTITY_RESOLUTION"
        )
    )

    block_ownership_count = sum(
        1
        for fact in final_facts
        if (
            fact.get(
                "resolution_method"
            )
            == "BLOCK_OWNERSHIP"
        )
    )

    unresolved_count = (
        total_facts
        - resolved_count
    )

    print()
    print("=" * 60)

    print(
        "Pipeline 统计"
    )

    print(
        f"最终 Fact："
        f"{total_facts}"
    )

    print(
        f"Grounding 成功："
        f"{grounded_count}"
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
        f"直接实体："
        f"{direct_count}"
    )

    print(
        f"Entity Resolution："
        f"{entity_resolution_count}"
    )

    print(
        f"Block Ownership："
        f"{block_ownership_count}"
    )

    # ========================================================
    # 保存实验结果
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
        "file":
            file_path.name,

        "model":
            os.getenv(
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

        "entity_blocks": [
            {
                "start_line":
                    block.start_line,

                "end_line":
                    block.end_line,

                "entity":
                    block.entity,
            }
            for block in entity_blocks
        ],

        "facts":
            final_facts,

        "summary": {
            "total_facts":
                total_facts,

            "grounded":
                grounded_count,

            "resolved_subjects":
                resolved_count,

            "unresolved_subjects":
                unresolved_count,

            "direct":
                direct_count,

            "entity_resolution":
                entity_resolution_count,

            "block_ownership":
                block_ownership_count,
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