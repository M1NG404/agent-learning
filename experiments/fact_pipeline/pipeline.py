import json
import os
from collections import Counter
from pathlib import Path
import re
import langextract as lx
from langextract.factory import ModelConfig
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR.parent

INPUT_DIR = (
    EXPERIMENTS_DIR
    / "fact_extraction"
    / "input"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
)


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# ============================================================
# 1. 找当前实验 Markdown
# ============================================================

def find_markdown_file() -> Path:
    """
    当前实验目录暂时只处理第一份 Markdown。
    """

    files = list(
        INPUT_DIR.glob("*.md")
    )

    if not files:
        raise FileNotFoundError(
            f"没有找到 Markdown：{INPUT_DIR}"
        )

    return files[0]


# ============================================================
# 2. 行号工具
# ============================================================

def add_line_numbers(
    text: str,
) -> str:
    """
    Entity Resolution 使用带行号文本。

    例如：

    原文：

    江阴利泰...
    本公司...

    转为：

    L0001 江阴利泰...
    L0002 本公司...

    注意：

    LangExtract Grounding 仍然使用原始文本，
    不能使用加行号后的文本。
    """

    lines = text.splitlines()

    numbered_lines = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        numbered_lines.append(
            f"L{index:04d} {line}"
        )

    return "\n".join(
        numbered_lines
    )


def char_position_to_line(
    text: str,
    char_pos: int,
) -> int:
    """
    LangExtract 给的是 char position。

    例如：

    char_start = 1234

    转换成：

    L0079
    """

    return (
        text[:char_pos]
        .count("\n")
        + 1
    )


# ============================================================
# 3. Entity Resolution Prompt
# ============================================================

ENTITY_RESOLUTION_PROMPT = """
你负责做尽调文档中的主体指代消解。

你的任务不是抽取业务事实。

你的唯一任务是识别：

“本公司”

等主体指代具体对应哪个明确实体。

输入文本带有行号，例如：

L0001 ...
L0002 ...

输出格式：

{
  "mentions": [
    {
      "mention": "本公司",
      "mention_line": 10,
      "resolved_entity": "江阴某某有限公司",
      "definition_line": 7
    }
  ]
}

字段说明：

mention：
原文中的指代表达。

mention_line：
该指代表达所在行。

resolved_entity：
该指代对应的完整实体名称。

definition_line：
该实体在当前文档业务块开始位置或明确主体定义位置。

规则：

1. 只能依据文档明确内容解析。
2. 不允许猜测。
3. 如果无法安全判断，不要输出该 mention。
4. resolved_entity 必须使用完整实体名称。
5. definition_line 必须对应当前主体区域的起始定义位置。
6. mention_line 必须对应该次“本公司”出现的位置。
7. 同一个实体可以存在多个 mention。
8. 只输出 JSON。
"""


# ============================================================
# 4. Entity Resolution
# ============================================================

def resolve_entities(
    numbered_text: str,
) -> dict:
    """
    返回：

    {
        "mentions": [...],
        "definitions": [...]
    }

    definitions 从 mentions 的 definition_line
    自动整理得到。
    """

    response = (
        client
        .chat
        .completions
        .create(
            model=os.getenv(
                "FACT_EXTRACT_MODEL",
                "qwen-plus",
            ),
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content":
                        ENTITY_RESOLUTION_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        numbered_text,
                },
            ],
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    data = json.loads(
        content
    )

    mentions = data.get(
        "mentions",
        [],
    )

    # ========================================================
    # 从 mention 自动整理主体定义
    # ========================================================

    definitions_map = {}

    for mention in mentions:

        entity = mention.get(
            "resolved_entity"
        )

        definition_line = (
            mention.get(
                "definition_line"
            )
        )

        if (
            not entity
            or definition_line is None
        ):
            continue

        if (
            entity
            not in definitions_map
        ):
            definitions_map[
                entity
            ] = definition_line

        else:
            definitions_map[
                entity
            ] = min(
                definitions_map[
                    entity
                ],
                definition_line,
            )

    definitions = [
        {
            "entity":
                entity,

            "definition_line":
                definition_line,
        }
        for (
            entity,
            definition_line
        )
        in definitions_map.items()
    ]

    definitions.sort(
        key=lambda item:
            item["definition_line"]
    )

    return {
        "mentions":
            mentions,

        "definitions":
            definitions,
    }


# ============================================================
# 5. Block Ownership
# ============================================================

def build_entity_blocks(
    definitions: list[dict],
    total_lines: int,
) -> list[dict]:
    """
    根据主体 definition line 建立业务块。

    当前文档例如：

    L0007 ~ L0038
        → 江阴利泰装饰材料有限公司

    L0039 ~ L0075
        → 江苏海达科技集团有限公司

    L0076 ~ L0112
        → 江阴海达特种人革有限公司

    L0113 ~ L0143
        → 江阴海达彩涂有限公司

    注意：

    Block Ownership 不是最终事实。

    它只是 LOW confidence fallback。
    """

    if not definitions:
        return []

    sorted_definitions = sorted(
        definitions,
        key=lambda item:
            item["definition_line"],
    )

    blocks = []

    for index, definition in enumerate(
        sorted_definitions
    ):

        start_line = definition[
            "definition_line"
        ]

        owner = definition[
            "entity"
        ]

        if (
            index + 1
            < len(
                sorted_definitions
            )
        ):

            next_definition = (
                sorted_definitions[
                    index + 1
                ]
            )

            end_line = (
                next_definition[
                    "definition_line"
                ]
                - 1
            )

        else:
            end_line = (
                total_lines
            )

        blocks.append(
            {
                "start_line":
                    start_line,

                "end_line":
                    end_line,

                "owner":
                    owner,
            }
        )

    return blocks


def find_block_owner(
    line_number: int,
    blocks: list[dict],
) -> str | None:
    """
    根据行号找到所在 block。

    必须只有一个匹配 block，
    否则返回 None。
    """

    matches = []

    for block in blocks:

        if (
            block[
                "start_line"
            ]
            <= line_number
            <= block[
                "end_line"
            ]
        ):
            matches.append(
                block[
                    "owner"
                ]
            )

    unique_matches = list(
        dict.fromkeys(
            matches
        )
    )

    if (
        len(
            unique_matches
        )
        != 1
    ):
        return None

    return unique_matches[0]


# ============================================================
# 6. 根据行号解析“本公司”
# ============================================================

def resolve_reference_by_line(
    line_number: int,
    mentions: list[dict],
) -> str | None:
    """
    Evidence 出现“本公司”时使用。

    优先：

    当前 Fact 行
    是否存在 Entity Resolution exact match。

    如果没有 exact match：

    暂时寻找最近的不晚于当前行的 mention。

    但如果最近位置对应多个实体，
    返回 None。
    """

    # ========================================================
    # Exact Line
    # ========================================================

    exact_entities = []

    for mention in mentions:

        mention_line = (
            mention.get(
                "mention_line"
            )
        )

        entity = (
            mention.get(
                "resolved_entity"
            )
        )

        if (
            mention_line
            == line_number
            and entity
        ):
            exact_entities.append(
                entity
            )

    exact_entities = list(
        dict.fromkeys(
            exact_entities
        )
    )

    if (
        len(
            exact_entities
        )
        == 1
    ):
        return exact_entities[0]

    if (
        len(
            exact_entities
        )
        > 1
    ):
        return None

    # ========================================================
    # 最近的前序 mention
    # ========================================================

    candidates = []

    for mention in mentions:

        mention_line = (
            mention.get(
                "mention_line"
            )
        )

        entity = (
            mention.get(
                "resolved_entity"
            )
        )

        if (
            mention_line is not None
            and entity
            and mention_line
            <= line_number
        ):
            candidates.append(
                mention
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item["mention_line"],
        reverse=True,
    )

    nearest_line = (
        candidates[0][
            "mention_line"
        ]
    )

    nearest_entities = {
        item[
            "resolved_entity"
        ]
        for item in candidates
        if (
            item[
                "mention_line"
            ]
            == nearest_line
        )
    }

    if (
        len(
            nearest_entities
        )
        != 1
    ):
        return None

    return next(
        iter(
            nearest_entities
        )
    )


# ============================================================
# 7. LangExtract Prompt
# ============================================================

FACT_EXTRACTION_PROMPT = """
从尽调文档中提取明确存在的事实。

每条 extraction 表示一个 Raw Fact。

attributes 必须包含：

subject
predicate
value
unit

规则：

1. extraction_text 必须是原文中连续、逐字存在的文本。

2. 不允许改写 extraction_text。

3. 不允许生成原文不存在的 evidence。

4. subject 表示该事实的候选主体。

5. 如果原文使用“本公司”，可以输出：

subject = "本公司"

6. 如果原文没有明确写主体，不允许为了完整而虚构主体。

7. predicate 使用尽可能简洁、明确的中文业务字段名称。

8. value 必须保留原文表达。

例如：

贰仟万元

不能自行转换成：

2000万元

9. unit 如果原文有明确单位则提取，否则为空字符串。

10. 一个 Raw Fact 只表达一个核心事实。

11. 不允许根据常识推断原文没有表达的信息。

12. 原文只有孤立日期、孤立数字时，不要自动赋予其不存在的业务语义。
"""


# ============================================================
# 8. LangExtract Raw Fact Extraction
# ============================================================

def extract_raw_facts(
    raw_text: str,
) -> list[dict]:
    """
    Markdown
        ↓
    LangExtract
        ↓
    Grounded Raw Facts

    关键：

    显式指定：

        provider="openai"

    因为 qwen-plus 是 OpenAI-compatible 模型，
    但不是 GPT model id。

    如果不指定 provider，
    LangExtract 可能自动选择 Ollama。
    """

    examples = [
        lx.data.ExampleData(
            text=(
                "江阴某有限公司于2018年12月19日"
                "召开股东会。"
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class=
                        "fact",

                    extraction_text=(
                        "江阴某有限公司于2018年12月19日"
                        "召开股东会"
                    ),

                    attributes={
                        "subject":
                            "江阴某有限公司",

                        "predicate":
                            "召开股东会时间",

                        "value":
                            "2018年12月19日",

                        "unit":
                            "",
                    },
                )
            ],
        )
    ]

    # ========================================================
    # 关键修复
    #
    # qwen-plus
    #    ↓
    # 显式指定 OpenAILanguageModel
    #    ↓
    # OPENAI_API_BASE_URL
    #    ↓
    # OpenAI-compatible Qwen API
    # ========================================================

    model_config = ModelConfig(
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

            "temperature":
                0,
        },
    )

    result = lx.extract(
        text_or_documents=
            raw_text,

        prompt_description=
            FACT_EXTRACTION_PROMPT,

        examples=
            examples,

        config=
            model_config,
    )

    facts = []

    for extraction in (
        result.extractions
    ):

        attributes = (
            extraction.attributes
            or {}
        )

        evidence = (
            extraction.extraction_text
            or ""
        )

        interval = (
            extraction.char_interval
        )

        char_start = None
        char_end = None

        if interval is not None:

            char_start = (
                interval.start_pos
            )

            char_end = (
                interval.end_pos
            )

        # ====================================================
        # Grounding Validation
        # ====================================================

        grounded = False

        if (
            char_start is not None
            and char_end is not None
        ):

            source_text = (
                raw_text[
                    char_start:
                    char_end
                ]
            )

            grounded = (
                source_text
                == evidence
            )

        # ====================================================
        # char -> line
        # ====================================================

        line_start = None
        line_end = None

        if char_start is not None:

            line_start = (
                char_position_to_line(
                    text=
                        raw_text,

                    char_pos=
                        char_start,
                )
            )

        if (
            char_end is not None
            and char_end > 0
        ):

            line_end = (
                char_position_to_line(
                    text=
                        raw_text,

                    char_pos=
                        char_end - 1,
                )
            )

        facts.append(
            {
                "original_subject_name":
                    attributes.get(
                        "subject",
                        "",
                    ),

                "predicate":
                    attributes.get(
                        "predicate",
                        "",
                    ),

                "value":
                    attributes.get(
                        "value",
                        "",
                    ),

                "unit":
                    attributes.get(
                        "unit",
                        "",
                    ),

                "line_start":
                    line_start,

                "line_end":
                    line_end,

                "char_start":
                    char_start,

                "char_end":
                    char_end,

                "evidence":
                    evidence,

                "grounded":
                    grounded,
            }
        )

    return facts


# ============================================================
# 9. Conservative Subject Resolver
# ============================================================
def is_weak_evidence(
    evidence: str,
    predicate: str,
    original_subject: str,
) -> bool:
    """
    判断 Evidence 是否弱到不应该使用 Block Ownership。

    当前第一版只拦截最明显的危险情况：
    - subject 为空
    - predicate 为空
    - evidence 只是一个孤立日期 / 数字
    """

    evidence = (evidence or "").strip()
    predicate = (predicate or "").strip()
    original_subject = (
        original_subject or ""
    ).strip()

    # 本身连 predicate 都没有
    if not predicate:
        return True

    # 有明确 subject 时，不属于这里
    if original_subject:
        return False

    # 孤立日期
    import re

    date_pattern = (
        r"^\d{4}年\d{1,2}月\d{1,2}日$"
    )

    if re.match(
        date_pattern,
        evidence,
    ):
        return True

    # 孤立数字
    number_pattern = (
        r"^[\d,.]+$"
    )

    if re.match(
        number_pattern,
        evidence,
    ):
        return True

    return False


def resolve_subject_conservatively(
    fact: dict,
    mentions: list[dict],
    blocks: list[dict],
) -> dict:
    """
    保守主体解析。

    新优先级：

    1. 明确 subject 出现在 evidence
       → DIRECT / MEDIUM

    2. original_subject == 本公司
       → ENTITY_RESOLUTION / HIGH

    3. 没有明确主体时
       → 判断 Evidence 是否允许 Block fallback

    4. Evidence 太弱
       → UNRESOLVED

    5. 最后才允许 Block Ownership
       → LOW
    """

    evidence = (
        fact.get(
            "evidence",
            ""
        )
        or ""
    )

    original_subject = (
        fact.get(
            "original_subject_name",
            ""
        )
        or ""
    )

    predicate = (
        fact.get(
            "predicate",
            ""
        )
        or ""
    )

    line_start = (
        fact.get(
            "line_start"
        )
    )

    invalid_subjects = {
        "",
        "本公司",
        "UNKNOWN",
        "未知",
        "None",
    }

    # ========================================================
    # 1. 明确主体优先
    # ========================================================

    if (
        original_subject
        not in invalid_subjects
        and original_subject
        in evidence
    ):

        return {
            "subject_name":
                original_subject,

            "subject_status":
                "RESOLVED",

            "subject_confidence":
                "MEDIUM",

            "resolution_method":
                "DIRECT",

            "subject_reason":
                (
                    "The explicit candidate subject "
                    "appears in evidence. Semantic "
                    "role validation has not yet "
                    "been performed."
                ),
        }

    # ========================================================
    # 2. 只有 original_subject 本身就是“本公司”
    #    才执行 Entity Resolution
    # ========================================================

    if original_subject == "本公司":

        if line_start is not None:

            resolved_entity = (
                resolve_reference_by_line(
                    line_number=
                        line_start,

                    mentions=
                        mentions,
                )
            )

            if resolved_entity:

                return {
                    "subject_name":
                        resolved_entity,

                    "subject_status":
                        "RESOLVED",

                    "subject_confidence":
                        "HIGH",

                    "resolution_method":
                        "ENTITY_RESOLUTION",

                    "subject_reason":
                        (
                            "The candidate subject is "
                            "'本公司' and Entity Resolution "
                            "resolved the reference."
                        ),
                }

        return {
            "subject_name":
                None,

            "subject_status":
                "UNRESOLVED",

            "subject_confidence":
                "LOW",

            "resolution_method":
                "NONE",

            "subject_reason":
                (
                    "The candidate subject is "
                    "'本公司', but the reference "
                    "could not be resolved safely."
                ),
        }

    # ========================================================
    # 3. Evidence 太弱时，禁止 Block Ownership
    # ========================================================

    if is_weak_evidence(
        evidence=
            evidence,

        predicate=
            predicate,

        original_subject=
            original_subject,
    ):

        return {
            "subject_name":
                None,

            "subject_status":
                "UNRESOLVED",

            "subject_confidence":
                "LOW",

            "resolution_method":
                "NONE",

            "subject_reason":
                (
                    "Evidence is too weak to support "
                    "subject inference. Block ownership "
                    "fallback was intentionally rejected."
                ),
        }

    # ========================================================
    # 4. Block Ownership
    # ========================================================

    if line_start is not None:

        block_owner = (
            find_block_owner(
                line_number=
                    line_start,

                blocks=
                    blocks,
            )
        )

        if block_owner:

            return {
                "subject_name":
                    block_owner,

                "subject_status":
                    "RESOLVED",

                "subject_confidence":
                    "LOW",

                "resolution_method":
                    "BLOCK_OWNERSHIP",

                "subject_reason":
                    (
                        "No explicit validated subject "
                        "was found. Subject was inferred "
                        "from document block ownership."
                    ),
            }

    # ========================================================
    # 5. 最终无法判断
    # ========================================================

    return {
        "subject_name":
            None,

        "subject_status":
            "UNRESOLVED",

        "subject_confidence":
            "LOW",

        "resolution_method":
            "NONE",

        "subject_reason":
            (
                "Insufficient evidence to determine "
                "the subject safely."
            ),
    }

# ============================================================
# 10. main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Fact Pipeline Experiment"
    )

    # ========================================================
    # Stage 0
    #
    # 读取 Markdown
    # ========================================================

    source_path = (
        find_markdown_file()
    )

    source_file = (
        source_path.name
    )

    raw_text = (
        source_path.read_text(
            encoding="utf-8"
        )
    )

    numbered_text = (
        add_line_numbers(
            raw_text
        )
    )

    total_lines = len(
        raw_text.splitlines()
    )

    print(
        f"Source："
        f"{source_file}"
    )

    print(
        f"Lines："
        f"{total_lines}"
    )

    # ========================================================
    # Stage 1
    #
    # Entity Resolution
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 1：Entity Resolution"
    )

    entity_result = (
        resolve_entities(
            numbered_text
        )
    )

    mentions = (
        entity_result[
            "mentions"
        ]
    )

    definitions = (
        entity_result[
            "definitions"
        ]
    )

    print(
        f"Mentions："
        f"{len(mentions)}"
    )

    print(
        f"Entity Definitions："
        f"{len(definitions)}"
    )

    for definition in definitions:

        print(
            f"L"
            f"{definition['definition_line']:04d}"
            " → "
            f"{definition['entity']}"
        )

    # ========================================================
    # Stage 1.5
    #
    # Block Ownership
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 1.5："
        "Block Ownership"
    )

    blocks = (
        build_entity_blocks(
            definitions=
                definitions,

            total_lines=
                total_lines,
        )
    )

    for block in blocks:

        print(
            f"L"
            f"{block['start_line']:04d}"
            " ~ "
            f"L"
            f"{block['end_line']:04d}"
            " → "
            f"{block['owner']}"
        )

    # ========================================================
    # Stage 2
    #
    # Grounded Raw Fact Extraction
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 2："
        "Grounded Fact Extraction"
    )

    raw_facts = (
        extract_raw_facts(
            raw_text
        )
    )

    print(
        f"Raw Facts："
        f"{len(raw_facts)}"
    )

    # ========================================================
    # Stage 3
    #
    # Conservative Subject Resolution
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Stage 3："
        "Conservative Subject Resolution"
    )

    final_facts = []

    for fact in raw_facts:

        subject_result = (
            resolve_subject_conservatively(
                fact=
                    fact,

                mentions=
                    mentions,

                blocks=
                    blocks,
            )
        )

        final_fact = {
            **fact,
            **subject_result,
        }

        # ====================================================
        # 为了暂时兼容之前实验
        # ====================================================

        final_fact[
            "subject_resolved"
        ] = (
            final_fact[
                "subject_status"
            ]
            == "RESOLVED"
        )

        final_facts.append(
            final_fact
        )

    # ========================================================
    # 输出 Fact
    # ========================================================

    for index, fact in enumerate(
        final_facts,
        start=1,
    ):

        print()
        print("-" * 60)

        print(
            f"Fact #{index}"
        )

        line_start = (
            fact.get(
                "line_start"
            )
        )

        if (
            line_start
            is not None
        ):

            print(
                f"Line："
                f"L{line_start:04d}"
            )

        print(
            "Original Subject：",
            fact.get(
                "original_subject_name"
            ),
        )

        print(
            "Resolved Subject：",
            fact.get(
                "subject_name"
            ),
        )

        print(
            "Predicate：",
            fact.get(
                "predicate"
            ),
        )

        print(
            "Value：",
            fact.get(
                "value"
            ),
        )

        print(
            "Unit：",
            fact.get(
                "unit"
            ),
        )

        print(
            "Evidence：",
            fact.get(
                "evidence"
            ),
        )

        print(
            "Grounded：",
            fact.get(
                "grounded"
            ),
        )

        print(
            "Subject Status：",
            fact.get(
                "subject_status"
            ),
        )

        print(
            "Confidence：",
            fact.get(
                "subject_confidence"
            ),
        )

        print(
            "Method：",
            fact.get(
                "resolution_method"
            ),
        )

        print(
            "Reason：",
            fact.get(
                "subject_reason"
            ),
        )

    # ========================================================
    # 统计
    # ========================================================

    grounded_count = sum(
        1
        for fact
        in final_facts
        if fact.get(
            "grounded"
        )
    )

    resolved_count = sum(
        1
        for fact
        in final_facts
        if (
            fact.get(
                "subject_status"
            )
            == "RESOLVED"
        )
    )

    unresolved_count = sum(
        1
        for fact
        in final_facts
        if (
            fact.get(
                "subject_status"
            )
            == "UNRESOLVED"
        )
    )

    confidence_counter = Counter(
        fact.get(
            "subject_confidence",
            "UNKNOWN",
        )
        for fact
        in final_facts
        if (
            fact.get(
                "subject_status"
            )
            == "RESOLVED"
        )
    )

    method_counter = Counter(
        fact.get(
            "resolution_method",
            "NONE",
        )
        for fact
        in final_facts
    )

    # ========================================================
    # 打印统计
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Pipeline Statistics"
    )

    print(
        f"Raw Facts："
        f"{len(final_facts)}"
    )

    print(
        f"Grounding："
        f"{grounded_count}"
        "/"
        f"{len(final_facts)}"
    )

    print()

    print(
        "Subject Resolution："
    )

    print(
        f"RESOLVED："
        f"{resolved_count}"
    )

    print(
        f"UNRESOLVED："
        f"{unresolved_count}"
    )

    print()

    print(
        "Confidence："
    )

    print(
        "HIGH：",
        confidence_counter.get(
            "HIGH",
            0,
        ),
    )

    print(
        "MEDIUM：",
        confidence_counter.get(
            "MEDIUM",
            0,
        ),
    )

    print(
        "LOW：",
        confidence_counter.get(
            "LOW",
            0,
        ),
    )

    print()

    print(
        "Resolution Method："
    )

    print(
        "ENTITY_RESOLUTION：",
        method_counter.get(
            "ENTITY_RESOLUTION",
            0,
        ),
    )

    print(
        "DIRECT：",
        method_counter.get(
            "DIRECT",
            0,
        ),
    )

    print(
        "BLOCK_OWNERSHIP：",
        method_counter.get(
            "BLOCK_OWNERSHIP",
            0,
        ),
    )

    print(
        "NONE：",
        method_counter.get(
            "NONE",
            0,
        ),
    )

    # ========================================================
    # LOW Confidence Facts
    # ========================================================

    low_confidence_facts = [
        fact
        for fact
        in final_facts
        if (
            fact.get(
                "subject_confidence"
            )
            == "LOW"
            and fact.get(
                "subject_status"
            )
            == "RESOLVED"
        )
    ]

    unresolved_facts = [
        fact
        for fact
        in final_facts
        if (
            fact.get(
                "subject_status"
            )
            == "UNRESOLVED"
        )
    ]

    print()
    print("=" * 60)

    print(
        "LOW Confidence Facts"
    )

    if not low_confidence_facts:

        print(
            "无"
        )

    else:

        for fact in (
            low_confidence_facts
        ):

            print(
                f"L"
                f"{fact.get('line_start', 0):04d}"
                " | "
                f"{fact.get('subject_name')}"
                " | "
                f"{fact.get('predicate')}"
                " | "
                f"{fact.get('value')}"
            )

    print()
    print("=" * 60)

    print(
        "UNRESOLVED Facts"
    )

    if not unresolved_facts:

        print(
            "无"
        )

    else:

        for fact in (
            unresolved_facts
        ):

            line_start = (
                fact.get(
                    "line_start"
                )
            )

            if line_start is None:
                line_text = "UNKNOWN"
            else:
                line_text = (
                    f"L"
                    f"{line_start:04d}"
                )

            print(
                f"{line_text}"
                " | "
                f"{fact.get('predicate')}"
                " | "
                f"{fact.get('value')}"
                " | "
                f"{fact.get('evidence')}"
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

    output = {
        "file":
            source_file,

        "entity_resolution": {
            "mentions":
                mentions,

            "definitions":
                definitions,
        },

        "blocks":
            blocks,

        "facts":
            final_facts,

        "summary": {
            "raw_facts":
                len(
                    final_facts
                ),

            "grounded":
                grounded_count,

            "resolved":
                resolved_count,

            "unresolved":
                unresolved_count,

            "confidence": {
                "HIGH":
                    confidence_counter.get(
                        "HIGH",
                        0,
                    ),

                "MEDIUM":
                    confidence_counter.get(
                        "MEDIUM",
                        0,
                    ),

                "LOW":
                    confidence_counter.get(
                        "LOW",
                        0,
                    ),
            },

            "resolution_method": {
                "ENTITY_RESOLUTION":
                    method_counter.get(
                        "ENTITY_RESOLUTION",
                        0,
                    ),

                "DIRECT":
                    method_counter.get(
                        "DIRECT",
                        0,
                    ),

                "BLOCK_OWNERSHIP":
                    method_counter.get(
                        "BLOCK_OWNERSHIP",
                        0,
                    ),

                "NONE":
                    method_counter.get(
                        "NONE",
                        0,
                    ),
            },
        },
    }

    output_file.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"结果已保存："
        f"{output_file}"
    )


if __name__ == "__main__":
    main()