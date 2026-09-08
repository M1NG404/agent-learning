import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR.parent

PIPELINE_RESULT = (
    EXPERIMENTS_DIR
    / "fact_pipeline"
    / "output"
    / "pipeline_result.json"
)

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
# 1. Raw Fact 分类
# ============================================================

ATTRIBUTE_PREDICATES = {
    "召开董事会时间",
    "董事会召开日期",

    "召开股东会时间",
    "股东会召开日期",

    "召开股东会地点",
    "股东会召开地点",

    "土地使用权面积",
    "房产面积",

    "土地使用权证号",
    "房产证号",
}


RELATION_PREDICATES = {
    "授信额度债务担保金额",

    "授信额度债务",
    "债务类型",
    "责任性质",
    "债务履行截止条件",
}


DROP_PREDICATES = {
    "公章",
    "简称",

    "授权签署",

    "章程中股东会召集程序、表决方式、表决人数等相关规定",

    "股东签字真实性",

    "日期",
}


def classify_fact(
    raw_fact: dict,
) -> str:
    """
    Raw Fact ->
        ATTRIBUTE
        RELATION
        DROP
        UNKNOWN
    """

    predicate = raw_fact.get(
        "predicate",
        "",
    )

    if predicate in ATTRIBUTE_PREDICATES:
        return "ATTRIBUTE"

    if predicate in RELATION_PREDICATES:
        return "RELATION"

    if predicate in DROP_PREDICATES:
        return "DROP"

    return "UNKNOWN"


# ============================================================
# 2. Attribute Predicate 标准化
# ============================================================

PREDICATE_MAPPING = {
    "召开董事会时间":
        "MEETING_DATE",

    "董事会召开日期":
        "MEETING_DATE",

    "召开股东会时间":
        "MEETING_DATE",

    "股东会召开日期":
        "MEETING_DATE",

    "召开股东会地点":
        "MEETING_LOCATION",

    "股东会召开地点":
        "MEETING_LOCATION",

    "土地使用权面积":
        "AREA",

    "房产面积":
        "AREA",

    "土地使用权证号":
        "CERTIFICATE_NO",

    "房产证号":
        "CERTIFICATE_NO",
}


def normalize_predicate(
    raw_predicate: str,
) -> str:
    return PREDICATE_MAPPING.get(
        raw_predicate,
        "UNKNOWN",
    )


def infer_subject_type(
    subject: str,
) -> str:
    """
    第一版简单规则。
    """

    if not subject:
        return "OTHER"

    if subject == "UNKNOWN":
        return "OTHER"

    if subject.endswith("有限公司"):
        return "COMPANY"

    if (
        "国用" in subject
        or "房权证" in subject
    ):
        return "PROPERTY"

    return "OTHER"


def normalize_attribute_fact(
    raw_fact: dict,
) -> dict | None:
    """
    Raw Attribute Fact
        ↓
    Standard AttributeFact
    """

    raw_predicate = raw_fact.get(
        "predicate",
        "",
    )

    predicate = normalize_predicate(
        raw_predicate
    )

    if predicate == "UNKNOWN":
        return None

    subject = raw_fact.get(
        "subject_name",
        "UNKNOWN",
    )

    return {
        "fact_type":
            "ATTRIBUTE",

        "subject_type":
            infer_subject_type(
                subject
            ),

        "subject":
            subject,

        "predicate":
            predicate,

        "value":
            raw_fact.get(
                "value",
                "",
            ),

        "unit":
            raw_fact.get(
                "unit",
                "",
            ),

        "evidence":
            raw_fact.get(
                "evidence",
                "",
            ),

        "line_start":
            raw_fact.get(
                "line_start"
            ),

        "line_end":
            raw_fact.get(
                "line_end"
            ),

        # 实验阶段保留
        "raw_predicate":
            raw_predicate,
    }


# ============================================================
# 3. Relation Raw Facts 分组
# ============================================================

def group_relation_facts(
    raw_facts: list[dict],
) -> list[dict]:
    """
    同一：

        line_start
        +
        evidence

    视为同一个 Relation Candidate。

    例如 L0079：

        债务类型
        责任性质
        债务履行截止条件

    会被合成一个 Candidate。
    """

    groups = {}

    for fact in raw_facts:

        if classify_fact(fact) != "RELATION":
            continue

        line_start = fact.get(
            "line_start"
        )

        evidence = fact.get(
            "evidence",
            "",
        )

        key = (
            line_start,
            evidence,
        )

        if key not in groups:

            groups[key] = {
                "line_start":
                    line_start,

                "evidence":
                    evidence,

                "raw_facts":
                    [],
            }

        groups[key][
            "raw_facts"
        ].append(
            fact
        )

    return list(
        groups.values()
    )


# ============================================================
# 4. Relation Context Expansion
# ============================================================

def find_relation_end_line(
    raw_text: str,
    start_line: int,
) -> int:
    """
    第一版 Relation Context 规则。

    例如：

    L0010 主关系句

    空行

    - A、土地
    - B、土地
    - C、房产

    2、本公司授权……

    自动得到：

    L0010 ~ L0014

    注意：
    这只是当前实验验证过的第一版规则，
    不是最终通用算法。
    """

    lines = raw_text.splitlines()

    end_line = start_line

    # Python index 从 0 开始。
    #
    # start_line 本身已经包含，
    # 所以从下一行开始看。
    for index in range(
        start_line,
        len(lines),
    ):

        line = lines[index].strip()

        current_line = (
            index + 1
        )

        # ----------------------------------------------------
        # 空行允许穿过
        # ----------------------------------------------------

        if not line:
            continue

        # ----------------------------------------------------
        # Markdown List
        # ----------------------------------------------------

        if line.startswith("-"):

            end_line = (
                current_line
            )

            continue

        # ----------------------------------------------------
        # 下一编号段落
        #
        # 例如：
        # 2、
        # 3、
        # ----------------------------------------------------

        if re.match(
            r"^\d+[、.]",
            line,
        ):
            break

        # ----------------------------------------------------
        # 其他文本暂时认为已经离开
        # 当前 Relation Context
        # ----------------------------------------------------

        break

    return end_line


def extract_line_range(
    raw_text: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    L0010 ~ L0014
        ↓
    原始文本 Context
    """

    lines = raw_text.splitlines()

    selected = lines[
        start_line - 1:end_line
    ]

    return "\n".join(
        selected
    ).strip()


def expand_relation_context(
    raw_text: str,
    candidate: dict,
) -> dict:
    """
    Relation Candidate
        ↓
    Relation Context
    """

    start_line = candidate.get(
        "line_start"
    )

    if start_line is None:

        return {
            **candidate,

            "context_start_line":
                None,

            "context_end_line":
                None,

            "context":
                candidate.get(
                    "evidence",
                    "",
                ),
        }

    end_line = find_relation_end_line(
        raw_text=raw_text,
        start_line=start_line,
    )

    context = extract_line_range(
        raw_text=raw_text,
        start_line=start_line,
        end_line=end_line,
    )

    return {
        **candidate,

        "context_start_line":
            start_line,

        "context_end_line":
            end_line,

        "context":
            context,
    }


# ============================================================
# 5. Relation Extractor Prompt
# ============================================================

RELATION_PROMPT = """
你是一个不良资产尽调关系事实抽取器。

输入包含：

1. 已解析主体
2. 已完成 Grounding 和 Context Expansion 的原文

请先判断核心业务关系类型，
然后一次性提取完整 RelationFact。

当前只允许：

GUARANTEE
DEBT


============================================================
GUARANTEE
============================================================

如果某主体为某债务提供：

抵押
质押
保证
连带保证
或者其他担保

输出：

{
  "fact_type": "RELATION",
  "relation_type": "GUARANTEE",
  "participants": {
    "guarantor": "",
    "debtor": "",
    "creditor": ""
  },
  "collaterals": [
    {
      "type": "",
      "certificate_no": "",
      "area": "",
      "unit": ""
    }
  ],
  "attributes": {
    "amount": "",
    "guarantee_type": "",
    "debt_scope": "",
    "end_condition": ""
  }
}


guarantor：
提供担保的主体。

如果原文使用“本公司”，
必须使用输入的“已解析主体”。


debtor：
原始债务人。


creditor：
债权人。


amount：
只能保存纯金额。

例如：

"壹亿元"

不能包含：

授信额度债务
利息
罚息
违约金
应付费用


guarantee_type：
担保方式。

例如：

"抵押、（或质押）担保"

"连带保证责任担保"


debt_scope：
担保覆盖的债务范围。


end_condition：
担保责任终止条件。


collaterals：
Context 中明确列出的抵押物、
质押物或其他担保物。

多个资产必须逐项输出。

collateral.type 当前只允许：

LAND_USE_RIGHT
REAL_ESTATE
OTHER

LAND_USE_RIGHT：
土地使用权。

REAL_ESTATE：
房屋、房产等不动产。

OTHER：
其他资产。

certificate_no：
保持原文权证号。

area：
只输出数字部分。

unit：
例如：

"平方米"

如果没有明确担保物：

collaterals = []


============================================================
DEBT
============================================================

如果某主体作为：

共同债务人
连带债务人
或者类似主体

直接承担债务，

输出：

{
  "fact_type": "RELATION",
  "relation_type": "DEBT",
  "participants": {
    "obligor": "",
    "debtor": "",
    "creditor": ""
  },
  "attributes": {
    "amount": "",
    "debt_type": "",
    "debt_scope": "",
    "liability_type": "",
    "end_condition": ""
  }
}


obligor：
承担债务义务的主体。

原文使用“本公司”时，
使用输入中的“已解析主体”。


debtor：
原始债务人。


creditor：
债权人。


amount：
纯金额。


debt_type：
核心债务类型。

例如：

"授信额度债务"


debt_scope：
债务覆盖范围。


liability_type：
责任性质。

例如：

"负有连带义务共同债务人"


end_condition：
债务责任终止条件。


============================================================
关系判断
============================================================

提供担保：

GUARANTEE


作为共同债务人直接承担债务：

DEBT


注意：

"连带保证责任担保"

属于：

GUARANTEE


"负有连带义务共同债务人"

属于：

DEBT


============================================================
通用规则
============================================================

1. 只能使用原文明确表达的信息。

2. 不允许猜测。

3. 原文没有的信息填写空字符串。

4. 数组没有数据时输出 []。

5. amount 保持原文形式，不允许换算。

6. participant 中不允许最终出现“本公司”。

7. 已解析主体用于替换原文中的“本公司”。

8. 不允许自行创建 Evidence 中不存在的资产。

9. 只输出 JSON。

10. 不要输出解释。
"""


# ============================================================
# 6. Relation Extraction
# ============================================================

def extract_relation(
    context: str,
    resolved_subject: str,
) -> dict:
    """
    Expanded Relation Context
        +
    Resolved Subject
        ↓
    Standard RelationFact
    """

    user_content = f"""
已解析主体：
{resolved_subject}

原文 Relation Context：
{context}
""".strip()

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
                    "role":
                        "system",

                    "content":
                        RELATION_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_content,
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

    return json.loads(
        content
    )


# ============================================================
# 7. Relation Subject
# ============================================================

def get_resolved_subject(
    candidate: dict,
) -> str:
    """
    Relation Candidate 中的 Raw Fact
    已经经过 fact_pipeline 的主体解析。

    所以第一版直接取第一条的 subject_name。
    """

    raw_facts = candidate.get(
        "raw_facts",
        [],
    )

    if not raw_facts:
        return "UNKNOWN"

    subject = (
        raw_facts[0]
        .get(
            "subject_name",
            "UNKNOWN",
        )
    )

    return subject


# ============================================================
# 8. Relation 基础校验
# ============================================================

def validate_relation(
    relation: dict,
    resolved_subject: str,
) -> list[str]:
    """
    不做业务 Gold 校验。

    这里只验证明显的结构问题。
    """

    errors = []

    fact_type = relation.get(
        "fact_type"
    )

    relation_type = relation.get(
        "relation_type"
    )

    participants = relation.get(
        "participants",
        {},
    )

    attributes = relation.get(
        "attributes",
        {},
    )

    if fact_type != "RELATION":

        errors.append(
            "fact_type 不是 RELATION"
        )

    if relation_type not in {
        "GUARANTEE",
        "DEBT",
    }:

        errors.append(
            "未知 relation_type："
            f"{relation_type}"
        )

        return errors

    # --------------------------------------------------------
    # GUARANTEE
    # --------------------------------------------------------

    if relation_type == "GUARANTEE":

        guarantor = participants.get(
            "guarantor",
            "",
        )

        if guarantor != resolved_subject:

            errors.append(
                "guarantor 与 resolved_subject 不一致："
                f"{guarantor}"
            )

        if not attributes.get(
            "guarantee_type"
        ):

            errors.append(
                "GUARANTEE 缺少 guarantee_type"
            )

        collaterals = relation.get(
            "collaterals",
            [],
        )

        if not isinstance(
            collaterals,
            list,
        ):

            errors.append(
                "collaterals 不是数组"
            )

    # --------------------------------------------------------
    # DEBT
    # --------------------------------------------------------

    if relation_type == "DEBT":

        obligor = participants.get(
            "obligor",
            "",
        )

        if obligor != resolved_subject:

            errors.append(
                "obligor 与 resolved_subject 不一致："
                f"{obligor}"
            )

        if not attributes.get(
            "liability_type"
        ):

            errors.append(
                "DEBT 缺少 liability_type"
            )

    # --------------------------------------------------------
    # 通用
    # --------------------------------------------------------

    if not participants.get(
        "debtor"
    ):

        errors.append(
            "缺少 debtor"
        )

    if not participants.get(
        "creditor"
    ):

        errors.append(
            "缺少 creditor"
        )

    amount = attributes.get(
        "amount",
        "",
    )

    if not amount:

        errors.append(
            "缺少 amount"
        )

    return errors


# ============================================================
# 9. main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Standard Fact Pipeline"
    )

    # ========================================================
    # 读取前一阶段 Pipeline Result
    # ========================================================

    if not PIPELINE_RESULT.exists():

        print(
            "找不到："
            f"{PIPELINE_RESULT}"
        )

        return

    pipeline_data = json.loads(
        PIPELINE_RESULT.read_text(
            encoding="utf-8"
        )
    )

    raw_facts = pipeline_data.get(
        "facts",
        [],
    )

    source_file = pipeline_data.get(
        "file"
    )

    if not source_file:

        print(
            "pipeline_result.json "
            "缺少 file"
        )

        return

    source_path = (
        INPUT_DIR
        / source_file
    )

    if not source_path.exists():

        print(
            "找不到原始 Markdown："
            f"{source_path}"
        )

        return

    raw_text = source_path.read_text(
        encoding="utf-8"
    )

    print(
        f"Source：{source_file}"
    )

    print(
        f"Raw Facts：{len(raw_facts)}"
    )

    # ========================================================
    # Classification
    # ========================================================

    attribute_raw = []
    relation_raw = []
    drop_raw = []
    unknown_raw = []

    for fact in raw_facts:

        category = classify_fact(
            fact
        )

        if category == "ATTRIBUTE":

            attribute_raw.append(
                fact
            )

        elif category == "RELATION":

            relation_raw.append(
                fact
            )

        elif category == "DROP":

            drop_raw.append(
                fact
            )

        else:

            unknown_raw.append(
                fact
            )

    print()
    print("=" * 60)
    print(
        "Classification"
    )

    print(
        f"ATTRIBUTE："
        f"{len(attribute_raw)}"
    )

    print(
        f"RELATION Raw："
        f"{len(relation_raw)}"
    )

    print(
        f"DROP："
        f"{len(drop_raw)}"
    )

    print(
        f"UNKNOWN："
        f"{len(unknown_raw)}"
    )

    # ========================================================
    # ATTRIBUTE
    # ========================================================

    attribute_facts = []

    for raw_fact in attribute_raw:

        normalized = (
            normalize_attribute_fact(
                raw_fact
            )
        )

        if normalized:

            attribute_facts.append(
                normalized
            )

    # ========================================================
    # RELATION Group
    # ========================================================

    candidates = group_relation_facts(
        raw_facts
    )

    print(
        f"Relation Candidates："
        f"{len(candidates)}"
    )

    # ========================================================
    # RELATION Context Expansion + Extraction
    # ========================================================

    relation_facts = []

    relation_errors = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        expanded = (
            expand_relation_context(
                raw_text=
                    raw_text,

                candidate=
                    candidate,
            )
        )

        resolved_subject = (
            get_resolved_subject(
                expanded
            )
        )

        print()
        print("-" * 60)

        print(
            f"Relation Candidate #{index}"
        )

        print(
            "Resolved Subject：",
            resolved_subject,
        )

        print(
            "Context："
            f"L{expanded['context_start_line']:04d}"
            f" ~ "
            f"L{expanded['context_end_line']:04d}"
        )

        print()

        print(
            expanded[
                "context"
            ]
        )

        relation = extract_relation(
            context=
                expanded["context"],

            resolved_subject=
                resolved_subject,
        )

        # 增加来源信息
        relation[
            "source"
        ] = {
            "file":
                source_file,

            "line_start":
                expanded[
                    "context_start_line"
                ],

            "line_end":
                expanded[
                    "context_end_line"
                ],

            "evidence":
                expanded[
                    "context"
                ],
        }

        errors = validate_relation(
            relation=
                relation,

            resolved_subject=
                resolved_subject,
        )

        print()
        print(
            "RelationFact："
        )

        print(
            json.dumps(
                relation,
                ensure_ascii=False,
                indent=2,
            )
        )

        if errors:

            print(
                "Validation：失败"
            )

            for error in errors:

                print(
                    f"- {error}"
                )

            relation_errors.append(
                {
                    "candidate":
                        index,

                    "errors":
                        errors,

                    "relation":
                        relation,
                }
            )

        else:

            print(
                "Validation：通过"
            )

        relation_facts.append(
            relation
        )

    # ========================================================
    # Standard Facts
    # ========================================================

    standard_facts = (
        attribute_facts
        +
        relation_facts
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 60)

    print(
        "Standard Fact Pipeline 统计"
    )

    print(
        f"Raw Facts："
        f"{len(raw_facts)}"
    )

    print(
        f"Attribute Facts："
        f"{len(attribute_facts)}"
    )

    print(
        f"Relation Facts："
        f"{len(relation_facts)}"
    )

    print(
        f"Dropped Raw Facts："
        f"{len(drop_raw)}"
    )

    print(
        f"Unknown Raw Facts："
        f"{len(unknown_raw)}"
    )

    print(
        f"Relation Validation Errors："
        f"{len(relation_errors)}"
    )

    print(
        f"最终 Standard Facts："
        f"{len(standard_facts)}"
    )

    # ========================================================
    # 保存
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "standard_facts.json"
    )

    output = {
        "source_file":
            source_file,

        "attribute_facts":
            attribute_facts,

        "relation_facts":
            relation_facts,

        "dropped_raw_facts":
            drop_raw,

        "unknown_raw_facts":
            unknown_raw,

        "relation_validation_errors":
            relation_errors,

        "summary": {
            "raw_facts":
                len(raw_facts),

            "attribute_facts":
                len(attribute_facts),

            "relation_facts":
                len(relation_facts),

            "dropped":
                len(drop_raw),

            "unknown":
                len(unknown_raw),

            "relation_validation_errors":
                len(
                    relation_errors
                ),

            "standard_facts":
                len(
                    standard_facts
                ),
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
        f"结果已保存："
        f"{output_file}"
    )


if __name__ == "__main__":
    main()