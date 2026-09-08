import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR.parent

PIPELINE_RESULT_FILE = (
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


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# ============================================================
# Verification Result
# ============================================================

@dataclass
class VerificationResult:
    supported: bool
    confidence: str
    reason: str


# ============================================================
# Verification Prompt
# ============================================================

VERIFICATION_PROMPT = """
你是一个尽调事实验证器。

你的任务不是重新抽取事实。

你的任务是判断：

给定的 Fact 是否真的被 Evidence 和 Local Context 支持。


输入包括：

1. Fact
2. Evidence
3. Local Context


你必须验证：

- subject 是否被原文支持
- predicate 是否真的被原文表达
- value 是否与原文一致
- subject 是否承担了正确的语义角色


特别注意：

Grounding 只表示 Evidence 来源于原文，
不能说明 Fact 的语义一定正确。

Evidence 有时只是完整事实的一部分。

这种情况下允许使用 Local Context
补足完整语义。


例如：

Fact:

subject = A公司
predicate = MEETING_DATE
value = 2018年12月19日

Evidence:

A公司于2018年12月19日

Local Context:

A公司于2018年12月19日
在公司会议室召开股东会。

应该判断：

supported = true


因为 Local Context 明确说明：

2018年12月19日
是该公司召开股东会的日期。


============================================================
反例 1：Predicate 错误
============================================================

Fact:

subject = A公司
predicate = CONTRACT_SIGN_DATE
value = 2018年12月19日

Local Context:

A公司于2018年12月19日召开股东会。

应该：

supported = false

因为原文表达的是股东会召开日期，
不是合同签署日期。


============================================================
反例 2：Subject Role 错误
============================================================

Fact:

subject = A公司
predicate = PROVIDE_GUARANTEE
value = 连带保证

Local Context:

A公司的债务由B公司提供连带保证。

应该：

supported = false

因为提供担保的是 B公司。


============================================================
反例 3：Fact 本身缺失核心语义
============================================================

Fact:

subject = null
predicate = ""
value = 2018年12月19日

Evidence:

2018年12月19日

即使 Local Context 中存在其他内容，
如果无法证明这个 Fact 的明确 predicate，
也应：

supported = false


============================================================
规则
============================================================

1. 只能根据 Evidence 和 Local Context 判断。

2. 不允许用常识补充原文没有的信息。

3. 不能因为 subject/value 字符串出现就直接判 true。

4. 必须验证 predicate 语义。

5. 必须验证 subject 的语义角色。

6. Evidence 本身不足时，可以使用 Local Context。

7. predicate 为空时，原则上不能形成一个可验证的业务事实。

8. subject 无法确定时，应谨慎判断。

9. confidence 只能：

HIGH
MEDIUM
LOW

10. 只输出 JSON。


输出：

{
  "supported": true,
  "confidence": "HIGH",
  "reason": "..."
}
"""


# ============================================================
# 调用 Verifier
# ============================================================

def verify_fact(
    fact: dict,
    evidence: str,
    local_context: str,
) -> VerificationResult:

    user_content = f"""
Fact：
{json.dumps(
    fact,
    ensure_ascii=False,
    indent=2,
)}

Evidence：
{evidence}

Local Context：
{local_context}
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
                        VERIFICATION_PROMPT,
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

    data = json.loads(
        content
    )

    return VerificationResult(
        supported=bool(
            data.get(
                "supported",
                False,
            )
        ),

        confidence=data.get(
            "confidence",
            "LOW",
        ),

        reason=data.get(
            "reason",
            "",
        ),
    )


# ============================================================
# Local Context
# ============================================================

def get_local_context(
    raw_text: str,
    line_start: int,
    line_end: int | None = None,
    before: int = 2,
    after: int = 2,
) -> str:
    """
    第一版非常简单：

    Fact 前 2 行
        +
    Fact 所在行
        +
    Fact 后 2 行

    例如：

    L0068：
        本公司于2018年12月19日

    会同时看到后面的：

        在公司会议室召开股东会...

    注意：

    这只是 Local Context 第一版实验，
    不是最终通用 Context 算法。
    """

    lines = raw_text.splitlines()

    if line_end is None:
        line_end = line_start

    start_index = max(
        0,
        line_start - 1 - before,
    )

    end_index = min(
        len(lines),
        line_end + after,
    )

    selected_lines = []

    for index in range(
        start_index,
        end_index,
    ):

        line_number = (
            index + 1
        )

        selected_lines.append(
            f"L{line_number:04d} "
            f"{lines[index]}"
        )

    return "\n".join(
        selected_lines
    )


# ============================================================
# 从真实 Pipeline 中找 Fact
# ============================================================

def find_fact(
    facts: list[dict],
    line_start: int,
    predicate_contains: str | None = None,
) -> dict | None:
    """
    根据 line + predicate 找真实 Raw Fact。

    predicate_contains=None：

    直接取该行第一条。

    这样做是因为当前 Fact Discovery
    的 predicate 仍存在一定漂移。
    """

    candidates = [
        fact
        for fact in facts
        if (
            fact.get(
                "line_start"
            )
            == line_start
        )
    ]

    if not candidates:
        return None

    if predicate_contains:

        for fact in candidates:

            predicate = (
                fact.get(
                    "predicate",
                    ""
                )
                or ""
            )

            if (
                predicate_contains
                in predicate
            ):
                return fact

        return None

    return candidates[0]


# ============================================================
# 测试 Case 配置
# ============================================================

CASES = [
    {
        "name":
            "L0007 董事会召开时间",

        "line_start":
            7,

        "predicate_contains":
            "时间",

        "expected":
            True,
    },

    {
    "name": "L0068 错误绑定的股东会召开时间",
    "line_start": 68,
    "predicate_contains": "时间",
    "expected": False,
},

    {
        "name":
            "L0139 孤立日期",

        "line_start":
            139,

        "predicate_contains":
            None,

        "expected":
            False,
    },

    {
        "name":
            "L0123 本决议召开依据",

        "line_start":
            123,

        "predicate_contains":
            "依据",

        "expected":
            True,
    },

 {
    "name": "L0079 错误扁平化的债务关系事实",
    "line_start": 79,
    "predicate_contains": "债务",
    "expected": False,
}
]


# ============================================================
# 将 Pipeline Raw Fact 转换成 Verification Fact
# ============================================================

def build_verification_fact(
    raw_fact: dict,
) -> dict:
    """
    只把 Verification 真正需要的字段送进去。

    不把：

    confidence
    resolution_method
    grounded

    等 Pipeline 自身判断告诉 Verifier，
    避免它受到这些结果影响。
    """

    return {
        "subject":
            raw_fact.get(
                "subject_name"
            ),

        "predicate":
            raw_fact.get(
                "predicate",
                "",
            ),

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
    }


# ============================================================
# main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Real Fact Verification Experiment"
    )

    # ========================================================
    # 读取 Pipeline Result
    # ========================================================

    if not PIPELINE_RESULT_FILE.exists():

        print(
            "找不到："
            f"{PIPELINE_RESULT_FILE}"
        )

        return

    pipeline_data = json.loads(
        PIPELINE_RESULT_FILE.read_text(
            encoding="utf-8"
        )
    )

    facts = pipeline_data.get(
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
        f"Pipeline Facts："
        f"{len(facts)}"
    )

    print(
        f"Cases："
        f"{len(CASES)}"
    )

    passed = 0

    results = []

    # ========================================================
    # 验证
    # ========================================================

    for index, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 60)

        print(
            f"Case #{index}："
            f"{case['name']}"
        )

        raw_fact = find_fact(
            facts=
                facts,

            line_start=
                case[
                    "line_start"
                ],

            predicate_contains=
                case[
                    "predicate_contains"
                ],
        )

        if raw_fact is None:

            print()
            print(
                "找不到匹配 Raw Fact"
            )

            print(
                "Expected：",
                case[
                    "expected"
                ],
            )

            results.append(
                {
                    "case":
                        case["name"],

                    "found":
                        False,

                    "passed":
                        False,
                }
            )

            continue

        verification_fact = (
            build_verification_fact(
                raw_fact
            )
        )

        evidence = (
            raw_fact.get(
                "evidence",
                "",
            )
        )

        line_start = (
            raw_fact.get(
                "line_start"
            )
        )

        line_end = (
            raw_fact.get(
                "line_end"
            )
        )

        local_context = (
            get_local_context(
                raw_text=
                    raw_text,

                line_start=
                    line_start,

                line_end=
                    line_end,

                before=
                    2,

                after=
                    2,
            )
        )

        # ====================================================
        # 打印输入
        # ====================================================

        print()
        print(
            "Fact："
        )

        print(
            json.dumps(
                verification_fact,
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            "Evidence："
        )

        print(
            evidence
        )

        print()
        print(
            "Local Context："
        )

        print(
            local_context
        )

        print()
        print(
            "Pipeline Subject Info："
        )

        print(
            "Status：",
            raw_fact.get(
                "subject_status"
            ),
        )

        print(
            "Confidence：",
            raw_fact.get(
                "subject_confidence"
            ),
        )

        print(
            "Method：",
            raw_fact.get(
                "resolution_method"
            ),
        )

        # ====================================================
        # Verifier
        # ====================================================

        verification = (
            verify_fact(
                fact=
                    verification_fact,

                evidence=
                    evidence,

                local_context=
                    local_context,
            )
        )

        print()
        print(
            "Verification Result："
        )

        print(
            "supported：",
            verification.supported,
        )

        print(
            "confidence：",
            verification.confidence,
        )

        print(
            "reason：",
            verification.reason,
        )

        expected = (
            case[
                "expected"
            ]
        )

        case_passed = (
            verification.supported
            == expected
        )

        print()

        if case_passed:

            print(
                "Validation：通过"
            )

            passed += 1

        else:

            print(
                "Validation：失败"
            )

            print(
                "Expected：",
                expected,
            )

        results.append(
            {
                "case":
                    case[
                        "name"
                    ],

                "found":
                    True,

                "expected":
                    expected,

                "actual":
                    verification.supported,

                "confidence":
                    verification.confidence,

                "reason":
                    verification.reason,

                "passed":
                    case_passed,

                "fact":
                    verification_fact,

                "evidence":
                    evidence,

                "local_context":
                    local_context,
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 60)

    print(
        "实验统计"
    )

    print(
        f"总 Case："
        f"{len(CASES)}"
    )

    print(
        f"通过："
        f"{passed}"
    )

    print(
        f"失败："
        f"{len(CASES) - passed}"
    )

    # ========================================================
    # 保存结果
    # ========================================================

    output_file = (
        BASE_DIR
        / "result.json"
    )

    output_file.write_text(
        json.dumps(
            {
                "source_file":
                    source_file,

                "cases":
                    results,

                "summary": {
                    "total":
                        len(
                            CASES
                        ),

                    "passed":
                        passed,

                    "failed":
                        len(
                            CASES
                        )
                        - passed,
                },
            },
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