import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

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
# Prompt
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

你必须重点验证：

- subject 是否正确
- predicate 是否被原文表达
- value 是否正确
- subject 是否承担正确的语义角色

注意：

Grounding 只说明 Evidence 来自原文，
并不代表 Fact 的语义一定正确。

Local Context 可以用于补足 Evidence 本身缺失的上下文。

例如：

Fact:
subject = A公司
predicate = MEETING_DATE
value = 2018年12月19日

Evidence:
A公司于2018年12月19日

Local Context:
A公司于2018年12月19日在公司会议室召开股东会。

这条 Fact 应当判断为 supported = true。

因为虽然 Evidence 本身没有“召开股东会”，
但 Local Context 明确补足了该语义。


反例：

Fact:
subject = A公司
predicate = CONTRACT_SIGN_DATE
value = 2018年12月19日

Evidence:
A公司于2018年12月19日

Local Context:
A公司于2018年12月19日在公司会议室召开股东会。

应判断：

supported = false

因为原文表达的是股东会召开时间，
不是合同签署时间。


角色反例：

Fact:
subject = A公司
predicate = PROVIDE_GUARANTEE
value = 连带保证

Local Context:
A公司的债务由B公司提供连带保证。

应判断：

supported = false

因为提供担保的是 B公司，不是 A公司。


规则：

1. 只能依据 Evidence 和 Local Context 判断。
2. 不允许使用常识补充。
3. 不允许因为 subject/value 字符串出现在文本中就直接判 true。
4. 必须验证 predicate 的语义。
5. 必须验证 subject 的语义角色。
6. 如果证据不足，应判断 supported = false。
7. confidence 只能是 HIGH / MEDIUM / LOW。
8. 只输出 JSON。

输出格式：

{
  "supported": true,
  "confidence": "HIGH",
  "reason": "..."
}
"""


# ============================================================
# Verifier
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
                    "role": "system",
                    "content": VERIFICATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
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
# Cases
# ============================================================

CASES = [
    {
        "name": "Case 1：完整证据直接支持",
        "fact": {
            "subject":
                "A公司",
            "predicate":
                "MEETING_DATE",
            "value":
                "2018年12月19日",
        },
        "evidence":
            "A公司于2018年12月19日在公司会议室召开股东会。",
        "local_context":
            "A公司于2018年12月19日在公司会议室召开股东会。",
        "expected":
            True,
    },

    {
        "name": "Case 2：Evidence 较短，但 Local Context 支持",
        "fact": {
            "subject":
                "江苏海达科技集团有限公司",
            "predicate":
                "MEETING_DATE",
            "value":
                "2018年12月19日",
        },
        "evidence":
            "本公司于2018年12月19日",
        "local_context":
            (
                "江苏海达科技集团有限公司（以下简称本公司）"
                "于2018年12月19日\n\n"
                "在公司会议室（地点）召开股东会"
                "同意以下事项并形成本决议："
            ),
        "expected":
            True,
    },

    {
        "name": "Case 3：Value 存在，但 Predicate 不成立",
        "fact": {
            "subject":
                "A公司",
            "predicate":
                "CONTRACT_SIGN_DATE",
            "value":
                "2018年12月19日",
        },
        "evidence":
            "A公司于2018年12月19日",
        "local_context":
            (
                "A公司于2018年12月19日"
                "在公司会议室召开股东会。"
            ),
        "expected":
            False,
    },

    {
        "name": "Case 4：主体角色错误",
        "fact": {
            "subject":
                "A公司",
            "predicate":
                "PROVIDE_GUARANTEE",
            "value":
                "连带保证",
        },
        "evidence":
            "A公司的债务由B公司提供连带保证。",
        "local_context":
            "A公司的债务由B公司提供连带保证。",
        "expected":
            False,
    },
]


# ============================================================
# main
# ============================================================

def main():

    print("=" * 60)
    print("Fact Verification Experiment")
    print(f"Cases：{len(CASES)}")

    passed = 0

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

        print()
        print("Fact：")

        print(
            json.dumps(
                case["fact"],
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print("Evidence：")
        print(
            case["evidence"]
        )

        print()
        print("Local Context：")
        print(
            case["local_context"]
        )

        result = verify_fact(
            fact=
                case["fact"],

            evidence=
                case["evidence"],

            local_context=
                case["local_context"],
        )

        print()
        print("Verification Result：")

        print(
            f"supported："
            f"{result.supported}"
        )

        print(
            f"confidence："
            f"{result.confidence}"
        )

        print(
            f"reason："
            f"{result.reason}"
        )

        expected = (
            case["expected"]
        )

        if (
            result.supported
            == expected
        ):

            print()
            print(
                "Validation：通过"
            )

            passed += 1

        else:

            print()
            print(
                "Validation：失败"
            )

            print(
                f"Expected："
                f"{expected}"
            )

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


if __name__ == "__main__":
    main()