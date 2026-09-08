import json
import os

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
# 测试 Case
# ============================================================

CASES = [
    {
        "name": "L0010 抵押/质押担保",

        "resolved_subject": "江阴利泰装饰材料有限公司",

        "evidence": (
            "同意本公司以本公司所有的以下资产为"
            "江阴东华铝材科技有限公司对中国银行股份有限公司江阴支行"
            "（或其下属分支机构）贰仟万元授信额度债务及其相应的利息、"
            "罚息、违约金、应付费用等提供抵押、（或质押）担保，"
            "直至前述债务全部清偿完毕。"
        ),

        "expected_relation_type": "GUARANTEE",
    },

    {
        "name": "L0042 连带保证",

        "resolved_subject": "江苏海达科技集团有限公司",

        "evidence": (
            "同意本公司为江阴东华铝材科技有限公司对"
            "中国银行股份有限公司江阴支行（或其下属分支机构）"
            "壹亿元授信额度债务及其相应的利息、罚息、违约金、"
            "应付费用等提供连带保证责任担保，"
            "直至前述债务全部清偿完毕。"
        ),

        "expected_relation_type": "GUARANTEE",
    },

    {
        "name": "L0079 共同债务",

        "resolved_subject": "江阴海达特种人革有限公司",

        "evidence": (
            "同意本公司为江阴东华铝材科技有限公司对"
            "中国银行股份有限公司江阴支行（或其下属分支机构）"
            "壹亿元授信额度债务及其相应的利息、罚息、违约金、"
            "应付费用等作为负有连带义务共同债务人承担债务，"
            "直至前述债务全部清偿完毕。"
        ),

        "expected_relation_type": "DEBT",
    },
]


# ============================================================
# Prompt
# ============================================================

PROMPT = """
你是一个不良资产尽调关系事实抽取器。

输入包含：

1. 已解析主体
2. 已经完成 Grounding 的原文 Evidence

你的任务：

先判断 Evidence 描述的核心业务关系类型，
然后一次性提取成完整 RelationFact。

当前只允许两种 relation_type：

GUARANTEE
DEBT


============================================================
一、GUARANTEE
============================================================

当原文表达：

某主体为某债务提供抵押、质押、保证等担保

使用：

{
  "fact_type": "RELATION",
  "relation_type": "GUARANTEE",
  "participants": {
    "guarantor": "",
    "debtor": "",
    "creditor": ""
  },
  "attributes": {
    "amount": "",
    "guarantee_type": "",
    "debt_scope": "",
    "end_condition": ""
  }
}


字段含义：

guarantor：
提供担保的主体。

如果原文使用“本公司”，
必须使用输入中的“已解析主体”。

debtor：
被担保债务的原始债务人。

creditor：
债权人。

amount：
纯金额。

例如：

正确：
"壹亿元"

错误：
"壹亿元授信额度债务及其相应的利息……"

guarantee_type：
具体担保方式。

例如：

"抵押、（或质押）担保"

"连带保证责任担保"

debt_scope：
担保所覆盖的债务范围。

例如：

"授信额度债务及其相应的利息、罚息、违约金、应付费用等"

end_condition：
担保责任终止条件。


============================================================
二、DEBT
============================================================

当原文表达：

某主体作为共同债务人、连带债务人等直接承担债务

使用：

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


字段含义：

obligor：
承担债务义务的主体。

如果原文使用“本公司”，
必须使用输入中的“已解析主体”。

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
承担债务的责任性质。

例如：

"负有连带义务共同债务人"

end_condition：
债务责任终止条件。


============================================================
通用规则
============================================================

1. 必须先根据原文语义判断 relation_type。

2. 提供担保：
   -> GUARANTEE

3. 直接作为共同债务人承担债务：
   -> DEBT

4. 不允许因为存在“连带”二字就一律判断为 GUARANTEE。

例如：

"连带保证责任担保"
-> GUARANTEE

"负有连带义务共同债务人"
-> DEBT

5. 所有字段只能使用原文明确表达的信息。

6. 原文没有的信息填写空字符串。

7. amount 必须保持原文形式，不允许换算。

8. 如果原文使用“本公司”，
   必须使用输入中的“已解析主体”。

9. 不允许输出“本公司”作为最终 participant。

10. 不允许补充常识或猜测。

11. creditor 如果原文包含：
    “中国银行股份有限公司江阴支行（或其下属分支机构）”

    可以输出原文完整表达，
    也可以输出明确主体：
    “中国银行股份有限公司江阴支行”

    但不能换成其他机构。

12. 只输出 JSON，不要输出解释。
"""


# ============================================================
# Relation Extraction
# ============================================================

def extract_relation(
    evidence: str,
    resolved_subject: str,
) -> dict:
    """
    输入：

        Grounded Evidence
        +
        Resolved Subject

    输出：

        GUARANTEE
        或
        DEBT

        RelationFact
    """

    user_content = f"""
已解析主体：
{resolved_subject}

原文 Evidence：
{evidence}
""".strip()

    response = client.chat.completions.create(
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
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    return json.loads(content)


# ============================================================
# 基础 Validation
# ============================================================

def validate_relation(
    relation: dict,
    resolved_subject: str,
    expected_relation_type: str,
) -> list[str]:
    """
    第一版 Validation。

    只检查最关键的结构和主体。
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

    # --------------------------------------------------------
    # Fact Type
    # --------------------------------------------------------

    if fact_type != "RELATION":
        errors.append(
            f"fact_type 错误：{fact_type}"
        )

    # --------------------------------------------------------
    # Relation Type
    # --------------------------------------------------------

    if relation_type != expected_relation_type:
        errors.append(
            "relation_type 错误："
            f"期望 {expected_relation_type}，"
            f"实际 {relation_type}"
        )

    # --------------------------------------------------------
    # 根据 Relation Type 校验主体
    # --------------------------------------------------------

    if relation_type == "GUARANTEE":

        guarantor = participants.get(
            "guarantor",
            "",
        )

        if guarantor == "本公司":
            errors.append(
                "guarantor 仍然是“本公司”"
            )

        if guarantor != resolved_subject:
            errors.append(
                "guarantor 与已解析主体不一致："
                f"{guarantor}"
            )

        # GUARANTEE 必须有 guarantee_type
        guarantee_type = attributes.get(
            "guarantee_type",
            "",
        )

        if not guarantee_type:
            errors.append(
                "GUARANTEE 缺少 guarantee_type"
            )

    elif relation_type == "DEBT":

        obligor = participants.get(
            "obligor",
            "",
        )

        if obligor == "本公司":
            errors.append(
                "obligor 仍然是“本公司”"
            )

        if obligor != resolved_subject:
            errors.append(
                "obligor 与已解析主体不一致："
                f"{obligor}"
            )

        liability_type = attributes.get(
            "liability_type",
            "",
        )

        if not liability_type:
            errors.append(
                "DEBT 缺少 liability_type"
            )

    # --------------------------------------------------------
    # 通用 Participants
    # --------------------------------------------------------

    debtor = participants.get(
        "debtor",
        "",
    )

    creditor = participants.get(
        "creditor",
        "",
    )

    if not debtor:
        errors.append(
            "缺少 debtor"
        )

    if not creditor:
        errors.append(
            "缺少 creditor"
        )

    # --------------------------------------------------------
    # amount
    # --------------------------------------------------------

    amount = attributes.get(
        "amount",
        "",
    )

    if not amount:
        errors.append(
            "缺少 amount"
        )

    # 防止 amount 把业务描述一起吞进去
    invalid_amount_words = [
        "授信额度债务",
        "利息",
        "罚息",
        "违约金",
        "应付费用",
    ]

    for word in invalid_amount_words:

        if word in amount:

            errors.append(
                f"amount 包含非金额信息：{amount}"
            )

            break

    return errors


# ============================================================
# 打印 Result
# ============================================================

def print_result(
    case_index: int,
    case: dict,
    relation: dict,
    errors: list[str],
):
    print()
    print("=" * 60)

    print(
        f"Case #{case_index}："
        f"{case['name']}"
    )

    print()

    print(
        "Resolved Subject："
    )

    print(
        case["resolved_subject"]
    )

    print()

    print(
        "Evidence："
    )

    print(
        case["evidence"]
    )

    print()

    print(
        "Expected Relation Type：",
        case[
            "expected_relation_type"
        ],
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

    print()

    print(
        "Validation："
    )

    if not errors:

        print(
            "通过"
        )

    else:

        print(
            f"失败，发现 {len(errors)} 个问题："
        )

        for error in errors:

            print(
                f"- {error}"
            )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Relation Extraction Experiment"
    )

    print(
        f"Cases：{len(CASES)}"
    )

    total_count = len(
        CASES
    )

    success_count = 0

    failure_count = 0

    results = []

    # ========================================================
    # 逐个 Case 运行
    # ========================================================

    for index, case in enumerate(
        CASES,
        start=1,
    ):

        relation = extract_relation(
            evidence=
                case["evidence"],

            resolved_subject=
                case["resolved_subject"],
        )

        errors = validate_relation(
            relation=
                relation,

            resolved_subject=
                case["resolved_subject"],

            expected_relation_type=
                case[
                    "expected_relation_type"
                ],
        )

        if not errors:
            success_count += 1
        else:
            failure_count += 1

        print_result(
            case_index=
                index,

            case=
                case,

            relation=
                relation,

            errors=
                errors,
        )

        results.append(
            {
                "case":
                    case["name"],

                "expected_relation_type":
                    case[
                        "expected_relation_type"
                    ],

                "relation":
                    relation,

                "validation_errors":
                    errors,
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
        f"{total_count}"
    )

    print(
        f"通过："
        f"{success_count}"
    )

    print(
        f"失败："
        f"{failure_count}"
    )

    # ========================================================
    # 输出 JSON
    # ========================================================

    output = {
        "summary": {
            "total":
                total_count,

            "success":
                success_count,

            "failure":
                failure_count,
        },

        "results":
            results,
    }

    output_path = (
        os.path.join(
            os.path.dirname(__file__),
            "result.json",
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()

    print(
        f"结果已保存："
        f"{output_path}"
    )


if __name__ == "__main__":
    main()