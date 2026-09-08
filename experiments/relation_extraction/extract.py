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

        # 这里不再只给 L0010，
        # 而是给完整 Relation Context：L0010 ~ L0014
        "evidence": """
1、同意本公司以本公司所有的以下资产为江阴东华铝材科技有限公司对中国银行股份有限公司江阴支行（或其下属分支机构）贰仟万元授信额度债务及其相应的利息、罚息、违约金、应付费用等提供抵押、（或质押）担保，直至前述债务全部清偿完毕。

- A、澄土国用（2008）第 3884 号项下 20879.3 平方米土地使用权
- B、澄土国用（2008）第 3885 号项下 10881 平方米土地使用权
- C、房权证澄字第 fhs0002840 号项下 27645.04 平方米房产
""".strip(),

        "expected_relation_type": "GUARANTEE",

        # 实验阶段人工知道：
        # 这一条应该有 3 个 collateral
        "expected_collateral_count": 3,
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

        # 保证担保，没有明确抵押物
        "expected_collateral_count": 0,
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

        # DEBT 关系不使用 collaterals
        "expected_collateral_count": None,
    },
]


# ============================================================
# Prompt
# ============================================================

PROMPT = """
你是一个不良资产尽调关系事实抽取器。

输入包含：

1. 已解析主体
2. 已完成 Grounding 的原文 Evidence

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

某主体为某项债务提供抵押、质押、保证等担保

使用：

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

必须保留原文金额形式，
禁止换算成阿拉伯数字。


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


collaterals：
原文明确列出的抵押物、质押物等担保财产。

如果存在多个资产，
必须逐项输出，禁止合并。

每一个 collateral 使用：

{
  "type": "",
  "certificate_no": "",
  "area": "",
  "unit": ""
}

其中：

type 当前只允许：

LAND_USE_RIGHT
REAL_ESTATE
OTHER


LAND_USE_RIGHT：
土地使用权。


REAL_ESTATE：
房屋、房产等不动产。


OTHER：
其他无法归入上述类型的担保财产。


certificate_no：
权证号。

必须保持原文形式。


area：
只保存面积数值。

例如：

原文：
20879.3 平方米

正确：

"area": "20879.3"

错误：

"area": "20879.3 平方米"


unit：
面积单位。

例如：

"平方米"


如果原文没有明确列出担保财产：

"collaterals": []

禁止根据常识推断不存在的 collateral。


============================================================
二、DEBT
============================================================

当原文表达：

某主体作为共同债务人、连带债务人等，
直接承担债务

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

例如：

"壹亿元"


debt_type：
核心债务类型。

例如：

"授信额度债务"


debt_scope：
债务覆盖范围。

例如：

"授信额度债务及其相应的利息、罚息、违约金、应付费用等"


liability_type：
承担债务的责任性质。

例如：

"负有连带义务共同债务人"


end_condition：
债务责任终止条件。


============================================================
三、关系类型判断规则
============================================================

1. 提供抵押、质押、保证等担保：

GUARANTEE


2. 作为共同债务人、连带债务人等直接承担债务：

DEBT


3. 不允许只因为出现“连带”二字就判断为 GUARANTEE。

例如：

"连带保证责任担保"

属于：

GUARANTEE


"负有连带义务共同债务人"

属于：

DEBT


============================================================
四、通用规则
============================================================

1. 必须根据 Evidence 的实际语义判断 relation_type。

2. 所有字段只能来自原文明确表达的信息。

3. 原文没有的信息填写空字符串。

4. 数组字段没有内容时使用空数组 []。

5. amount 必须保持原文形式，不允许自行换算。

6. 如果原文使用“本公司”，
   participant 中必须使用输入的“已解析主体”。

7. 最终 participant 中禁止出现“本公司”。

8. 禁止补充原文没有表达的常识信息。

9. creditor 如果原文为：

“中国银行股份有限公司江阴支行（或其下属分支机构）”

可以保留完整原文表达，
也可以使用明确主体：

“中国银行股份有限公司江阴支行”

但禁止换成其他机构。

10. collateral 的 certificate_no、area、unit
必须来自 Evidence。

11. 不允许因为看到土地证号或房产证号，
就自行补充 Evidence 中不存在的权属关系。

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

        Relation Context
        +
        Resolved Subject

    输出：

        GUARANTEE RelationFact

        或

        DEBT RelationFact
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
# Validation
# ============================================================

def validate_relation(
    relation: dict,
    resolved_subject: str,
    expected_relation_type: str,
    expected_collateral_count: int | None,
) -> list[str]:
    """
    实验阶段基础校验。

    当前验证：

    1. Relation Type
    2. Resolved Subject
    3. debtor
    4. creditor
    5. amount
    6. GUARANTEE 的 guarantee_type
    7. DEBT 的 liability_type
    8. collateral 数量
    9. collateral 基本字段
    """

    errors = []

    # ========================================================
    # 基础结构
    # ========================================================

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

    # ========================================================
    # Fact Type
    # ========================================================

    if fact_type != "RELATION":

        errors.append(
            f"fact_type 错误：{fact_type}"
        )

    # ========================================================
    # Relation Type
    # ========================================================

    if relation_type != expected_relation_type:

        errors.append(
            "relation_type 错误："
            f"期望 {expected_relation_type}，"
            f"实际 {relation_type}"
        )

    # ========================================================
    # GUARANTEE
    # ========================================================

    if relation_type == "GUARANTEE":

        guarantor = participants.get(
            "guarantor",
            "",
        )

        # ----------------------------------------------------
        # Resolved Subject
        # ----------------------------------------------------

        if guarantor == "本公司":

            errors.append(
                "guarantor 仍然是“本公司”"
            )

        if guarantor != resolved_subject:

            errors.append(
                "guarantor 与已解析主体不一致："
                f"{guarantor}"
            )

        # ----------------------------------------------------
        # guarantee_type
        # ----------------------------------------------------

        guarantee_type = attributes.get(
            "guarantee_type",
            "",
        )

        if not guarantee_type:

            errors.append(
                "GUARANTEE 缺少 guarantee_type"
            )

        # ----------------------------------------------------
        # collaterals
        # ----------------------------------------------------

        collaterals = relation.get(
            "collaterals",
            [],
        )

        if not isinstance(
            collaterals,
            list,
        ):

            errors.append(
                "collaterals 必须是数组"
            )

            collaterals = []

        # 实验阶段我们知道预期数量
        if expected_collateral_count is not None:

            if (
                len(collaterals)
                != expected_collateral_count
            ):

                errors.append(
                    "collateral 数量错误："
                    f"期望 {expected_collateral_count}，"
                    f"实际 {len(collaterals)}"
                )

        # ----------------------------------------------------
        # collateral 字段校验
        # ----------------------------------------------------

        allowed_types = {
            "LAND_USE_RIGHT",
            "REAL_ESTATE",
            "OTHER",
        }

        for index, collateral in enumerate(
            collaterals,
            start=1,
        ):

            collateral_type = collateral.get(
                "type",
                "",
            )

            certificate_no = collateral.get(
                "certificate_no",
                "",
            )

            area = collateral.get(
                "area",
                "",
            )

            unit = collateral.get(
                "unit",
                "",
            )

            if collateral_type not in allowed_types:

                errors.append(
                    f"collateral #{index} "
                    f"type 非法：{collateral_type}"
                )

            if not certificate_no:

                errors.append(
                    f"collateral #{index} "
                    "缺少 certificate_no"
                )

            if not area:

                errors.append(
                    f"collateral #{index} "
                    "缺少 area"
                )

            if area and "平方米" in area:

                errors.append(
                    f"collateral #{index} "
                    f"area 不应包含单位：{area}"
                )

            if area and unit != "平方米":

                errors.append(
                    f"collateral #{index} "
                    f"unit 异常：{unit}"
                )

    # ========================================================
    # DEBT
    # ========================================================

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

    # ========================================================
    # 通用 Participants
    # ========================================================

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

    # ========================================================
    # Amount
    # ========================================================

    amount = attributes.get(
        "amount",
        "",
    )

    if not amount:

        errors.append(
            "缺少 amount"
        )

    # amount 不能吞掉后面的业务描述
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
                "amount 包含非金额信息："
                f"{amount}"
            )

            break

    return errors


# ============================================================
# 打印结果
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
        case[
            "resolved_subject"
        ]
    )

    print()

    print(
        "Evidence："
    )

    print(
        case[
            "evidence"
        ]
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
    # 执行所有 Case
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

            expected_collateral_count=
                case[
                    "expected_collateral_count"
                ],
        )

        if not errors:

            success_count += 1

        else:

            failure_count += 1

        print_result(
            case_index=index,
            case=case,
            relation=relation,
            errors=errors,
        )

        results.append(
            {
                "case":
                    case["name"],

                "expected_relation_type":
                    case[
                        "expected_relation_type"
                    ],

                "expected_collateral_count":
                    case[
                        "expected_collateral_count"
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
    # 保存结果
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

    output_path = os.path.join(
        os.path.dirname(
            __file__
        ),
        "result.json",
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