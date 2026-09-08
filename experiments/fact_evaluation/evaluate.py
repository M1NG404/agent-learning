import json
from collections import Counter
from pathlib import Path


# ============================================================
# 路径
# ============================================================

BASE_DIR = Path(__file__).parent

GOLD_FILE = (
    BASE_DIR
    / "gold.json"
)

PRED_FILE = (
    BASE_DIR.parent
    / "standard_fact_pipeline"
    / "output"
    / "standard_facts.json"
)


# ============================================================
# 1. ATTRIBUTE Key
# ============================================================

def attribute_key(
    fact: dict,
) -> tuple:
    """
    ATTRIBUTE 严格匹配键。

    line_start 必须加入。

    否则：

    L0039：
    江苏海达 + MEETING_DATE + 2018-12-19

    L0068：
    江苏海达 + MEETING_DATE + 2018-12-19

    会被错误认为是同一条 Fact。
    """

    return (
        fact.get(
            "subject_type",
            "",
        ),
        fact.get(
            "subject",
            "",
        ),
        fact.get(
            "predicate",
            "",
        ),
        fact.get(
            "value",
            "",
        ),
        fact.get(
            "unit",
            "",
        ),
        fact.get(
            "line_start"
        ),
    )


# ============================================================
# 2. RELATION 严格匹配 Key
# ============================================================

def relation_key(
    fact: dict,
) -> tuple:
    """
    RELATION 严格匹配。

    participants
    attributes
    collaterals

    都必须完全一致。

    source 不参与匹配。
    """

    participants = json.dumps(
        fact.get(
            "participants",
            {},
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    attributes = json.dumps(
        fact.get(
            "attributes",
            {},
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    collaterals = json.dumps(
        fact.get(
            "collaterals",
            [],
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    return (
        fact.get(
            "relation_type",
            "",
        ),
        participants,
        attributes,
        collaterals,
        fact.get(
            "line_start",
            fact.get(
                "source",
                {},
            ).get(
                "line_start"
            ),
        ),
    )


# ============================================================
# 3. RELATION 身份 Key
# ============================================================

def relation_identity_key(
    fact: dict,
) -> tuple:
    """
    只判断：

    “是不是同一条关系”

    暂时不比较 attributes。

    用于区分：

    关系没找到
    VS
    关系找到了，但是字段抽错了。
    """

    participants = json.dumps(
        fact.get(
            "participants",
            {},
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    line_start = fact.get(
        "line_start"
    )

    if line_start is None:
        line_start = (
            fact.get(
                "source",
                {},
            )
            .get(
                "line_start"
            )
        )

    return (
        fact.get(
            "relation_type",
            "",
        ),
        participants,
        line_start,
    )


# ============================================================
# 4. Counter-based Evaluation
# ============================================================

def evaluate_facts(
    gold_facts: list[dict],
    pred_facts: list[dict],
    key_func,
) -> dict:
    """
    使用 Counter，而不是 dict。

    这样即使两条 Fact 的 key 相同，
    也不会被覆盖。

    例如：

    Gold 有 1 条
    Prediction 有 2 条

    会得到：

    TP = 1
    FP = 1
    """

    gold_keys = [
        key_func(
            fact
        )
        for fact in gold_facts
    ]

    pred_keys = [
        key_func(
            fact
        )
        for fact in pred_facts
    ]

    gold_counter = Counter(
        gold_keys
    )

    pred_counter = Counter(
        pred_keys
    )

    matched_counter = (
        gold_counter
        & pred_counter
    )

    false_positive_counter = (
        pred_counter
        - gold_counter
    )

    false_negative_counter = (
        gold_counter
        - pred_counter
    )

    tp = sum(
        matched_counter.values()
    )

    fp = sum(
        false_positive_counter.values()
    )

    fn = sum(
        false_negative_counter.values()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
            > 0
        )
        else 0
    )

    # ========================================================
    # 找具体 FP / FN Fact
    # ========================================================

    remaining_gold = Counter(
        gold_counter
    )

    remaining_pred = Counter(
        pred_counter
    )

    # 先扣掉 matched
    for key, count in (
        matched_counter.items()
    ):
        remaining_gold[
            key
        ] -= count

        remaining_pred[
            key
        ] -= count

    false_positives = []

    for fact in pred_facts:

        key = key_func(
            fact
        )

        if (
            remaining_pred[
                key
            ]
            > 0
        ):
            false_positives.append(
                fact
            )

            remaining_pred[
                key
            ] -= 1

    false_negatives = []

    for fact in gold_facts:

        key = key_func(
            fact
        )

        if (
            remaining_gold[
                key
            ]
            > 0
        ):
            false_negatives.append(
                fact
            )

            remaining_gold[
                key
            ] -= 1

    return {
        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,
    }


# ============================================================
# 5. Relation Field Comparison
# ============================================================

def compare_relation_fields(
    gold_fact: dict,
    pred_fact: dict,
) -> dict:
    """
    已经确定是同一条 Relation 后，

    对内部字段逐个比较。

    这样可以知道：

    是整个 Relation 错了，

    还是：

    debt_scope
    amount
    liability_type

    其中某个字段错了。
    """

    differences = []

    # ========================================================
    # relation_type
    # ========================================================

    gold_relation_type = (
        gold_fact.get(
            "relation_type",
            "",
        )
    )

    pred_relation_type = (
        pred_fact.get(
            "relation_type",
            "",
        )
    )

    if (
        gold_relation_type
        != pred_relation_type
    ):
        differences.append(
            {
                "field":
                    "relation_type",

                "gold":
                    gold_relation_type,

                "pred":
                    pred_relation_type,
            }
        )

    # ========================================================
    # participants
    # ========================================================

    gold_participants = (
        gold_fact.get(
            "participants",
            {},
        )
    )

    pred_participants = (
        pred_fact.get(
            "participants",
            {},
        )
    )

    participant_keys = (
        set(
            gold_participants.keys()
        )
        |
        set(
            pred_participants.keys()
        )
    )

    for key in sorted(
        participant_keys
    ):

        gold_value = (
            gold_participants.get(
                key,
                "",
            )
        )

        pred_value = (
            pred_participants.get(
                key,
                "",
            )
        )

        if (
            gold_value
            != pred_value
        ):
            differences.append(
                {
                    "field":
                        f"participants.{key}",

                    "gold":
                        gold_value,

                    "pred":
                        pred_value,
                }
            )

    # ========================================================
    # attributes
    # ========================================================

    gold_attributes = (
        gold_fact.get(
            "attributes",
            {},
        )
    )

    pred_attributes = (
        pred_fact.get(
            "attributes",
            {},
        )
    )

    attribute_keys = (
        set(
            gold_attributes.keys()
        )
        |
        set(
            pred_attributes.keys()
        )
    )

    for key in sorted(
        attribute_keys
    ):

        gold_value = (
            gold_attributes.get(
                key,
                "",
            )
        )

        pred_value = (
            pred_attributes.get(
                key,
                "",
            )
        )

        if (
            gold_value
            != pred_value
        ):
            differences.append(
                {
                    "field":
                        f"attributes.{key}",

                    "gold":
                        gold_value,

                    "pred":
                        pred_value,
                }
            )

    # ========================================================
    # collaterals
    # ========================================================

    gold_collaterals = (
        gold_fact.get(
            "collaterals",
            [],
        )
    )

    pred_collaterals = (
        pred_fact.get(
            "collaterals",
            [],
        )
    )

    if (
        gold_collaterals
        != pred_collaterals
    ):
        differences.append(
            {
                "field":
                    "collaterals",

                "gold":
                    gold_collaterals,

                "pred":
                    pred_collaterals,
            }
        )

    return {
        "matched":
            len(
                differences
            )
            == 0,

        "differences":
            differences,
    }


# ============================================================
# 6. Relation Identity Evaluation
# ============================================================

def evaluate_relation_identity(
    gold_facts: list[dict],
    pred_facts: list[dict],
) -> dict:
    """
    第一层：

    关系身份是否找对。

    relation_type
    +
    participants
    +
    line_start

    一致就认为：
    “找到了同一条关系”。

    然后继续做字段级比较。
    """

    gold_by_identity = {}

    for fact in gold_facts:

        key = (
            relation_identity_key(
                fact
            )
        )

        gold_by_identity.setdefault(
            key,
            [],
        ).append(
            fact
        )

    pred_by_identity = {}

    for fact in pred_facts:

        key = (
            relation_identity_key(
                fact
            )
        )

        pred_by_identity.setdefault(
            key,
            [],
        ).append(
            fact
        )

    gold_keys = set(
        gold_by_identity.keys()
    )

    pred_keys = set(
        pred_by_identity.keys()
    )

    matched_keys = (
        gold_keys
        & pred_keys
    )

    missing_keys = (
        gold_keys
        - pred_keys
    )

    extra_keys = (
        pred_keys
        - gold_keys
    )

    field_comparisons = []

    for key in matched_keys:

        gold_fact = (
            gold_by_identity[
                key
            ][0]
        )

        pred_fact = (
            pred_by_identity[
                key
            ][0]
        )

        comparison = (
            compare_relation_fields(
                gold_fact=
                    gold_fact,

                pred_fact=
                    pred_fact,
            )
        )

        field_comparisons.append(
            {
                "identity":
                    key,

                "gold":
                    gold_fact,

                "pred":
                    pred_fact,

                "differences":
                    comparison[
                        "differences"
                    ],

                "fields_exact":
                    comparison[
                        "matched"
                    ],
            }
        )

    return {
        "matched_relations":
            len(
                matched_keys
            ),

        "gold_relations":
            len(
                gold_keys
            ),

        "pred_relations":
            len(
                pred_keys
            ),

        "missing_relations": [
            gold_by_identity[
                key
            ][0]
            for key in missing_keys
        ],

        "extra_relations": [
            pred_by_identity[
                key
            ][0]
            for key in extra_keys
        ],

        "field_comparisons":
            field_comparisons,
    }


# ============================================================
# 7. 打印 Strict Result
# ============================================================

def print_result(
    name: str,
    result: dict,
):
    print()
    print("=" * 60)
    print(
        name
    )

    print(
        f"TP："
        f"{result['tp']}"
    )

    print(
        f"FP："
        f"{result['fp']}"
    )

    print(
        f"FN："
        f"{result['fn']}"
    )

    print(
        "Precision："
        f"{result['precision']:.2%}"
    )

    print(
        "Recall："
        f"{result['recall']:.2%}"
    )

    print(
        "F1："
        f"{result['f1']:.2%}"
    )

    print()
    print(
        "False Positives："
    )

    if not result[
        "false_positives"
    ]:

        print(
            "无"
        )

    else:

        for fact in result[
            "false_positives"
        ]:

            print(
                json.dumps(
                    fact,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    print()
    print(
        "False Negatives："
    )

    if not result[
        "false_negatives"
    ]:

        print(
            "无"
        )

    else:

        for fact in result[
            "false_negatives"
        ]:

            print(
                json.dumps(
                    fact,
                    ensure_ascii=False,
                    indent=2,
                )
            )


# ============================================================
# 8. 打印 Relation 字段级结果
# ============================================================

def print_relation_identity_result(
    result: dict,
):
    print()
    print("=" * 60)

    print(
        "RELATION FIELD-LEVEL"
    )

    print(
        "Gold Relations：",
        result[
            "gold_relations"
        ],
    )

    print(
        "Pred Relations：",
        result[
            "pred_relations"
        ],
    )

    print(
        "Matched Relation Identity：",
        result[
            "matched_relations"
        ],
    )

    print()

    print(
        "Missing Relations："
    )

    if not result[
        "missing_relations"
    ]:

        print(
            "无"
        )

    else:

        for fact in result[
            "missing_relations"
        ]:

            print(
                json.dumps(
                    fact,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    print()

    print(
        "Extra Relations："
    )

    if not result[
        "extra_relations"
    ]:

        print(
            "无"
        )

    else:

        for fact in result[
            "extra_relations"
        ]:

            print(
                json.dumps(
                    fact,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    print()

    print(
        "Relation Field Differences："
    )

    has_difference = False

    for item in result[
        "field_comparisons"
    ]:

        if item[
            "fields_exact"
        ]:
            continue

        has_difference = True

        pred = item[
            "pred"
        ]

        source = pred.get(
            "source",
            {},
        )

        line_start = source.get(
            "line_start",
            pred.get(
                "line_start"
            ),
        )

        print("-" * 60)

        if line_start is not None:

            print(
                f"Relation："
                f"L{line_start:04d}"
            )

        print(
            "Relation Type：",
            pred.get(
                "relation_type",
                "",
            ),
        )

        for difference in item[
            "differences"
        ]:

            print(
                f"字段："
                f"{difference['field']}"
            )

            print(
                f"  Gold："
                f"{difference['gold']}"
            )

            print(
                f"  Pred："
                f"{difference['pred']}"
            )

    if not has_difference:

        print(
            "无"
        )


# ============================================================
# 9. main
# ============================================================

def main():

    # ========================================================
    # 文件检查
    # ========================================================

    if not GOLD_FILE.exists():

        print(
            f"找不到 Gold："
            f"{GOLD_FILE}"
        )

        return

    if not PRED_FILE.exists():

        print(
            f"找不到 Prediction："
            f"{PRED_FILE}"
        )

        return

    # ========================================================
    # 读取
    # ========================================================

    gold = json.loads(
        GOLD_FILE.read_text(
            encoding="utf-8"
        )
    )

    pred = json.loads(
        PRED_FILE.read_text(
            encoding="utf-8"
        )
    )

    gold_attributes = (
        gold.get(
            "attribute_facts",
            [],
        )
    )

    pred_attributes = (
        pred.get(
            "attribute_facts",
            [],
        )
    )

    gold_relations = (
        gold.get(
            "relation_facts",
            [],
        )
    )

    pred_relations = (
        pred.get(
            "relation_facts",
            [],
        )
    )

    # ========================================================
    # ATTRIBUTE Strict
    # ========================================================

    attribute_result = (
        evaluate_facts(
            gold_facts=
                gold_attributes,

            pred_facts=
                pred_attributes,

            key_func=
                attribute_key,
        )
    )

    # ========================================================
    # RELATION Strict
    # ========================================================

    relation_result = (
        evaluate_facts(
            gold_facts=
                gold_relations,

            pred_facts=
                pred_relations,

            key_func=
                relation_key,
        )
    )

    # ========================================================
    # RELATION Field-level
    # ========================================================

    relation_identity_result = (
        evaluate_relation_identity(
            gold_facts=
                gold_relations,

            pred_facts=
                pred_relations,
        )
    )

    # ========================================================
    # 输出
    # ========================================================

    print("=" * 60)

    print(
        "Fact Evaluation"
    )

    print(
        f"Gold Attribute："
        f"{len(gold_attributes)}"
    )

    print(
        f"Pred Attribute："
        f"{len(pred_attributes)}"
    )

    print(
        f"Gold Relation："
        f"{len(gold_relations)}"
    )

    print(
        f"Pred Relation："
        f"{len(pred_relations)}"
    )

    # --------------------------------------------------------
    # ATTRIBUTE
    # --------------------------------------------------------

    print_result(
        name=
            "ATTRIBUTE STRICT",

        result=
            attribute_result,
    )

    # --------------------------------------------------------
    # RELATION STRICT
    # --------------------------------------------------------

    print_result(
        name=
            "RELATION STRICT",

        result=
            relation_result,
    )

    # --------------------------------------------------------
    # RELATION FIELD LEVEL
    # --------------------------------------------------------

    print_relation_identity_result(
        relation_identity_result
    )


if __name__ == "__main__":
    main()