import json
from pathlib import Path


# ============================================================
# 路径
# ============================================================

BASE_DIR = Path(__file__).parent

PIPELINE_RESULT = (
    BASE_DIR.parent
    / "fact_pipeline"
    / "output"
    / "pipeline_result.json"
)


# ============================================================
# 1. Predicate 标准化规则
# ============================================================

PREDICATE_MAPPING = {
    # --------------------------------------------------------
    # Meeting
    # --------------------------------------------------------
    "召开董事会时间": "MEETING_DATE",
    "董事会召开日期": "MEETING_DATE",

    "召开股东会时间": "MEETING_DATE",
    "股东会召开日期": "MEETING_DATE",

    "召开股东会地点": "MEETING_LOCATION",
    "股东会召开地点": "MEETING_LOCATION",

    # --------------------------------------------------------
    # Property
    # --------------------------------------------------------
    "土地使用权面积": "AREA",
    "房产面积": "AREA",

    "土地使用权证号": "CERTIFICATE_NO",
    "房产证号": "CERTIFICATE_NO",
}


# ============================================================
# 2. Raw Fact 分类规则
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


# ============================================================
# 3. 分类
# ============================================================

def classify_fact(
    raw_fact: dict,
) -> str:
    """
    把 Raw Fact 分类为：

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
# 4. Predicate 标准化
# ============================================================

def normalize_predicate(
    raw_predicate: str,
) -> str:
    """
    把模型自由生成的 predicate
    映射成标准 predicate。
    """

    return PREDICATE_MAPPING.get(
        raw_predicate,
        "UNKNOWN",
    )


# ============================================================
# 5. Subject Type
# ============================================================

def infer_subject_type(
    subject: str,
) -> str:
    """
    第一版先使用简单规则判断 Subject Type。

    后续再替换成真正的 Entity Type Resolver。
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


# ============================================================
# 6. ATTRIBUTE Normalization
# ============================================================

def normalize_attribute_fact(
    raw_fact: dict,
) -> dict | None:
    """
    把一个 ATTRIBUTE Raw Fact
    转成标准 AttributeFact。
    """

    raw_predicate = raw_fact.get(
        "predicate",
        "",
    )

    canonical_predicate = (
        normalize_predicate(
            raw_predicate
        )
    )

    if canonical_predicate == "UNKNOWN":
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
            canonical_predicate,

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

        # 保留模型原始输出，
        # 方便后续调试 Schema。
        "raw_predicate":
            raw_predicate,
    }


# ============================================================
# 7. RELATION Grouping
# ============================================================

def group_relation_facts(
    raw_facts: list[dict],
) -> dict:
    """
    把属于同一条原文关系的多个 Raw Fact 分到一组。

    当前第一版使用：

        line_start + evidence

    作为 Group Key。

    例如：

    L0079 同一句话抽出了：

        授信额度债务
        债务类型
        责任性质
        债务履行截止条件

    最终应该进入同一个 Relation Group。
    """

    groups = {}

    for fact in raw_facts:

        if (
            classify_fact(fact)
            != "RELATION"
        ):
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

        groups.setdefault(
            key,
            [],
        ).append(
            fact
        )

    return groups


# ============================================================
# 8. 打印 Attribute Fact
# ============================================================

def print_attribute_fact(
    index: int,
    fact: dict,
):
    print("-" * 60)

    print(
        f"Attribute Fact #{index}"
    )

    print(
        "Subject：",
        fact.get(
            "subject",
            "UNKNOWN",
        ),
    )

    print(
        "Subject Type：",
        fact.get(
            "subject_type",
            "OTHER",
        ),
    )

    print(
        "Raw Predicate：",
        fact.get(
            "raw_predicate",
            "",
        ),
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

    if line_start is not None:
        print(
            f"Line：L{line_start:04d}"
        )

    print(
        "Evidence：",
        fact.get(
            "evidence",
            "",
        ),
    )


# ============================================================
# 9. 打印 Relation Group
# ============================================================

def print_relation_group(
    index: int,
    line_start,
    evidence: str,
    facts: list[dict],
):
    print("-" * 60)

    print(
        f"Relation Group #{index}"
    )

    if line_start is not None:
        print(
            f"Line：L{line_start:04d}"
        )
    else:
        print(
            "Line：None"
        )

    print(
        "Evidence：",
        evidence,
    )

    print(
        "Raw Facts："
    )

    for fact in facts:

        print(
            "  - "
            f"{fact.get('predicate', '')}"
            " = "
            f"{fact.get('value', '')}"
            " "
            f"{fact.get('unit', '')}"
        )


# ============================================================
# 10. 打印 DROP
# ============================================================

def print_drop_fact(
    index: int,
    fact: dict,
):
    print("-" * 60)

    print(
        f"Drop #{index}"
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
        "Evidence：",
        fact.get(
            "evidence",
            "",
        ),
    )


# ============================================================
# 11. 打印 UNKNOWN
# ============================================================

def print_unknown_fact(
    index: int,
    fact: dict,
):
    print("-" * 60)

    print(
        f"Unknown #{index}"
    )

    print(
        "Subject：",
        fact.get(
            "subject_name",
            "UNKNOWN",
        ),
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
        "Evidence：",
        fact.get(
            "evidence",
            "",
        ),
    )


# ============================================================
# 12. main
# ============================================================

def main():

    # ========================================================
    # 读取 Pipeline Result
    # ========================================================

    if not PIPELINE_RESULT.exists():

        print(
            "找不到 Pipeline 输出文件："
        )

        print(
            PIPELINE_RESULT
        )

        return

    data = json.loads(
        PIPELINE_RESULT.read_text(
            encoding="utf-8"
        )
    )

    raw_facts = data.get(
        "facts",
        [],
    )

    # ========================================================
    # 分类
    # ========================================================

    attribute_raw_facts = []
    relation_raw_facts = []
    drop_facts = []
    unknown_facts = []

    for raw_fact in raw_facts:

        fact_class = (
            classify_fact(
                raw_fact
            )
        )

        if fact_class == "ATTRIBUTE":

            attribute_raw_facts.append(
                raw_fact
            )

        elif fact_class == "RELATION":

            relation_raw_facts.append(
                raw_fact
            )

        elif fact_class == "DROP":

            drop_facts.append(
                raw_fact
            )

        else:

            unknown_facts.append(
                raw_fact
            )

    # ========================================================
    # ATTRIBUTE 标准化
    # ========================================================

    normalized_attributes = []

    for raw_fact in attribute_raw_facts:

        normalized = (
            normalize_attribute_fact(
                raw_fact
            )
        )

        if normalized:

            normalized_attributes.append(
                normalized
            )

        else:

            # 理论上当前不会进入这里。
            # 如果映射表漏了，会进入 UNKNOWN。
            unknown_facts.append(
                raw_fact
            )

    # ========================================================
    # RELATION 分组
    # ========================================================

    relation_groups = (
        group_relation_facts(
            raw_facts
        )
    )

    # ========================================================
    # 总体统计
    # ========================================================

    print("=" * 60)
    print(
        "Fact Normalization"
    )

    print(
        f"Raw Facts："
        f"{len(raw_facts)}"
    )

    print(
        f"ATTRIBUTE："
        f"{len(attribute_raw_facts)}"
    )

    print(
        f"RELATION Raw Facts："
        f"{len(relation_raw_facts)}"
    )

    print(
        f"RELATION Groups："
        f"{len(relation_groups)}"
    )

    print(
        f"DROP："
        f"{len(drop_facts)}"
    )

    print(
        f"UNKNOWN："
        f"{len(unknown_facts)}"
    )

    # ========================================================
    # ATTRIBUTE 输出
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Normalized Attribute Facts"
    )

    for index, fact in enumerate(
        normalized_attributes,
        start=1,
    ):

        print_attribute_fact(
            index=index,
            fact=fact,
        )

    # ========================================================
    # RELATION Groups 输出
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Relation Groups"
    )

    for index, (
        (
            line_start,
            evidence,
        ),
        facts,
    ) in enumerate(
        relation_groups.items(),
        start=1,
    ):

        print_relation_group(
            index=index,
            line_start=line_start,
            evidence=evidence,
            facts=facts,
        )

    # ========================================================
    # DROP 输出
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Dropped Facts"
    )

    for index, fact in enumerate(
        drop_facts,
        start=1,
    ):

        print_drop_fact(
            index=index,
            fact=fact,
        )

    # ========================================================
    # UNKNOWN 输出
    # ========================================================

    print()
    print("=" * 60)
    print(
        "Unknown Facts"
    )

    if not unknown_facts:

        print(
            "无"
        )

    else:

        for index, fact in enumerate(
            unknown_facts,
            start=1,
        ):

            print_unknown_fact(
                index=index,
                fact=fact,
            )


if __name__ == "__main__":
    main()