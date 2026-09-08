from dataclasses import dataclass
from typing import Optional


@dataclass
class SubjectCandidate:
    subject_name: Optional[str]
    subject_status: str
    resolution_method: str
    confidence: str
    reason: str


def verify_subject(
    evidence: str,
    predicate: str,
    llm_subject: Optional[str] = None,
    resolved_reference: Optional[str] = None,
    block_owner: Optional[str] = None,
    block_ambiguous: bool = False,
) -> SubjectCandidate:
    """
    第一版 Subject Verification。

    核心原则：
    1. 不强制一定解析出主体
    2. 明确证据优先
    3. Block Ownership 只能作为 LOW confidence fallback
    4. 有歧义就 UNKNOWN
    """

    # ============================================================
    # 1. 明确的“本公司”
    # ============================================================

    if "本公司" in evidence:
        if resolved_reference:
            return SubjectCandidate(
                subject_name=resolved_reference,
                subject_status="RESOLVED",
                resolution_method="ENTITY_RESOLUTION",
                confidence="HIGH",
                reason="Evidence contains '本公司' and the reference is explicitly resolved.",
            )

        return SubjectCandidate(
            subject_name=None,
            subject_status="UNRESOLVED",
            resolution_method="NONE",
            confidence="LOW",
            reason="Evidence contains '本公司' but no resolved reference is available.",
        )

    # ============================================================
    # 2. Predicate 语义角色验证
    # ============================================================

    if predicate == "GUARANTEE":
        """
        当前实验只验证非常明确的模式：

        A公司为B公司提供担保
        B公司的债务由A公司提供担保

        不能仅因为某公司名出现在 evidence 中
        就认定它是 guarantor。
        """

        if "提供担保" in evidence or "提供连带保证" in evidence:

            if llm_subject and llm_subject in evidence:
                # 简单检查：
                # subject 是否出现在“提供担保”之前。
                subject_pos = evidence.find(llm_subject)

                guarantee_pos = evidence.find("提供担保")

                if guarantee_pos == -1:
                    guarantee_pos = evidence.find("提供连带保证")

                if (
                    subject_pos != -1
                    and guarantee_pos != -1
                    and subject_pos < guarantee_pos
                ):
                    return SubjectCandidate(
                        subject_name=llm_subject,
                        subject_status="RESOLVED",
                        resolution_method="SEMANTIC_ROLE",
                        confidence="HIGH",
                        reason="The entity appears in the evidence and is positioned as the guarantor.",
                    )

    # ============================================================
    # 3. 明确主体，且 evidence 中只有一个候选
    # ============================================================

    if llm_subject and llm_subject in evidence:
        return SubjectCandidate(
            subject_name=llm_subject,
            subject_status="RESOLVED",
            resolution_method="DIRECT",
            confidence="MEDIUM",
            reason="The subject explicitly appears in the evidence, but semantic role validation is limited.",
        )

    # ============================================================
    # 4. Block Ownership
    # ============================================================

    if block_owner:
        if block_ambiguous:
            return SubjectCandidate(
                subject_name=None,
                subject_status="UNRESOLVED",
                resolution_method="NONE",
                confidence="LOW",
                reason="The block contains multiple possible subjects, so block ownership is unsafe.",
            )

        return SubjectCandidate(
            subject_name=block_owner,
            subject_status="RESOLVED",
            resolution_method="BLOCK_OWNERSHIP",
            confidence="LOW",
            reason="No explicit subject is available; inferred only from document block ownership.",
        )

    # ============================================================
    # 5. 无法判断
    # ============================================================

    return SubjectCandidate(
        subject_name=None,
        subject_status="UNRESOLVED",
        resolution_method="NONE",
        confidence="LOW",
        reason="Insufficient evidence to determine the subject.",
    )


CASES = [
    {
        "name": "Case 1：单一明确主体",
        "evidence": "A公司提供担保。",
        "predicate": "GUARANTEE",
        "llm_subject": "A公司",
        "expected_subject": "A公司",
        "expected_status": "RESOLVED",
    },
    {
        "name": "Case 2：本公司",
        "evidence": "本公司提供担保。",
        "predicate": "GUARANTEE",
        "llm_subject": "本公司",
        "resolved_reference": "A公司",
        "expected_subject": "A公司",
        "expected_status": "RESOLVED",
    },
    {
        "name": "Case 3：两个公司同时出现",
        "evidence": "A公司的债务由B公司提供担保。",
        "predicate": "GUARANTEE",
        "llm_subject": "B公司",
        "expected_subject": "B公司",
        "expected_status": "RESOLVED",
    },
    {
        "name": "Case 4：隐式主体，只能依赖 Block",
        "evidence": "- A、某土地使用权",
        "predicate": "AREA",
        "block_owner": "A公司",
        "expected_subject": "A公司",
        "expected_status": "RESOLVED",
    },
    {
        "name": "Case 5：Block 内存在多个可能主体",
        "evidence": "- A、某土地使用权",
        "predicate": "AREA",
        "block_owner": "A公司",
        "block_ambiguous": True,
        "expected_subject": None,
        "expected_status": "UNRESOLVED",
    },
    {
        "name": "Case 6：孤立日期",
        "evidence": "2018年12月19日",
        "predicate": "MEETING_DATE",
        "expected_subject": None,
        "expected_status": "UNRESOLVED",
    },
]


def main():
    print("=" * 60)
    print("Subject Verification Experiment")
    print(f"Cases：{len(CASES)}")

    passed = 0

    for index, case in enumerate(CASES, start=1):
        print()
        print("=" * 60)
        print(case["name"])

        result = verify_subject(
            evidence=case["evidence"],
            predicate=case["predicate"],
            llm_subject=case.get("llm_subject"),
            resolved_reference=case.get("resolved_reference"),
            block_owner=case.get("block_owner"),
            block_ambiguous=case.get("block_ambiguous", False),
        )

        print()
        print("Evidence：")
        print(case["evidence"])

        print()
        print("Result：")
        print(f"subject_name：{result.subject_name}")
        print(f"subject_status：{result.subject_status}")
        print(f"resolution_method：{result.resolution_method}")
        print(f"confidence：{result.confidence}")
        print(f"reason：{result.reason}")

        subject_ok = (
            result.subject_name
            == case["expected_subject"]
        )

        status_ok = (
            result.subject_status
            == case["expected_status"]
        )

        if subject_ok and status_ok:
            print()
            print("Validation：通过")
            passed += 1
        else:
            print()
            print("Validation：失败")

            print(
                f"Expected Subject："
                f"{case['expected_subject']}"
            )

            print(
                f"Expected Status："
                f"{case['expected_status']}"
            )

    print()
    print("=" * 60)
    print("实验统计")
    print(f"总 Case：{len(CASES)}")
    print(f"通过：{passed}")
    print(f"失败：{len(CASES) - passed}")


if __name__ == "__main__":
    main()