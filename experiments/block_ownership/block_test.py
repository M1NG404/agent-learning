from dataclasses import dataclass


@dataclass
class EntityBlock:
    """
    表示一个主体负责的文档区间。

    例如：
    L0007 ~ L0038
    属于江阴利泰装饰材料有限公司
    """
    start_line: int
    end_line: int
    entity: str


def find_owner(line: int, blocks: list[EntityBlock]) -> str:
    """
    根据 Fact 所在行号，找到它所属的主体。

    找不到则返回 UNKNOWN。
    """

    for block in blocks:
        if block.start_line <= line <= block.end_line:
            return block.entity

    return "UNKNOWN"


def main():
    # 先手工构造我们上一轮已经观察到的主体区间
    blocks = [
        EntityBlock(
            start_line=7,
            end_line=38,
            entity="江阴利泰装饰材料有限公司",
        ),
        EntityBlock(
            start_line=39,
            end_line=75,
            entity="江苏海达科技集团有限公司",
        ),
        EntityBlock(
            start_line=76,
            end_line=112,
            entity="江阴海达特种人革有限公司",
        ),
        EntityBlock(
            start_line=113,
            end_line=143,
            entity="江阴海达彩涂有限公司",
        ),
    ]

    # 专门测试上一轮 UNKNOWN 的几个行号
    test_lines = [
        12,   # 土地 A
        13,   # 土地 B
        14,   # 房产 C
        68,   # 日期
        125,  # 股东签字真实性
        139,  # 决议日期
    ]

    print("Block Ownership 测试")
    print("=" * 60)

    for line in test_lines:
        owner = find_owner(
            line=line,
            blocks=blocks,
        )

        print(
            f"L{line:04d} -> {owner}"
        )

def build_blocks(
    mentions: list[dict],
    total_lines: int,
) -> list[EntityBlock]:
    """
    根据 Entity Resolution 的 definition_line，
    自动构建主体区间。
    """

    # definition_line -> entity
    definitions = {}

    for mention in mentions:
        definition_line = mention.get("definition_line")
        entity = mention.get("resolved_entity")

        if (
            definition_line is not None
            and entity
            and entity != "UNKNOWN"
        ):
            # 同一个 definition_line 可能出现多次，
            # dict 可以自动去重
            definitions[definition_line] = entity

    # 按定义出现的行号排序
    sorted_definitions = sorted(
        definitions.items()
    )

    blocks = []

    for index, (start_line, entity) in enumerate(
        sorted_definitions
    ):
        # 如果后面还有另一个主体，
        # 当前 block 到下一个主体前一行结束
        if index + 1 < len(sorted_definitions):
            next_start_line = sorted_definitions[index + 1][0]
            end_line = next_start_line - 1
        else:
            # 最后一个主体一直到文档结尾
            end_line = total_lines

        blocks.append(
            EntityBlock(
                start_line=start_line,
                end_line=end_line,
                entity=entity,
            )
        )

    return blocks

if __name__ == "__main__":
    main()