from pathlib import Path
import re

BASE_DIR = Path(__file__).parent

INPUT_DIR = (
    BASE_DIR.parent
    / "fact_extraction"
    / "input"
)

def find_relation_end_line(
    text: str,
    start_line: int,
) -> int:
    """
    从关系主句开始，自动向后寻找属于它的上下文。

    第一版规则：

    1. start_line 本身属于关系。
    2. 空行允许存在。
    3. Markdown 列表项 "- ..." 属于当前关系。
    4. 遇到下一个中文编号段落，例如：
       2、
       3、
       则停止。
    """

    lines = text.splitlines()

    end_line = start_line

    for index in range(
        start_line,
        len(lines),
    ):
        line = lines[index].strip()

        current_line_number = index + 1

        # 空行先跳过
        if not line:
            continue

        # ----------------------------------------------------
        # Markdown 列表项
        # ----------------------------------------------------

        if line.startswith("-"):
            end_line = current_line_number
            continue

        # ----------------------------------------------------
        # 遇到下一个编号段落
        #
        # 例如：
        # 2、本公司授权……
        # ----------------------------------------------------

        if re.match(
            r"^\d+[、.]",
            line,
        ):
            break

        # 其他文本暂时结束 Context
        break

    return end_line

def extract_line_range(
    text: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    从原始 Markdown 中提取指定行范围。

    行号从 1 开始。
    """

    lines = text.splitlines()

    # Python list 下标从 0 开始
    selected = lines[
        start_line - 1:end_line
    ]

    return "\n".join(selected)


def main():

    files = sorted(
        INPUT_DIR.glob("*.md")
    )

    if not files:
        print("没有找到 Markdown")
        return

    file_path = files[0]

    raw_text = file_path.read_text(
        encoding="utf-8"
    )

    start_line = 10

    end_line = find_relation_end_line(
        text=raw_text,
        start_line=start_line,
    )

    context = extract_line_range(
        text=raw_text,
        start_line=start_line,
        end_line=end_line,
    )

    print("=" * 60)
    print(
        f"Relation Context："
        f"L{start_line:04d} ~ L{end_line:04d}"
    )

    print()
    print(context)


if __name__ == "__main__":
    main()