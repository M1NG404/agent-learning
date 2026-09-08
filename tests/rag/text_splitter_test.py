from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

text = """
# 标的物基本情况

标的物位于江苏省无锡市江阴市华士镇。
建筑面积为12580.36平方米，土地用途为工业用地。
权证记载所有权人为东华铝材科技有限公司。

# 抵押情况

该资产已抵押给XX银行。
抵押金额为5000万元，抵押期限至2028年12月31日。

# 评估情况

评估机构采用收益法和市场法进行评估。
最终评估价值为8600万元。
"""

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

sections = header_splitter.split_text(text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10,
)

chunks = text_splitter.split_documents(sections)

for index, chunk in enumerate(chunks):
    print(f"\n--- Chunk {index} ---")
    print("metadata:", chunk.metadata)
    print(chunk.page_content)