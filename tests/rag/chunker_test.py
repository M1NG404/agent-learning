from app.pojo.rag.document import Document
from app.service.rag.impl.markdown_chunker import MarkdownChunker


document = Document(
    id="doc-001",
    filename="尽调报告.md",
    content="""
# 标的物基本情况

建筑面积：12580.36㎡
土地用途：工业用地

# 抵押情况

抵押金额：5000万元
抵押权人：XX银行
""",
)

chunker = MarkdownChunker()

chunks = chunker.split(document)

for chunk in chunks:
    print(chunk)