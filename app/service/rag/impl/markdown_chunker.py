from app.pojo.rag.chunk import Chunk
from app.pojo.rag.document import Document
from app.service.rag.interface.chunker_interface import ChunkerInterface

class MarkdownChunker(ChunkerInterface):

    def split(
    self,
    document: Document,
) -> list[Chunk]:

        chunks = []
        current_title = ""
        current_lines = []

        for line in document.content.splitlines():

            # 遇到 Markdown 标题，说明新章节开始
            if line.startswith("#"):

                # 先保存上一个章节
                if current_lines:
                    content = "\n".join(current_lines).strip()

                    if content:
                        chunks.append(
                            Chunk(
                                id=f"{document.id}-chunk-{len(chunks)}",
                                document_id=document.id,
                                content=content,
                                index=len(chunks),
                                metadata={
                                    **document.metadata,
                                    "filename": document.filename,
                                    "title": current_title,
                                },
                            )
                        )

                # 更新当前章节
                current_title = line.lstrip("#").strip()
                current_lines = []

            else:
                current_lines.append(line)

        # 循环结束后，最后一个章节还没有被保存
        if current_lines:
            content = "\n".join(current_lines).strip()

            if content:
                chunks.append(
                    Chunk(
                        id=f"{document.id}-chunk-{len(chunks)}",
                        document_id=document.id,
                        content=content,
                        index=len(chunks),
                        metadata={
                            **document.metadata,
                            "filename": document.filename,
                            "title": current_title,
                        },
                    )
                )

        return chunks