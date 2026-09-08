
from typing import Protocol

from app.pojo.rag.chunk import Chunk
from app.pojo.rag.document import Document

# 定义文档切分器统一接口
class ChunkerInterface(Protocol):

    def split(
            self,
            ducument:Document
    )->list[Chunk]:
        ...