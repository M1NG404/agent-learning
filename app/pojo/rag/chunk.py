from typing import Any

from pydantic import BaseModel


# RAG 检索中的最小文本单元
class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    index: int
    metadata: dict[str, Any] = {}