from typing import Any

from pydantic import BaseModel


# 表示一份完整文档，后续会被 Chunker 切分成多个 Chunk
class Document(BaseModel):
    id: str
    filename: str
    content: str
    metadata: dict[str, Any] = {}