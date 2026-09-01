

from pydantic import BaseModel


class SaveMemoryArgs(BaseModel):
    key: str
    value: str