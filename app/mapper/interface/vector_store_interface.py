from typing import Protocol
class VectorStoreInterface(Protocol):

    def upsert(
            self,
            key:str,
            value:str,
            vector:list[float]
    )->None:
        ...

    def search(
            self,
            query_vector:list[float],
            top_k:int=1
    )->list[dict]:
        ...