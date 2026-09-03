from qdrant_client import QdrantClient, models
import uuid


class QdrantVectorStore:

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str
    ):
        self.client = client
        self.collection_name = collection_name

    # 新增或更新
    def upsert(
        self,
        key: str,
        value: str,
        vector: list[float]
    ) -> None:

        # 1. 业务唯一 key → 稳定的 Qdrant point id
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                key
            )
        )

        # 2. 写入 Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "key": key,
                        "value": value
                    }
                )
            ]
        )

    # search    
    def search(
            self,
            query_vector:list[float],
            top_k:int =1
    )->list[dict]:

        response=self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

        results=[]

        for point in response.points:
            results.append(
                {
                   "key":point.payload["key"],
                   "value":point.payload["value"],
                   "score":point.score
                }
            )
        return results