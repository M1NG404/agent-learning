import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

from app.service.embedding_service import EmbeddingService


load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

embedding_service = EmbeddingService(
    client=openai_client
)

qdrant_client = QdrantClient(
    url="http://localhost:6333",
    check_compatibility=False
)


# 1. Memory 文本
memory_text = "favorite_language: Python"


# 2. 文本 → Vector
vector = embedding_service.embed(
    memory_text
)


# 3. 写入 Qdrant
qdrant_client.upsert(
    collection_name="agent_memory",
    points=[
        models.PointStruct(
            id=1,
            vector=vector,
            payload={
                "key": "favorite_language",
                "value": "Python"
            }
        )
    ]
)

print("写入成功")

query_vector=embedding_service.embed(
    "我最喜欢的编程语言"
)

results=qdrant_client.query_points(
    collection_name="agent_memory",
    query=query_vector,
    limit=3
)

print(results)