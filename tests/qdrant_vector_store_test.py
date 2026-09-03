import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from memory.embedding import EmbeddingService
from memory.impl.qdrant_vector_store import QdrantVectorStore


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

from qdrant_client import QdrantClient, models


vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="agent_memory"
)


# 1. 生成 Memory 向量
memory_vector = embedding_service.embed(
    "favorite_language: Python"
)

# 2. 写入
vector_store.upsert(
    key="favorite_language",
    value="Python",
    vector=memory_vector
)

# 3. 生成 Query 向量
query_vector = embedding_service.embed(
    "我最喜欢什么编程语言？"
)

# 4. 搜索
results = vector_store.search(
    query_vector=query_vector,
    top_k=3
)

print(results)