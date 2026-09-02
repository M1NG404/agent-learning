import os

from dotenv import load_dotenv
from openai import OpenAI
from memory.similarity import cosine_similarity
from memory.embedding import EmbeddingService


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL")
)

embedding_service = EmbeddingService(
    client=client
)

query = "我平时主要用什么开发语言？"

memory_a = "用户喜欢使用 Java 编程"
memory_b = "用户叫小明"
memory_c = "用户喜欢吃火锅"


query_vector=embedding_service.embed(query)
   
memory_a_vector = embedding_service.embed(memory_a)
memory_b_vector = embedding_service.embed(memory_b)
memory_c_vector = embedding_service.embed(memory_c)


score_a = cosine_similarity(
    query_vector,
    memory_a_vector
)

score_b = cosine_similarity(
    query_vector,
    memory_b_vector
)

score_c = cosine_similarity(
    query_vector,
    memory_c_vector
)


print("Java Memory:", score_a)
print("Name Memory:", score_b)
print("Food Memory:", score_c)