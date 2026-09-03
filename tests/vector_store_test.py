from memory.impl.in_memory_vector_store import InMemoryVectorStore


vector_store = InMemoryVectorStore()

vector_store.upsert(
    key="name",
    value="小明",
    vector=[1.0, 0.0]
)

vector_store.upsert(
    key="language",
    value="Java",
    vector=[0.9, 0.1]
)

vector_store.upsert(
    key="food",
    value="火锅",
    vector=[0.0, 1.0]
)

results = vector_store.search(
    query_vector=[1.0, 0.0],
    top_k=2
)

print(results)