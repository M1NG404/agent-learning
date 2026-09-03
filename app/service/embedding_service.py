from openai import OpenAI


class EmbeddingService:
    def __init__(self, client: OpenAI):
        self.client = client

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text
        )

        return response.data[0].embedding