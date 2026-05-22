from memories.embedder import Embedding, OpenRouterEmbedder


def test_openrouter_embedder_posts_documented_embeddings_payload() -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        calls.append((url, payload, headers, timeout))
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "openai/text-embedding-3-small",
        }

    embedder = OpenRouterEmbedder(
        api_key="secret",
        model="openai/text-embedding-3-small",
        dimensions=3,
        post_json=post_json,
        timeout=4.0,
    )

    embedding = embedder.embed("hello")

    assert embedding == Embedding(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        vector=[0.1, 0.2, 0.3],
    )
    assert calls == [
        (
            "https://openrouter.ai/api/v1/embeddings",
            {
                "input": "hello",
                "model": "openai/text-embedding-3-small",
                "dimensions": 3,
            },
            {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            4.0,
        )
    ]
