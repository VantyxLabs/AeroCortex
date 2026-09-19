import os

# Keep unit tests off cloud backends unless a test constructs them directly.
os.environ["NEO4J_ENABLED"] = "false"
os.environ["PINECONE_ENABLED"] = "false"
os.environ["GROQ_ENABLED"] = "false"
os.environ.setdefault("API_KEY", "change-me-local-dev-key")
