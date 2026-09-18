import os

# Keep unit tests on the NetworkX fallback unless a test constructs Neo4jBackend directly.
os.environ["NEO4J_ENABLED"] = "false"
os.environ.setdefault("API_KEY", "change-me-local-dev-key")
