"""Idempotent Neo4j ontology seed.

Usage:
    python scripts/seed_neo4j.py
    python scripts/seed_neo4j.py --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.knowledge_graph import Neo4jBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the AeroCortex Neo4j ontology")
    parser.add_argument("--reset", action="store_true", help="DETACH DELETE all nodes first")
    args = parser.parse_args()

    backend = Neo4jBackend()
    try:
        summary = backend.seed_ontology(reset=args.reset)
        print(f"Neo4j seed complete: {summary['nodes']} nodes, {summary['edges']} edges")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
