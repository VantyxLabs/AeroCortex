"""Idempotent MongoDB seed for indexes and semantic rules.

Loads `data/knowledge/semantic_rules.json` into the `rules` collection when it
is empty (or when --force is passed). JSON remains the fallback seed file.

Usage:
    python scripts/seed_mongo.py
    python scripts/seed_mongo.py --force
    python scripts/seed_mongo.py --reset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as CONFIG_ROOT, config  # noqa: E402
from memory.document_store import DocumentStore  # noqa: E402


def _rules_path() -> Path:
    return CONFIG_ROOT / config.memory.semantic_rules_path


def _rule_doc(item: dict) -> dict:
    rule_id = str(item.get("rule_id") or item.get("_id") or "")
    success = int(item.get("success_count") or 0)
    failure = int(item.get("failure_count") or 0)
    return {
        "_id": rule_id,
        "rule_id": rule_id,
        "if_condition": item.get("if_condition") or item.get("trigger"),
        "then_action": item.get("then_action") or item.get("action"),
        "confidence": item.get("confidence", 0.8),
        "hits": item.get("hits", success + failure),
        "source": item.get("source", "json_seed"),
        "trigger": item.get("trigger"),
        "condition": item.get("condition"),
        "action": item.get("action"),
        "success_count": item.get("success_count", 1),
        "failure_count": item.get("failure_count", 0),
        "description": item.get("description", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed AeroCortex MongoDB indexes and rules")
    parser.add_argument("--reset", action="store_true", help="Drop episodes, missions, and rules first")
    parser.add_argument("--force", action="store_true", help="Re-upsert rules even if the collection is not empty")
    args = parser.parse_args()

    store = DocumentStore()
    try:
        if not store.ping_sync():
            print("MongoDB is unreachable; refusing to seed.", file=sys.stderr)
            return 1

        db = store._db_sync()
        if db is None:
            print("MongoDB is unreachable; refusing to seed.", file=sys.stderr)
            return 1

        if args.reset:
            db.episodes.delete_many({})
            db.missions.delete_many({})
            db.rules.delete_many({})
            print("Reset: dropped episodes, missions, and rules")

        store.ensure_indexes_sync()

        existing = store.get_rules_sync()
        seeded = 0
        if args.force or args.reset or not existing:
            path = _rules_path()
            if not path.exists():
                print(f"Rules seed file missing: {path}", file=sys.stderr)
                return 1
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                doc = _rule_doc(item)
                if not doc["_id"]:
                    continue
                store.upsert_rule_sync(doc)
                seeded += 1

        rules = store.get_rules_sync()
        missions = store.count_missions_sync()
        episodes = db.episodes.count_documents({})
        print(
            "Mongo seed complete: "
            f"{episodes} episodes, {missions} missions, {len(rules)} rules"
            + (f" ({seeded} upserted)" if seeded else " (rules already present)")
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
