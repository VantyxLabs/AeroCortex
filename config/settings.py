import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _env(key: str, default: Any = None) -> Any:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class FlightEnvelopeConfig(BaseModel):
    min_battery_percent: float = 15.0
    critical_battery_percent: float = 10.0
    max_altitude_m: float = 150.0
    min_altitude_m: float = 5.0
    max_velocity_mps: float = 20.0
    max_wind_speed_mps: float = 12.0
    max_gps_hdop: float = 3.0
    min_comms_rssi_dbm: float = -85.0
    geofence: Dict[str, float] = Field(default_factory=lambda: {
        "min_lat": 10.9000, "max_lat": 11.1500,
        "min_lon": 76.8500, "max_lon": 77.1000
    })


class HybridWeightsConfig(BaseModel):
    vector_similarity: float = 0.6
    graph_relevance: float = 0.4


class MemoryConfig(BaseModel):
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)
    top_k_episodes: int = 3
    chroma_db_dir: str = "data/knowledge/chroma"
    chroma_collection: str = "aerocortex_episodes"
    semantic_rules_path: str = "data/knowledge/semantic_rules.json"
    knowledge_graph_path: str = "data/knowledge/knowledge_graph.json"
    reinforcement_rate: float = 0.2


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    enabled: bool = False


class MongoConfig(BaseModel):
    uri: str = "mongodb://localhost:27017"
    db_name: str = "aerocortex"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gemma3:latest"
    timeout_seconds: float = 5.0
    connect_timeout_seconds: float = 0.3
    min_confidence_threshold: float = 0.70


class SystemConfig(BaseModel):
    app_name: str = "AeroCortex Cognitive Edge Architecture"
    version: str = "1.0.0"
    log_level: str = "INFO"
    offline_mode: bool = True
    device_target: str = "edge_uav"


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    flight_envelope: FlightEnvelopeConfig = Field(default_factory=FlightEnvelopeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-local-dev-key"
    dashboard_port: int = 8501


def _overlay_env(cfg: AppConfig) -> AppConfig:
    """Environment variables win over config.yaml defaults."""
    cfg.api_host = _env("API_HOST", cfg.api_host)
    cfg.api_port = _env_int("API_PORT", cfg.api_port)
    cfg.api_key = _env("API_KEY", cfg.api_key)

    cfg.mongo.uri = _env("MONGO_URI", cfg.mongo.uri)
    cfg.mongo.db_name = _env("MONGO_DB", cfg.mongo.db_name)

    cfg.neo4j.enabled = _env_bool("NEO4J_ENABLED", cfg.neo4j.enabled)
    cfg.neo4j.uri = _env("NEO4J_URI", cfg.neo4j.uri)
    cfg.neo4j.user = _env("NEO4J_USER", cfg.neo4j.user)
    cfg.neo4j.password = _env("NEO4J_PASSWORD", cfg.neo4j.password)

    cfg.llm.base_url = _env("OLLAMA_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = _env("OLLAMA_MODEL", cfg.llm.model)

    chroma_dir = _env("CHROMA_DIR")
    if chroma_dir:
        cfg.memory.chroma_db_dir = chroma_dir

    cfg.system.offline_mode = _env_bool("OFFLINE_MODE", cfg.system.offline_mode)
    cfg.system.device_target = _env("DEVICE_TARGET", cfg.system.device_target)
    cfg.dashboard_port = _env_int("DASHBOARD_PORT", cfg.dashboard_port)
    return cfg


def load_config(config_file: Optional[Path] = None) -> AppConfig:
    target_file = config_file or CONFIG_PATH
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            api_data = data.get("api", {})
            dash_data = data.get("dashboard", {})
            memory_data = data.get("memory", {})
            cfg = AppConfig(
                system=SystemConfig(**data.get("system", {})),
                flight_envelope=FlightEnvelopeConfig(**data.get("flight_envelope", {})),
                memory=MemoryConfig(**memory_data),
                neo4j=Neo4jConfig(**data.get("neo4j", {})),
                mongo=MongoConfig(**data.get("mongo", {})),
                llm=LLMConfig(**data.get("llm", {})),
                api_host=api_data.get("host", "0.0.0.0"),
                api_port=api_data.get("port", 8000),
                dashboard_port=dash_data.get("port", 8501),
            )
            return _overlay_env(cfg)
        except Exception:
            pass
    return _overlay_env(AppConfig())


config = load_config()
