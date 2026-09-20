# AWS Integration — Phase 0 Notes

Answers to section 4 of `docs/AWS_INTEGRATION_SPEC.md`, with file references and a phase plan.

## Facts verified

### 1. Mission / episode persistence (Mongo)

- **Owner:** `memory/document_store.py` — class `DocumentStore`, singleton via `get_document_store()` (lines 13–23, ~270+).
- **Write path:** `LearningAgent.process_mission_outcome` → `store.insert_episode_sync(episode_doc)` (`agents/learning_agent.py` 100–108), then vector + rules + graph.
- **Episode shape** (`learning_agent.py` 85–98): `mission_id`, `timestamp`, `failure_type`, `telemetry`, `situation`, `retrieved_context`, `plan`, `safety_verdict`, `outcome`, `success`, `planner_source`, `latency_ms`; Mongo `_id` = episode UUID.
- **`GET /missions`:** `api/telemetry_api.py` ~316–326 — `document_store.list_missions(limit, skip)`, returns `{total_events, limit, skip, history, persisted}`.

### 2. Working memory

- **File:** `memory/working_memory.py` — class `WorkingMemory`.
- **State:** `telemetry_history` (deque), `latest_telemetry`, `active_anomaly`, `mission_phase`, `current_waypoint`, `battery_status`, `communication_state`, `gps_state`, `current_recovery_plan`, `safety_status`.
- **Writers:** Situation / graph nodes via `update_telemetry`, `set_anomaly`, `set_recovery_plan`, `set_safety_status` (`orchestration/graph.py`).
- **Readers:** `GET /status` → `graph.working_memory.get_snapshot()`; `POST /reset` → `clear()`.

### 3. Simulator state

- **`MissionSimulator`** (`simulation/mission_simulator.py`): owns `generator`, `graph`, `history` list of step records.
- **`TelemetryGenerator`** (`simulation/telemetry_generator.py` 13–30): `step_idx`, lat/lon/alt/vel/heading, battery, waypoint, mission_state, wind, gps, comms.
- **`POST /reset`:** `graph.working_memory.clear()` + `simulator.reset()` (clears generator + history). Scenario injection is per-request via `FailureScenarioInjector`, not stored long-term.

### 4. Learning Agent side effects (order)

`agents/learning_agent.py` `process_mission_outcome`:

1. Skip if no anomaly.
2. Build `episode_doc`; **Mongo** `insert_episode_sync` (may fail → `persisted=false`).
3. Ensure `episode_id` (UUID if Mongo failed).
4. **Vector** `episodic_memory.store_experience` (Pinecone / Chroma) — upsert by id → **idempotent**.
5. **Semantic rules** `reinforce_action` — **not idempotent** (success/fail counts, confidence).
6. **Knowledge graph** `add_mission_resolution` — **not idempotent** (edge weight reinforcement).

Return: `status`, `episode_id`, `persisted`, action/outcome, `planner_source`, `latency_ms`, `rule_updated`.

### 5. Knowledge graph + semantic rules on disk

- Paths from config: `data/knowledge/knowledge_graph.json`, `data/knowledge/semantic_rules.json` (`config` / `settings`).
- **KG seeds:** in-code `DEFAULT_FAILURES/CONDITIONS/ACTIONS/OUTCOMES/EDGES` in `memory/knowledge_graph.py` (~23–81); NetworkX `_load_or_seed_graph` / `_seed_default_graph` / `save()` via `nx.node_link_data`.
- **Rules seeds:** `SemanticMemory._seed_default_rules()` (~64+) then `save()` JSON list; also may load from Mongo first.

### 6. LLM / planner tiers

- **`GroqClient`** (`llm/groq_client.py`): `generate_json(user_prompt, system)`, `ping()`, `is_configured`; returns `None` on failure.
- **`PlannerAgent.plan_recovery`** (`agents/planner_agent.py`): if Groq configured → Groq (+ one repair retry); else Ollama; else `deterministic_reasoner`. Sets `source` to `groq` | `ollama` | `offline_reasoner`. Validates against `LLMRecoveryPlan` / `RecoveryPlan`.
- **Spec conflict:** spec mentions Groq model `llama-3.3-70b-versatile`; repo uses `openai/gpt-oss-20b` via env/`config.yaml`. **Keep repo model; do not hardcode Bedrock model either.**

### 7. `/healthz` + dashboard

- **API** (`api/telemetry_api.py` `_compute_health` / `healthz`): deps `mongo`, `neo4j`, `pinecone`, `chroma`, `groq`, `ollama`; overall `ok` | `degraded` | `unavailable` (503 if vector and mongo both hard-fail); plus `engine`, `vector_engine`. Cached ~8s.
- **Dashboard** (`dashboard/static/js/app.js` `renderHealth` / `setDep`): classes `ok` / `bad` on strip tiles; shows status strings as returned.

### 8. Dashboard `/simulate` contract

- **Request:** `{ "scenario": "<name>", "steps": 1, "inject_step": 1 }` (`dashboard/static/js/api.js` ~128–132).
- **Response:** `{ "result": { telemetry, situation, planner_plan, safety_verdict, final_plan, memory_context, graph_paths?, knowledge_graph?, execution_status, logs, ... } }` (`simulation/mission_simulator.py` step_record).

### 9. Dashboard config

- Local: `dashboard/server.py` `GET /config.json` → `api_base_url`, `upstream_api_url`, `api_key`, `default_scenarios`, `poll_interval_ms`, `version`.
- Vercel: `scripts/vercel-prepare.js` writes `public/config.json` same shape (`api_base_url: "/api"`, env-baked key/upstream).

### 10. Settings load

- `config/settings.py`: `load_dotenv(PROJECT_ROOT / ".env")` at import (~12); YAML then env overrides (`_env`, `_env_bool`, …).
- **SSM must run before** `from config import config` / `from api.telemetry_api import app` inside `lambda_handler.py`.

### 11. Import-time singletons (Lambda cold start)

- `api/telemetry_api.py` 24–26: `graph = AeroCortexGraph()`, `simulator = MissionSimulator(graph=graph)`, `document_store = get_document_store()` at **module import**.
- `AeroCortexGraph` constructs Memory/Planner/… which construct Pinecone/Chroma/Neo4j/Groq.
- **Phase 1 must introduce lazy getters** for graph/simulator/document_store without breaking local uvicorn.

### 12. `requirements.txt` runtime vs Lambda

- **Keep for Lambda API:** fastapi, uvicorn (optional), pydantic, pyyaml, python-dotenv, networkx, neo4j (optional), pymongo/motor (local only — AWS uses Dynamo), langgraph, langchain-core, numpy, httpx, requests, pinecone, mangum, boto3, aws-lambda-powertools, aws-xray-sdk.
- **Exclude from Lambda image:** streamlit, chromadb (lazy optional), plotly, pandas (if unused by API), pytest.

## Spec adjustments

1. Planner default model string in prose ≠ repo — follow env/`GROQ_MODEL`.
2. Health already has caching; DynamoDB/Bedrock keys are additive.
3. Module-level `graph`/`simulator` need lazy init before packaging for Lambda.
4. Learning already returns `persisted`; add optional `learning_mode` in async path.
5. Neo4j Aura remains optional; AWS path uses NetworkX + S3 snapshots as primary graph durability.

## Phase plan (files touched)

| Phase | New files | Modify |
|-------|-----------|--------|
| 0 | `docs/AWS_PHASE0_NOTES.md` | — |
| 1 | `requirements-lambda.txt`, `Dockerfile.lambda`, `lambda_handler.py`, `cloud/secrets.py`, `infra/template.yaml` (API+HttpApi), `docs/AWS_DEPLOY.md` start | `memory/vector_store.py` (lazy chroma), `api/telemetry_api.py` (lazy graph/sim) |
| 2 | `cloud/dynamo_*.py`, `cloud/factories.py`, `tests/aws/test_dynamo*.py` | document_store usage via factory, working_memory, simulator, healthz, dashboard tiles |
| 3 | `cloud/s3_snapshots.py`, tests | knowledge_graph.py, semantic_memory.py |
| 4 | `llm/bedrock_client.py`, tests | planner_agent.py, healthz, settings |
| 5 | `cloud/sqs_episodes.py`, `learning_handler.py`, tests | learning_agent / graph learning node |
| 6 | `infra/cloudfront-strip-api.js`, `scripts/deploy_dashboard.sh` | template.yaml CloudFront |
| 7 | `observability.py` | agents emit metrics; CW dashboard in SAM |
| 8 | finish `docs/AWS_DEPLOY.md`, README link | `.env.example` |

**Acceptance for Phase 0:** this document.
