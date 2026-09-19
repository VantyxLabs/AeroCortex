/** Live mission control — Minimalist Monochrome SPA. */
(function () {
  const state = {
    history: [],
    lastResult: null,
    pollTimer: null,
    busy: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value == null || value === "" ? "—" : String(value);
  }

  function setError(msg) {
    const el = $("conn-error");
    if (!el) return;
    if (!msg) {
      el.textContent = "";
      el.classList.remove("visible");
      return;
    }
    el.textContent = msg;
    el.classList.add("visible");
  }

  function fmt(n, digits) {
    if (digits === undefined) digits = 1;
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
    return Number(n).toFixed(digits);
  }

  function depClass(value) {
    const v = String(value || "").toLowerCase();
    if (v === "ok") return "ok";
    if (!v || v === "—" || v === "unset") return "";
    return "bad";
  }

  function setDep(id, value) {
    const el = $(id);
    if (!el) return;
    el.textContent = value || "—";
    el.classList.remove("ok", "bad");
    const c = depClass(value);
    if (c) el.classList.add(c);
  }

  function renderHealth(h) {
    if (!h) return;
    setText("sys-status", h.status || "—");
    const d = h.dependencies || {};
    setDep("dep-mongo", d.mongo);
    setDep("dep-neo4j", d.neo4j);
    setDep("dep-pinecone", d.pinecone);
    setDep("dep-groq", d.groq);
    setDep("dep-vector", h.vector_engine || d.chroma);
    setDep("dep-engine", h.engine);
  }

  function renderKpis(t) {
    if (!t) {
      setText("kpi-mission", "—");
      setText("kpi-state", "—");
      setText("kpi-alt", "—");
      setText("kpi-vel", "—");
      setText("kpi-bat", "—");
      setText("kpi-bat-sub", "—");
      setText("kpi-wind", "—");
      return;
    }
    setText("kpi-mission", t.mission_id || "—");
    setText("kpi-state", t.mission_state || "—");
    setText("kpi-alt", fmt(t.altitude));
    setText("kpi-vel", fmt(t.velocity));
    const bat = fmt(t.battery_level);
    setText("kpi-bat", bat === "—" ? "—" : bat + "%");
    const volts = fmt(t.battery_voltage, 2);
    setText("kpi-bat-sub", volts === "—" ? "—" : volts + " V");
    setText("kpi-wind", fmt(t.wind_speed));
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderSituation(sit) {
    const el = $("panel-situation");
    if (!el) return;
    el.classList.remove("inverted");
    if (!sit) {
      el.innerHTML =
        "<h4>Nominal</h4><p class=\"muted-note\">Awaiting mission step. Deterministic detectors idle.</p>";
      return;
    }
    if (sit.anomaly_detected) {
      el.classList.add("inverted");
      el.innerHTML =
        "<h4>" +
        escapeHtml(sit.failure_type || "ANOMALY") +
        " · " +
        escapeHtml(sit.severity || "") +
        "</h4><p>" +
        escapeHtml(sit.description || "") +
        '</p><p class="meta">Confidence ' +
        fmt(sit.confidence, 2) +
        " · Phase " +
        escapeHtml(sit.mission_phase || "—") +
        "</p>";
    } else {
      el.innerHTML =
        "<h4>Nominal — all channels normal</h4><p>Deterministic rules report zero flight envelope threshold violations.</p>";
    }
  }

  function renderPlanner(plan) {
    const el = $("panel-planner");
    if (!el) return;
    if (!plan) {
      el.classList.remove("inverted");
      el.innerHTML =
        "<p class=\"muted-note\">Execute a mission step to surface a recovery plan.</p>";
      return;
    }
    const steps = (plan.steps || []).map(escapeHtml).join(" · ");
    el.innerHTML =
      '<h4>Action <span class="code">' +
      escapeHtml(plan.action || "—") +
      "</span></h4><p>" +
      escapeHtml(plan.reason || "") +
      "</p><p><strong>Steps.</strong> " +
      (steps || "—") +
      '</p><p class="meta">Confidence ' +
      fmt(plan.confidence, 2) +
      " · Risk " +
      escapeHtml(plan.risk_level || "—") +
      " · Source " +
      escapeHtml(plan.source || "—") +
      "</p>";
  }

  function renderSafety(safety, finalPlan) {
    const el = $("panel-safety");
    if (!el) return;
    el.classList.remove("inverted");
    if (!safety) {
      el.innerHTML =
        "<p class=\"muted-note\">Safety gatekeeper awaits planner output.</p>";
      return;
    }
    const approved = !!safety.approved;
    if (!approved) el.classList.add("inverted");
    const violated = (safety.violated_constraints || []).join(", ") || "None";
    el.innerHTML =
      "<h4>Verdict " +
      (approved ? "Approved" : "Rejected") +
      "</h4><p>" +
      escapeHtml(safety.reason || "") +
      "</p><p><strong>Fallback.</strong> <span class=\"code\">" +
      escapeHtml(safety.fallback_action || "—") +
      "</span></p><p><strong>Executed.</strong> <span class=\"code\">" +
      escapeHtml((finalPlan && finalPlan.action) || "—") +
      '</span></p><p class="meta">Constraints ' +
      escapeHtml(violated) +
      "</p>";
  }

  function renderMemory(memoryPayload, lastResult) {
    const el = $("panel-memory");
    if (!el) return;
    const candidates =
      (memoryPayload &&
        memoryPayload.last_retrieval &&
        memoryPayload.last_retrieval.candidates) ||
      [];
    const fromStep =
      (lastResult &&
        lastResult.memory_context &&
        lastResult.memory_context.retrieved_experiences) ||
      [];

    let rows = [];
    if (fromStep.length) {
      rows = fromStep.slice(0, 5).map(function (item) {
        const exp = item.experience || item;
        return {
          mission: exp.mission_id || item.episode_id || "—",
          failure: exp.failure || "—",
          action: exp.action || item.action || "—",
          outcome: exp.outcome || "—",
          score: item.final_score != null ? item.final_score : item.vector_similarity,
        };
      });
    } else if (candidates.length) {
      rows = candidates.slice(0, 5).map(function (c) {
        return {
          mission: c.episode_id || "—",
          failure: "—",
          action: c.action || "—",
          outcome: "—",
          score: c.final_score != null ? c.final_score : c.vector_similarity,
        };
      });
    }

    if (!rows.length) {
      el.innerHTML = "<p class=\"muted-note\">No retrieved episodes yet.</p>";
      return;
    }

    el.innerHTML =
      '<div class="table-wrap"><table class="data"><thead><tr>' +
      "<th>Mission</th><th>Failure</th><th>Action</th><th>Outcome</th><th>Score</th>" +
      "</tr></thead><tbody>" +
      rows
        .map(function (r) {
          const score =
            typeof r.score === "number" ? r.score.toFixed(3) : String(r.score || "—");
          return (
            "<tr><td>" +
            escapeHtml(String(r.mission)) +
            "</td><td>" +
            escapeHtml(String(r.failure)) +
            '</td><td class="code">' +
            escapeHtml(String(r.action)) +
            "</td><td>" +
            escapeHtml(String(r.outcome)) +
            '</td><td class="code">' +
            escapeHtml(score) +
            "</td></tr>"
          );
        })
        .join("") +
      "</tbody></table></div>";
  }

  function renderKg(memoryPayload, scenario, stepOverride) {
    const el = $("panel-kg");
    if (!el) return;
    const kg =
      (stepOverride && stepOverride.knowledge_graph) ||
      (memoryPayload && memoryPayload.knowledge_graph) ||
      {};
    let paths =
      (stepOverride && stepOverride.graph_paths) ||
      (stepOverride &&
        stepOverride.memory_context &&
        stepOverride.memory_context.graph_paths) ||
      (state.lastResult && state.lastResult.graph_paths) ||
      (state.lastResult &&
        state.lastResult.memory_context &&
        state.lastResult.memory_context.graph_paths) ||
      (memoryPayload &&
        memoryPayload.last_retrieval &&
        memoryPayload.last_retrieval.graph_paths) ||
      [];

    if (!Array.isArray(paths)) paths = [];

    if (paths.length) {
      el.innerHTML =
        '<ul class="kg-list">' +
        paths
          .slice(0, 5)
          .map(function (p) {
            return (
              '<li><span class="code">' +
              escapeHtml(p.action || "—") +
              '</span><div class="score">Relevance ' +
              fmt(p.graph_relevance, 3) +
              " · Scenario " +
              escapeHtml(scenario || "—") +
              " · Engine " +
              escapeHtml(String(kg.engine || "—")) +
              "</div></li>"
            );
          })
          .join("") +
        "</ul>";
      return;
    }

    el.innerHTML =
      '<p class="muted-note">No graph paths for this step yet. Engine ' +
      escapeHtml(String(kg.engine || "—")) +
      " · nodes " +
      escapeHtml(String(kg.nodes_count != null ? kg.nodes_count : "—")) +
      ".</p>";
  }

  function renderStats(memoryPayload) {
    if (!memoryPayload) return;
    setText("stat-episodes", memoryPayload.episodic_experiences_count != null ? memoryPayload.episodic_experiences_count : 0);
    setText("stat-rules", memoryPayload.semantic_rules_count != null ? memoryPayload.semantic_rules_count : 0);
    const kg = memoryPayload.knowledge_graph || {};
    setText("stat-nodes", kg.nodes_count != null ? kg.nodes_count : 0);
    setText("stat-edges", kg.edges_count != null ? kg.edges_count : 0);
    setText(
      "stat-vector",
      "Vector " + (memoryPayload.vector_engine || kg.engine_id || "—")
    );
  }

  function renderLogs(logs) {
    const el = $("log-body");
    if (!el) return;
    if (!logs || !logs.length) {
      el.textContent = "AeroCortex cognitive system initialized. Ready for mission step.";
      return;
    }
    el.textContent = logs.join("\n");
    const details = el.closest("details");
    if (details) details.open = true;
  }

  function paintCharts() {
    if (!window.AeroCharts || !state.history.length) return;
    AeroCharts.plot($("chart-battery"), [
      { values: state.history.map(function (h) { return Number(h.battery_level); }) },
    ]);
    AeroCharts.plot($("chart-altvel"), [
      { values: state.history.map(function (h) { return Number(h.altitude); }) },
      { values: state.history.map(function (h) { return Number(h.velocity); }) },
    ]);
    AeroCharts.plot($("chart-windgps"), [
      { values: state.history.map(function (h) { return Number(h.wind_speed); }) },
      { values: state.history.map(function (h) { return Number(h.gps_accuracy); }) },
    ]);
  }

  function renderCharts() {
    const empty = $("chart-empty");
    const grid = $("chart-grid");
    if (!empty || !grid || !window.AeroCharts) return;
    if (!state.history.length) {
      empty.hidden = false;
      grid.hidden = true;
      return;
    }
    empty.hidden = true;
    grid.hidden = false;
    // Wait a frame so un-hidden canvases have real CSS width before drawing.
    requestAnimationFrame(function () {
      paintCharts();
    });
  }

  function applyStepResult(step) {
    state.lastResult = step;
    const t = step.telemetry || {};
    state.history.push(t);
    if (state.history.length > 100) state.history.shift();
    renderKpis(t);
    renderSituation(step.situation);
    renderPlanner(step.planner_plan);
    renderSafety(step.safety_verdict, step.final_plan);
    renderLogs(step.logs);
    renderMemory(null, step);
    renderKg(null, ($("scenario") && $("scenario").value) || "", step);
    renderCharts();
  }

  async function refreshHealth() {
    const h = await AeroAPI.healthz();
    renderHealth(h);
    return h;
  }

  async function refreshMemory() {
    const mem = await AeroAPI.memory();
    renderStats(mem);
    renderMemory(mem, state.lastResult);
    renderKg(mem, ($("scenario") && $("scenario").value) || "", state.lastResult);
    return mem;
  }

  function persistForm() {
    const url = window.AeroAPI ? AeroAPI.FIXED_BASE : "/api";
    if ($("api-url")) $("api-url").value = url;
    if (window.AeroAPI) {
      AeroAPI.saveSettings(url, ($("api-key") && $("api-key").value) || "");
    }
  }

  function unwrapStep(data) {
    if (!data) return null;
    if (data.result && data.result.telemetry) return data.result;
    if (data.telemetry) return data;
    if (Array.isArray(data.results) && data.results.length) {
      const last = data.results[data.results.length - 1];
      if (last && last.telemetry) return last;
    }
    return data.result || data;
  }

  async function onStep() {
    if (state.busy) return;
    if (!window.AeroAPI) {
      setError("AeroAPI missing — hard-refresh (Ctrl+Shift+R)");
      return;
    }
    persistForm();
    state.busy = true;
    const btn = $("btn-step");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Running...";
    }
    setError("Calling /api/simulate...");
    try {
      const data = await AeroAPI.simulate(
        ($("scenario") && $("scenario").value) || "GPS_INTERFERENCE",
        1
      );
      const step = unwrapStep(data);
      if (!step || !step.telemetry) {
        throw new Error(
          "API returned no telemetry · keys: " +
            Object.keys(data || {}).join(",")
        );
      }
      applyStepResult(step);
      setError(
        "Step OK · " +
          ((step.planner_plan && step.planner_plan.action) ||
            step.execution_status ||
            "done")
      );
      refreshMemory().catch(function () {});
      refreshHealth().catch(function () {});
    } catch (err) {
      setError("Step failed: " + (err && err.message ? err.message : String(err)));
    } finally {
      state.busy = false;
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Step Mission →";
      }
    }
  }

  async function onReset() {
    if (state.busy) return;
    persistForm();
    state.busy = true;
    try {
      await AeroAPI.reset();
      state.history = [];
      state.lastResult = null;
      renderKpis(null);
      renderSituation(null);
      renderPlanner(null);
      renderSafety(null, null);
      renderLogs(null);
      renderCharts();
      await refreshMemory();
      await refreshHealth();
      setError("Reset complete");
    } catch (err) {
      setError("Reset failed: " + (err && err.message ? err.message : String(err)));
    } finally {
      state.busy = false;
    }
  }

  function fillScenarios(list) {
    const sel = $("scenario");
    if (!sel) return;
    sel.innerHTML = "";
    (list || []).forEach(function (name) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
    if ([].some.call(sel.options, function (o) { return o.value === "GPS_INTERFERENCE"; })) {
      sel.value = "GPS_INTERFERENCE";
    }
  }

  async function init() {
    if (!window.AeroAPI) {
      setError("Scripts failed to load — hard-refresh Ctrl+Shift+R");
      return;
    }
    setError("Booting dashboard...");
    let cfg = {
      api_base_url: "/api",
      api_key: "",
      upstream_api_url: "",
      default_scenarios: [
        "NORMAL",
        "GPS_INTERFERENCE",
        "GPS_LOSS",
        "BATTERY_DEGRADATION",
        "LOW_BATTERY",
        "COMMUNICATION_LOSS",
        "STRONG_WIND",
        "SENSOR_ANOMALY",
        "COMBINED_FAILURE",
      ],
      poll_interval_ms: 10000,
    };
    try {
      cfg = Object.assign(cfg, await AeroAPI.loadBootstrapConfig());
    } catch (err) {
      setError("config.json failed: " + (err && err.message ? err.message : String(err)));
    }

    if ($("api-url")) $("api-url").value = AeroAPI.FIXED_BASE;
    if ($("api-key") && cfg.api_key) $("api-key").value = cfg.api_key;
    fillScenarios(cfg.default_scenarios);
    persistForm();

    if ($("btn-step")) $("btn-step").addEventListener("click", onStep);
    if ($("btn-reset")) $("btn-reset").addEventListener("click", onReset);
    if ($("api-key")) $("api-key").addEventListener("change", persistForm);

    setError(
      "Upstream " +
        (cfg.upstream_api_url || "unset") +
        " via /api · key " +
        ((cfg.api_key || "").slice(0, 8) || "(none)") +
        "…"
    );

    try {
      const h = await refreshHealth();
      const d = (h && h.dependencies) || {};
      setError(
        "Health " +
          (h && h.status) +
          " · mongo " +
          (d.mongo || "?") +
          " · neo4j " +
          (d.neo4j || "?") +
          " · groq " +
          (d.groq || "?") +
          " · vector " +
          ((h && h.vector_engine) || "?") +
          " · click Step Mission"
      );
    } catch (err) {
      setError("Health failed: " + (err && err.message ? err.message : String(err)));
    }

    try {
      await refreshMemory();
    } catch (err) {
      setError("Memory failed: " + (err && err.message ? err.message : String(err)));
    }

    state.pollTimer = setInterval(function () {
      refreshHealth().catch(function () {});
    }, cfg.poll_interval_ms || 10000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
