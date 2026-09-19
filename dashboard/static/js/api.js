/** REST client for AeroCortex telemetry API (same-origin /api proxy). */
(function (global) {
  const STORAGE_KEY = "aerocortex.apiKey";
  const STORAGE_URL = "aerocortex.apiBaseUrl";
  const FIXED_BASE = "/api";

  function normalizeBase(url) {
    const raw = String(url || "").trim().replace(/\/$/, "");
    // Always prefer same-origin proxy. Absolute localhost / Render URLs hang or CORS-fail
    // when the dashboard SPA is served from :8501.
    if (!raw || raw === FIXED_BASE) return FIXED_BASE;
    if (/^https?:\/\//i.test(raw)) return FIXED_BASE;
    if (raw.indexOf("127.0.0.1") >= 0 || raw.indexOf("localhost") >= 0) return FIXED_BASE;
    if (raw.charAt(0) !== "/") return FIXED_BASE;
    return raw;
  }

  function getSettings() {
    let apiKey = "";
    try {
      apiKey = localStorage.getItem(STORAGE_KEY) || "";
      // Migrate / clear bad absolute URLs left from earlier builds.
      const stored = localStorage.getItem(STORAGE_URL) || "";
      if (stored && normalizeBase(stored) === FIXED_BASE && stored !== FIXED_BASE) {
        localStorage.setItem(STORAGE_URL, FIXED_BASE);
      }
    } catch (_) {
      /* private mode / blocked storage */
    }
    return { baseUrl: FIXED_BASE, apiKey: apiKey };
  }

  function saveSettings(baseUrl, apiKey) {
    try {
      localStorage.setItem(STORAGE_URL, FIXED_BASE);
      localStorage.setItem(STORAGE_KEY, apiKey || "");
    } catch (_) {
      /* ignore */
    }
  }

  async function request(path, options) {
    options = options || {};
    const { baseUrl, apiKey } = getSettings();
    const headers = Object.assign(
      { Accept: "application/json" },
      options.headers || {}
    );
    if (apiKey) {
      headers["X-API-Key"] = apiKey;
    }
    if (options.body && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const timeoutMs = options.timeoutMs || 90000;
    const controller = new AbortController();
    const timer = setTimeout(function () {
      controller.abort();
    }, timeoutMs);
    let res;
    try {
      res = await fetch(baseUrl + path, {
        method: options.method || "GET",
        headers: headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
        cache: "no-store",
      });
    } catch (err) {
      if (err && err.name === "AbortError") {
        throw new Error("Request timed out after " + timeoutMs / 1000 + "s (" + baseUrl + path + ")");
      }
      throw new Error(
        (err && err.message ? err.message : "Network error") +
          " · " +
          baseUrl +
          path
      );
    } finally {
      clearTimeout(timer);
    }
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = { raw: text };
    }
    if (!res.ok) {
      const detail =
        (data && (data.detail || data.message)) || res.statusText || "request failed";
      const err = new Error(
        typeof detail === "string" ? detail : JSON.stringify(detail)
      );
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  async function loadBootstrapConfig() {
    const res = await fetch("/config.json", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error("Failed to load dashboard config (" + res.status + ")");
    }
    return res.json();
  }

  global.AeroAPI = {
    FIXED_BASE: FIXED_BASE,
    getSettings: getSettings,
    saveSettings: saveSettings,
    normalizeBase: normalizeBase,
    loadBootstrapConfig: loadBootstrapConfig,
    healthz: function () {
      return request("/healthz", { timeoutMs: 45000 });
    },
    memory: function () {
      return request("/memory", { timeoutMs: 45000 });
    },
    status: function () {
      return request("/status", { timeoutMs: 30000 });
    },
    simulate: function (scenario, steps) {
      if (steps === undefined) steps = 1;
      return request("/simulate", {
        method: "POST",
        body: { scenario: scenario, steps: steps, inject_step: 1 },
        timeoutMs: 120000,
      });
    },
    reset: function () {
      return request("/reset", { method: "POST", timeoutMs: 30000 });
    },
  };
})(window);
