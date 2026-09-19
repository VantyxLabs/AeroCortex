/**
 * Copy dashboard/static → public and write config.json from Vercel env.
 * Env (available at build time):
 *   AEROCORTEX_API_URL  — Render origin (default https://aerocortex.onrender.com)
 *   AEROCORTEX_API_KEY  — same as Render API_KEY
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src = path.join(root, "dashboard", "static");
const dest = path.join(root, "public");

function rm(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) rm(p);
    else fs.unlinkSync(p);
  }
  fs.rmdirSync(dir);
}

function copy(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const a = path.join(from, entry.name);
    const b = path.join(to, entry.name);
    if (entry.isDirectory()) copy(a, b);
    else fs.copyFileSync(a, b);
  }
}

if (!fs.existsSync(src)) {
  console.error("Missing dashboard/static — cannot prepare Vercel public/");
  process.exit(1);
}

if (fs.existsSync(dest)) rm(dest);
copy(src, dest);

const upstream = (
  process.env.AEROCORTEX_API_URL ||
  process.env.DASHBOARD_API_URL ||
  "https://aerocortex.onrender.com"
).replace(/\/$/, "");

const config = {
  api_base_url: "/api",
  upstream_api_url: upstream,
  api_key: process.env.AEROCORTEX_API_KEY || process.env.API_KEY || "",
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
  poll_interval_ms: 30000,
  version: "1.0.0",
};

fs.writeFileSync(path.join(dest, "config.json"), JSON.stringify(config, null, 2));
console.log("vercel-prepare: public/ ready · upstream=" + upstream);
