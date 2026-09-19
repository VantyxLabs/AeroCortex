# Deploy AeroCortex dashboard to Vercel

Vercel hosts the **UI only**. The Python API stays on **Render**. Same-origin `/api` on Vercel is rewritten to your Render service (no CORS hassle).

Root `package.json` + `.vercelignore` force a **static** build so Vercel does not bundle the Python `api/` package as a serverless function.

```
Browser → Vercel (static SPA) → Render (FastAPI + Mongo/Neo4j/Groq/Pinecone)
```

## Steps

### 1. Push the repo to GitHub

Include `vercel.json`, `scripts/vercel-prepare.js`, and `.vercelignore`.

### 2. Import on Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import the AeroCortex repo
3. Framework: **Other**
4. Root Directory: *(leave empty)*
5. Confirm build uses:
   - **Build Command:** `node scripts/vercel-prepare.js`
   - **Output Directory:** `public`

### 3. Add environment variables

Vercel → Project → **Settings** → **Environment Variables** (Production):

| Name | Example |
|------|---------|
| `AEROCORTEX_API_URL` | `https://aerocortex.onrender.com` |
| `AEROCORTEX_API_KEY` | *(same value as Render `API_KEY`)* |

These are baked into `public/config.json` at **build** time. After changing them, **Redeploy**.

### 4. Point rewrites at your Render host

`vercel.json` already proxies `/api/*` → `https://aerocortex.onrender.com`. If your Render URL differs, edit every `destination` in `vercel.json` and redeploy.

### 5. Deploy

Click **Deploy**. Open the `*.vercel.app` URL → hard-refresh → **Step Mission**.

First request after Render sleep can take 30–60s (free tier cold start).

## CLI (optional)

```bash
npm i -g vercel
cd AeroCortex
vercel login
vercel link
vercel env add AEROCORTEX_API_URL
vercel env add AEROCORTEX_API_KEY
vercel --prod
```

## Local prepare smoke test

```bash
set AEROCORTEX_API_URL=https://aerocortex.onrender.com
set AEROCORTEX_API_KEY=your-key
node scripts/vercel-prepare.js
```

Creates `./public` (gitignored).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| 401 on Step | `AEROCORTEX_API_KEY` ≠ Render `API_KEY` → fix env, redeploy |
| 502 / timeout | Render cold start or wrong `AEROCORTEX_API_URL` / rewrite host |
| Health stays `—` | DevTools → `/api/healthz` and `/config.json` |
| Wrong API host | Update `vercel.json` destinations + `AEROCORTEX_API_URL` |

## What not to deploy on Vercel

The FastAPI app (`main.py --api`), Mongo, Neo4j, Chroma, and long mission runs belong on **Render/Docker**, not Vercel serverless.
