# hoichoi AdBreak AI

Context-aware video segmentation and intelligent ad placement for long-form Bengali drama —
built for hoichoi Hackathon'26 (Problem 1).

Video in → scenes segmented → break candidates scored → brand matched → VMAP manifest + debug JSON + playable demo.

> Live demo is hosted on Render's free tier: the first load after idle can take ~1 minute while the server wakes up.

## Local development

```powershell
# backend (http://localhost:8000)
cd backend
uv run uvicorn app.main:app --reload --port 8000

# frontend (http://localhost:5173, proxies /api to the backend)
cd frontend
npm run dev
```

Copy `.env.example` to `.env` at the repo root and fill in the API keys.
Check them with `uv run --directory backend python ../scripts/check_keys.py`.

## Brand catalogue

`data/brands/brands.json` holds the brand catalogue (synthetic brands supplied with the problem statement).
Brands are matched purely from their `target_contexts` / `negative_contexts` text, so a new brand
added to this file needs no code changes.
