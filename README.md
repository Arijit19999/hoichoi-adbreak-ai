# AdBreak AI — context-aware ad breaks for Bengali drama

hoichoi Hackathon'26 · Problem 1: Context-Aware Video Segmentation & Intelligent Ad Placement

**Video in → scenes segmented → break candidates scored → brand matched → VMAP manifest + debug JSON + a player that
cuts to the ad and resumes.**

> Live demo runs on Render's free tier: the first load after it has been idle can take ~1 minute to wake up.
> Processing a new 20–40 min episode takes a few minutes (progress is shown in the sidebar).

## What it decides

| Question | How it is answered |
|---|---|
| **Where** is a natural, non-jarring cut? | Scene boundaries from a multimodal model (picture + Bengali audio), snapped to the exact shot cut / fade-to-black by frame differencing, and only where nobody is speaking: neural VAD over the whole episode plus word-timestamped ASR at each candidate cut. |
| **Whether** a break is warranted? | Each surviving boundary gets a 0–1 score (AI "natural pause" judgement, scene ending, transition type, silence, audio dip). A small dynamic program then picks the best set of breaks under the pacing rules: max breaks/hour, minimum gap, max ad-load %, no breaks at the very start/end. |
| **What** brand belongs in the slot? | Brand safety first, enforced in code; then fit driven by the scene's *dominant* activity; then an independent audit by a second model. |

## Pipeline

```
video.mp4
 ├─ ingest      content hash (cache key) · playback copy · 16 kHz audio           ffmpeg
 ├─ speech      speech/silence timeline + loudness over the full episode         Silero VAD (ONNX)
 ├─ scenes      6-min overlapping windows → scenes + analysis per scene           Gemini (video + audio)
 │               · summary, dialogue gist, setting, dominant activity, mood
 │               · explicit absent/possible/present verdict for EVERY negative context in the catalogue
 │               · ending type + "interruptibility" 0-10
 │              boundaries merged across windows, snapped to the real cut        frame differencing
 ├─ candidates  hard gates: real cut · no speech 0.4 s before / 0.15 s after (VAD) · no word crossing the cut (ASR)
 ├─ safety      per break × brand, in code (see below); unseen contexts checked from scene text
 ├─ fit         dominant-activity fit per safe brand (LLM 0-10 + keyword overlap)
 ├─ select      DP over candidates maximising break score + 0.3 × best safe fit under pacing rules
 ├─ audit       different model re-checks each placement; flagged → next brand or drop the break
 └─ output      manifest.vmap.xml (IAB VMAP 1.0 + inline VAST 3.0) · debug.json · breaks.json
```

Every stage writes a JSON artifact to `outputs/<content-hash>/` and is skipped when it already exists, so a run
interrupted by an API limit resumes where it stopped. Results are keyed by the **bytes** of the video, never its
file name: nothing is special-cased for the sample episodes.

## Brand safety (negative contexts are a hard block)

A brand is blocked at a break when, for any of its `negative_contexts`:

* the scene **before** the break has it `present` **or** `possible`, or lists it among its sensitive events
* the scene **after** the break has it `present`

Scene verdicts are the *worst* verdict across every analysis window that overlaps the scene. If the model skips a
context, it counts as `possible`. Contexts the video analysis never checked (a brand added later) are checked from
the stored scene description by a text model, and anything it cannot rule out counts as `possible`. A break with no
safe brand is not placed. After placement, an **independent auditor on a different model** re-checks each
break/brand pair; a flagged placement falls back to the next safe brand or the break is dropped. If the audit
itself cannot run, it counts as a violation.

## New (9th) brand: zero code changes

Brands are data. Add an entry with the same schema to `data/brands/brands.json`, or use the **Brands** tab in the
web app, then **Re-plan**. Matching only reads `target_contexts` / `negative_contexts` / `category`; missing creative
files get a generated placeholder video. The catalogue shipped with the problem statement is used as-is
(synthetic "Brand A"–"Brand H").

## Models

| Task | Model | Why |
|---|---|---|
| Scene segmentation + understanding | Gemini `gemini-3.8-flash`, falls back to `gemini-flash-lite-latest` | Only option that watches video **and** hears Bengali audio together |
| Brand fit, unseen-context checks | Groq `openai/gpt-oss-120b` | Fast, free-tier text reasoning |
| Independent audit | Groq `qwen/qwen3.8-27b` | A different model family from the matcher |
| Cut-time speech check | Groq `whisper-large-v3` | Word timestamps around each candidate cut |
| Speech/silence timeline | Silero VAD (ONNX, via faster-whisper) | Runs on CPU, no GPU needed |

Every model call retries with backoff, remembers daily-quota exhaustion and moves to the next model, so a
rate-limited provider degrades quality instead of breaking the demo.

## Outputs

* `GET /api/videos/{id}/manifest.vmap` — VMAP 1.0; each `AdBreak` has an inline VAST 3.0 linear ad with the chosen
  creative, plus an extension with the break score and scenes.
* `GET /api/videos/{id}/debug.json` — every scene, every candidate with its features, rejection reasons, per-brand
  safety verdicts, fit ranking and audit results, and the final plan.
* The web player loads the **VMAP itself** (not the debug JSON): it stops one frame before the cut, plays the ad,
  and resumes at the new scene. Seeking over an unplayed break plays it first.

## Run locally

```powershell
# repo root: copy .env.example to .env and fill in GEMINI_API_KEY and GROQ_API_KEY
uv run --directory backend python ../scripts/check_keys.py          # verify keys + ffmpeg

# CLI: full pipeline on one file (results in outputs/<hash>/)
cd backend
uv run python -m app.pipeline.run ../data/samples/mohanagar.mp4
uv run python -m app.pipeline.run ../data/samples/mohanagar.mp4 --replan   # breaks/brands/manifest only

# web app
uv run uvicorn app.main:app --reload --port 8000        # API + built frontend
cd ../frontend && npm install && npm run dev            # dev UI on :5173 (proxies to :8000)
```

Requirements: Python 3.12 + uv, Node 22, ffmpeg on PATH.

### Self-check against the judging rules

```powershell
cd backend
uv run python -m app.verify        # every processed video; exit code 1 on any failure
```

No model calls: for every placed break it re-measures the shot change from the frames, checks nobody is speaking
at the cut, re-applies the brand-safety rules, checks pacing / ad load, parses the VMAP against the plan, and checks
every creative file's length.

On a slow uplink, set `GEMINI_PROXY_HEIGHT=360` to upload a small same-timeline copy to Gemini instead of the
original (4-5x smaller).

## Deploy

`render.yaml` + `Dockerfile` (single container: React build served by FastAPI). Set `GEMINI_API_KEY` and
`GROQ_API_KEY` as secrets. The free instance has an ephemeral disk, so processed results disappear on restart:
`SEED_VIDEO_URLS` (comma-separated links) are re-processed in the background by the full pipeline after each start,
one at a time, and skipped when already present. An external uptime ping on `/api/health` keeps the free instance
from sleeping.

## Known limitations

* Free-tier API quotas: when the primary Gemini model's daily quota is used up, the lighter fallback model is used
  (lower scene quality, same safety guarantees).
* Whisper's Bengali transcripts are rough; they are used only for word timing at the cut, never for meaning.
* One creative per break (no multi-ad pods); ad creatives are generated placeholders because the catalogue's video
  files were not supplied.
