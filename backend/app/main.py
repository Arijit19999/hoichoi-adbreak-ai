import json
import logging
import shutil
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ads, brands, jobs
from .config import REPO_ROOT, get_settings
from .pipeline import breaks, run, vmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

app = FastAPI(title="hoichoi AdBreak AI")
settings = get_settings()
OUTPUTS = settings.outputs
UPLOADS = OUTPUTS / "_uploads"
MAX_UPLOAD_BYTES = 2 * 1024**3


def _work(video_id: str) -> Path:
    work = OUTPUTS / video_id
    if not video_id.isalnum() or not (work / "meta.json").exists():
        raise HTTPException(404, "unknown video")
    return work


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------- health / jobs ----------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict() for j in jobs.all_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.to_dict()


# ---------- videos ----------

def _process(src: Path, label: str, cleanup: bool) -> jobs.Job:
    def work(job: jobs.Job) -> str:
        try:
            result = run.run_pipeline(src, progress=jobs.progress_for(job))
            return result.name
        finally:
            if cleanup:
                src.unlink(missing_ok=True)
    return jobs.submit("process", label, work)


@app.post("/api/videos")
async def upload_video(file: UploadFile):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename or 'video.mp4').suffix or '.mp4'}"
    size = 0
    with dst.open("wb") as out:
        while chunk := await file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, "video larger than 2 GB")
            out.write(chunk)
    return _process(dst, file.filename or dst.name, cleanup=True).to_dict()


class UrlIn(BaseModel):
    url: str


@app.post("/api/videos/from-url")
def video_from_url(body: UrlIn):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS / f"{uuid.uuid4().hex}.mp4"

    def work(job: jobs.Job) -> str:
        progress = jobs.progress_for(job)
        progress("download", body.url)
        try:
            if "drive.google.com" in body.url:
                import gdown
                if gdown.download(body.url, str(dst), quiet=True, fuzzy=True) is None:
                    raise RuntimeError("Google Drive download failed (is the file shared publicly?)")
            else:
                with httpx.stream("GET", body.url, follow_redirects=True, timeout=60) as r:
                    r.raise_for_status()
                    with dst.open("wb") as out:
                        for chunk in r.iter_bytes(4 * 1024 * 1024):
                            out.write(chunk)
            progress("download", f"{dst.stat().st_size / 1e6:.0f} MB")
            return run.run_pipeline(dst, progress=progress).name
        finally:
            dst.unlink(missing_ok=True)

    return jobs.submit("process", body.url, work).to_dict()


@app.get("/api/videos")
def list_videos():
    videos = []
    for meta_path in OUTPUTS.glob("*/meta.json"):
        work = meta_path.parent
        meta, plan = _read(meta_path), _read(work / "breaks.json")
        scenes = _read(work / "scenes.json")
        videos.append({
            **meta,
            "ready": plan is not None,
            "scenes": len(scenes["scenes"]) if scenes else None,
            "breaks": len(plan["breaks"]) if plan else None,
            "modified": (work / "breaks.json").stat().st_mtime if plan else meta_path.stat().st_mtime,
        })
    return sorted(videos, key=lambda v: v["modified"], reverse=True)


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    work = _work(video_id)
    return {
        "meta": _read(work / "meta.json"),
        "plan": _read(work / "breaks.json"),
        "video_url": f"media/{video_id}/video.mp4",
        "manifest_url": f"api/videos/{video_id}/manifest.vmap",
        "debug_url": f"api/videos/{video_id}/debug.json",
    }


@app.get("/api/videos/{video_id}/debug.json")
def get_debug(video_id: str):
    path = _work(video_id) / "debug.json"
    if not path.exists():
        raise HTTPException(404, "not planned yet")
    return FileResponse(path, media_type="application/json")


@app.get("/api/videos/{video_id}/scenes.json")
def get_scenes(video_id: str):
    path = _work(video_id) / "scenes.json"
    if not path.exists():
        raise HTTPException(404, "not analysed yet")
    return FileResponse(path, media_type="application/json")


@app.get("/api/videos/{video_id}/manifest.vmap")
def get_manifest(video_id: str, request: Request):
    plan = _read(_work(video_id) / "breaks.json")
    if plan is None:
        raise HTTPException(404, "not planned yet")
    xml = vmap.build_vmap(plan, str(request.base_url))
    return Response(xml, media_type="application/xml",
                    headers={"Content-Disposition": f'inline; filename="{video_id}.vmap.xml"'})


class RulesIn(BaseModel):
    max_breaks_per_hour: float = Field(6.0, gt=0, le=30)
    min_gap_s: float = Field(300.0, ge=0)
    max_ad_load_pct: float = Field(10.0, gt=0, le=50)
    no_break_before_s: float = Field(180.0, ge=0)
    no_break_in_last_s: float = Field(90.0, ge=0)
    min_break_score: float = Field(0.45, ge=0, le=1)


@app.get("/api/rules")
def default_rules():
    return breaks.PacingRules().to_dict()


@app.post("/api/videos/{video_id}/replan")
def replan(video_id: str, rules: RulesIn | None = None):
    work = _work(video_id)
    if not (work / "scenes.json").exists():
        raise HTTPException(409, "video is still being analysed")
    pacing = breaks.PacingRules(**(rules.model_dump() if rules else {}))

    def job_fn(job: jobs.Job) -> str:
        run.make_plan(work, jobs.progress_for(job), rules=pacing)
        return video_id

    job = jobs.submit("replan", video_id, job_fn)
    job.video_id = video_id
    return job.to_dict()


@app.get("/api/track")
def track():
    return Response(status_code=204)


# ---------- brand catalogue ----------

class CreativeIn(BaseModel):
    id: str
    duration_sec: int = Field(gt=0, le=120)
    language: str = "bn"
    url: str


class BrandIn(BaseModel):
    brand_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    display_name: str
    category: str
    target_contexts: list[str] = Field(min_length=1)
    negative_contexts: list[str] = Field(default_factory=list)
    creatives: list[CreativeIn] = Field(default_factory=list)


def _catalogue_path() -> Path:
    return settings.resolve(settings.brands_path)


@app.get("/api/brands")
def get_brands():
    return json.loads(_catalogue_path().read_text(encoding="utf-8"))


@app.post("/api/brands")
def upsert_brand(brand: BrandIn):
    data = brand.model_dump()
    if not data["creatives"]:
        data["creatives"] = [
            {"id": f"{brand.brand_id}_{d}s", "duration_sec": d, "language": "bn",
             "url": f"ads/{brand.brand_id}/{brand.brand_id}_{d}s.mp4"}
            for d in (15, 30)
        ]
    for creative in data["creatives"]:
        if ".." in creative["url"] or creative["url"].startswith(("/", "\\")):
            raise HTTPException(422, f"invalid creative url {creative['url']}")
    catalogue = [b for b in get_brands() if b["brand_id"] != brand.brand_id] + [data]
    _catalogue_path().write_text(json.dumps(catalogue, ensure_ascii=False, indent=2), encoding="utf-8")
    ads.ensure_creatives(brands.load_catalogue())
    return data


@app.delete("/api/brands/{brand_id}")
def delete_brand(brand_id: str):
    catalogue = get_brands()
    remaining = [b for b in catalogue if b["brand_id"] != brand_id]
    if len(remaining) == len(catalogue):
        raise HTTPException(404, "unknown brand")
    _catalogue_path().write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"deleted": brand_id}


# ---------- static: media, ad creatives, web app ----------

app.mount("/media", StaticFiles(directory=OUTPUTS), name="media")
(REPO_ROOT / "data" / "ads").mkdir(parents=True, exist_ok=True)
app.mount("/ads", StaticFiles(directory=REPO_ROOT / "data" / "ads"), name="ads")

dist = REPO_ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")


@app.on_event("shutdown")
def _cleanup_uploads() -> None:
    shutil.rmtree(UPLOADS, ignore_errors=True)
