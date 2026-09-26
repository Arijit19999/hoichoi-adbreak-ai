"""In-process background jobs (one at a time: the free server has 0.1 CPU / 512 MB)."""

import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("jobs")


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"            # queued | running | done | error
    stage: str = ""
    log: list[str] = field(default_factory=list)
    video_id: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def submit(kind: str, label: str, fn: Callable[[Job], str | None]) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
    with _lock:
        _jobs[job.id] = job

    def run() -> None:
        job.status = "running"
        try:
            video_id = fn(job)
            job.video_id = video_id or job.video_id
            job.status = "done"
        except Exception as e:  # noqa: BLE001 - surface every failure to the UI
            log.error("job %s failed: %s", job.id, traceback.format_exc())
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
        finally:
            job.finished = time.time()

    _executor.submit(run)
    return job


def progress_for(job: Job) -> Callable[[str, str], None]:
    def progress(stage: str, message: str) -> None:
        job.stage = stage
        job.log.append(f"{time.strftime('%H:%M:%S')} {stage}: {message}")
        log.info("[%s] %s: %s", job.id, stage, message)
    return progress


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def all_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.created, reverse=True)


BACKEND_DIR = Path(__file__).resolve().parents[1]
_PROGRESS_LINE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\S+)\s+(.*)$")


def run_cli(job: Job, args: list[str]) -> str:
    """Run the pipeline CLI in a low-priority child process; returns the video id.

    The web server stays responsive (health checks) on a tiny CPU share, and everything the
    pipeline allocated is returned to the OS when the child exits.
    """
    cmd = [sys.executable, "-m", "app.pipeline.run", *args]
    proc = subprocess.Popen(
        cmd, cwd=BACKEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        preexec_fn=(lambda: os.nice(10)) if os.name != "nt" else None,
    )
    tail: deque[str] = deque(maxlen=25)
    work = None
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        match = _PROGRESS_LINE.match(line)
        if match:
            clock, stage, message = match.groups()
            job.stage = stage
            job.log.append(f"{clock} {stage}: {message}")
            if stage == "done":
                work = message.strip()
    code = proc.wait()
    if code != 0 or work is None:
        errors = [t for t in tail if "Error" in t or "error" in t] or list(tail)[-3:]
        raise RuntimeError(f"pipeline exited with code {code}: {' | '.join(errors)[-500:]}")
    return Path(work).name
