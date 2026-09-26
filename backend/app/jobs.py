"""In-process background jobs (one at a time: the free server has 0.1 CPU / 512 MB)."""

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

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
