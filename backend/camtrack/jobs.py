"""Background job runner with progress reporting and cooperative cancellation."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    project_id: str
    status: str = "queued"          # queued / running / done / failed / cancelled
    stage: str = ""
    progress: float = 0.0           # 0..1 overall
    message: str = ""
    error: Optional[str] = None
    result: Any = None
    log: list = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self):
        return dict(id=self.id, kind=self.kind, project_id=self.project_id, status=self.status,
                    stage=self.stage, progress=round(self.progress, 4), message=self.message,
                    error=self.error, log=self.log[-60:], started=self.started, finished=self.finished,
                    elapsed=(self.finished or time.time()) - self.started)


class Reporter:
    """Handed to pipeline code. Maps stage-local progress into overall progress."""

    def __init__(self, job: Job):
        self.job = job
        self._lo, self._hi = 0.0, 1.0

    def stage(self, name: str, lo: float, hi: float):
        self.check()
        self._lo, self._hi = lo, hi
        self.job.stage = name
        self.job.progress = lo
        self.log(f"[{name}]")

    def __call__(self, frac: float, message: str = ""):
        self.check()
        frac = min(1.0, max(0.0, frac))
        self.job.progress = self._lo + (self._hi - self._lo) * frac
        if message:
            self.job.message = message

    def log(self, msg: str):
        self.job.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.job.message = msg

    def cancelled(self) -> bool:
        return self.job._cancel.is_set()

    def check(self):
        if self.job._cancel.is_set():
            raise Cancelled()


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, project_id: str, fn: Callable[[Reporter], Any]) -> Job:
        with self._lock:
            for j in self.jobs.values():
                if j.project_id == project_id and j.status in ("queued", "running") and j.kind != "ingest":
                    raise RuntimeError(f"A '{j.kind}' job is already running for this project")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, project_id=project_id)
            self.jobs[job.id] = job

        def run():
            job.status = "running"
            rep = Reporter(job)
            try:
                job.result = fn(rep)
                job.status = "done"
                job.progress = 1.0
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled by user"
            except InterruptedError:  # native code (pycolmap) honouring our cancellation token
                job.status = "cancelled" if job._cancel.is_set() else "failed"
                job.message = "Cancelled by user" if job._cancel.is_set() else "Interrupted"
            except Exception as e:  # surfaced to the UI verbatim
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.log.append(traceback.format_exc())
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{kind}-{job.id}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status not in ("queued", "running"):
            return False
        j._cancel.set()
        return True

    def active_for(self, project_id: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.project_id == project_id and j.status in ("queued", "running")]
