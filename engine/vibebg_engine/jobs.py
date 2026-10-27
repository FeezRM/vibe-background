"""Background export jobs with progress and cancellation."""

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from vibebg_engine.pipeline import Cancelled, Progress, RenderSettings, render


@dataclass
class Job:
    id: str
    src: Path
    out: Path
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "render"  # "relight" while AI Relight keyframes are prepared, then "render"
    frame: int = 0
    total: int = 0
    elapsed: float = 0.0
    eta: float | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {"id": self.id, "src": str(self.src), "out": str(self.out),
                "status": self.status, "stage": self.stage, "frame": self.frame,
                "total": self.total,
                "elapsed": self.elapsed, "eta": self.eta, "error": self.error}


class JobManager:
    """Runs one export at a time, in submission order."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._gpu = threading.Lock()

    def submit(self, src: Path, out: Path, settings: RenderSettings, start: float = 0.0,
               duration: float | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], src=src, out=out)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, settings, start, duration),
                         daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is not None:
            job.cancel.set()
        return job

    def _run(self, job: Job, settings: RenderSettings, start: float, duration: float | None):
        def on_progress(p: Progress):
            job.frame, job.total, job.elapsed, job.eta = p.frame, p.total, p.elapsed, p.eta
            job.stage = p.stage

        with self._gpu:
            if job.cancel.is_set():
                job.status = "cancelled"
                return
            job.status = "running"
            try:
                render(job.src, job.out, settings, on_progress=on_progress, cancel=job.cancel,
                       start=start, duration=duration)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - thread boundary; reported via job status
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
